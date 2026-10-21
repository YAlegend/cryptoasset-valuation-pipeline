"""Week-t signals against the week t -> t+1 return (design 4-5).

A week is the observation dated Sunday, on a complete Sunday grid: an absent Sunday is a
NaN cell, so a shift never reaches across it into the week before.

A Sunday price more than ``spike_ratio`` times BOTH its neighbours (or below 1/ratio of both)
is a vendor spike and is nulled: on 2026-01-18 BTG and FUN read 10x for nine days, which
alone tripled the MVRV decile spread. A jump that stays is a real move and is kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.factors.settings import FactorConfig

SIGNALS = ("size", "momentum", "value", "mvrv")
PANEL_COLUMNS = ["week", "asset", "market_cap_usd", "ret", *SIGNALS]


def weekly_panel(daily: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    dates = pd.to_datetime(daily["metric_date"])
    keep = (dates.dt.dayofweek == 6) & ~daily["asset"].isin(set(config.exclude))
    sundays = daily[keep].assign(metric_date=dates[keep])
    if sundays.empty:
        out = pd.DataFrame(columns=PANEL_COLUMNS)
        out.attrs["spikes_nulled"] = 0
        return out
    grid = pd.date_range(sundays["metric_date"].min(), sundays["metric_date"].max(), freq="W-SUN")

    def wide(column: str) -> pd.DataFrame:
        return sundays.pivot(index="metric_date", columns="asset", values=column).reindex(grid)

    price, cap = wide("price_usd"), wide("market_cap_usd")
    up, down, k = price / price.shift(1), price / price.shift(-1), config.spike_ratio
    spikes = ((up > k) & (down > k)) | ((up < 1 / k) & (down < 1 / k))
    price = price.mask(spikes)
    columns = {
        "market_cap_usd": cap,
        "ret": price.shift(-1) / price - 1,
        "size": np.log(cap),
        "momentum": price / price.shift(config.momentum_weeks) - 1,
        "value": wide("active_addresses") / cap,
        "mvrv": wide("mvrv"),
    }
    long = pd.concat({name: frame.stack() for name, frame in columns.items()}, axis=1)
    long = long.rename_axis(["week", "asset"]).reset_index()
    eligible = long[(long["market_cap_usd"] >= config.min_market_cap_usd) & long["ret"].notna()]
    counts = eligible.groupby("week")["asset"].transform("size")
    out = eligible[counts >= config.min_assets].reset_index(drop=True)[PANEL_COLUMNS]
    out.attrs["spikes_nulled"] = int(spikes.to_numpy().sum())
    return out
