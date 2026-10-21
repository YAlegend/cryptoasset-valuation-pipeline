"""One long table for spec 11.5, at grain (model_id, signal, portfolio, metric).

ONE Holm family of 13: four decile HL means, three LTW factor means, two factor alphas, four
Fama-MacBeth slopes. A test that cannot run is a refused ROW carrying the reason, never an
absent one (the fct_volatility_test contract).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pandas as pd

from cryptoval.econometrics.base import Unusable
from cryptoval.factors.inference import fama_macbeth, nw_alpha, nw_mean
from cryptoval.factors.panel import SIGNALS, weekly_panel
from cryptoval.factors.portfolios import decile_returns, ltw_factors
from cryptoval.factors.settings import FactorConfig
from cryptoval.hashing import stable_hash
from cryptoval.scoring.multiplicity import correct_family_wise

FACTOR_COLUMNS = [
    "model_id",
    "signal",
    "portfolio",
    "metric",
    "value",
    "n_weeks",
    "reason",
    "params_hash",
    "panel_vintage_ts",
]
RETURN_COLUMNS = ["week", "series", "value"]
COUNT_KEYS = {"model_id": "*", "signal": "*"}
ALPHA_SIGNALS = ("value", "mvrv")  # size and momentum ARE factors, so they get no alpha


def run_factors(
    daily: pd.DataFrame, config: FactorConfig, panel_vintage_ts: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = {
        "params_hash": stable_hash(dataclasses.asdict(config)),
        "panel_vintage_ts": panel_vintage_ts,
    }
    rows: list[dict[str, Any]] = []
    series: dict[str, pd.Series] = {}

    def row(**values: Any) -> None:
        out = dict.fromkeys(FACTOR_COLUMNS)
        out.update(common, **values)
        rows.append(out)

    def test(model_id: str, signal: str, portfolio: str | None, compute: Callable[[], dict]):
        try:
            result = compute()
        except Unusable as exc:
            row(
                model_id=model_id,
                signal=signal,
                portfolio=portfolio,
                metric="refused",
                reason=str(exc),
            )
            return
        n = result.pop("n_weeks")
        for metric, value in result.items():
            row(
                model_id=model_id,
                signal=signal,
                portfolio=portfolio,
                metric=metric,
                value=value,
                n_weeks=n,
            )

    lags = config.newey_west_lags
    panel = weekly_panel(daily, config)
    deciles = {signal: decile_returns(panel, signal, config) for signal in SIGNALS}
    for signal, table in deciles.items():
        n = float(len(table))
        for name in table.columns.drop(["HL", "n_assets"]):
            row(
                model_id="decile",
                signal=signal,
                portfolio=name,
                metric="mean_return",
                value=float(table[name].mean()) if n else None,
                n_weeks=n,
            )
        row(
            model_id="decile",
            signal=signal,
            metric="mean_assets",
            value=float(table["n_assets"].mean()) if n else None,
            n_weeks=n,
        )
        test("decile", signal, "HL", lambda t=table: nw_mean(t["HL"], lags))
        series[f"HL_{signal}"] = table["HL"]

    factors = ltw_factors(panel, config)
    for name in factors.columns:
        test("factor", name, None, lambda f=factors[name]: nw_mean(f, lags))
        series[name] = factors[name]
    for signal in ALPHA_SIGNALS:
        test("alpha", signal, "HL", lambda s=signal: nw_alpha(deciles[s]["HL"], factors, lags))

    slopes = fama_macbeth(panel, SIGNALS, config)
    for signal in SIGNALS:
        test("fama_macbeth", signal, None, lambda s=signal: nw_mean(slopes[s], lags))

    table = correct_family_wise(
        pd.DataFrame(rows, columns=FACTOR_COLUMNS),
        p_metric="p_value",
        columns=FACTOR_COLUMNS,
        count_keys={**COUNT_KEYS, **common},
    )
    frames = [
        s.dropna().rename("value").rename_axis("week").reset_index().assign(series=name)
        for name, s in series.items()
    ]
    returns = pd.concat(frames, ignore_index=True).reindex(columns=RETURN_COLUMNS)
    return table, returns
