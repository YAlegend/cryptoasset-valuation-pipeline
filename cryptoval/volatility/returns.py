"""Percent log returns (design 3), and the calendar each pair is measured on.

Crypto trades every day, so a crypto series and a crypto-crypto pair are guarded for a
contiguous daily grid. A crypto-traditional pair is measured on the TRADITIONAL calendar:
the crypto price is sampled on the equity's trading dates, and both returns span the same
gaps (a Monday return covers Friday to Monday for both legs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.econometrics.base import Unusable
from cryptoval.scoring.targets import _require_contiguous_days

# Two years, the anchoring chain's floor. Measured on the 2026-10-18 vintage: SOL (362
# returns) and TRX (363) fall below it; every other non-stablecoin clears it.
MIN_RETURNS = 730


def price_series(panel: pd.DataFrame, symbol: str) -> pd.Series:
    """Positive prices indexed by date. NULL and non-positive prices are dropped, so an
    asset whose history starts late begins at its first price rather than at a gap."""
    rows = panel[panel["symbol"] == symbol]
    if rows.empty:
        raise Unusable(f"{symbol} is not in the panel")
    values = pd.to_numeric(rows["price_usd"], errors="coerce").astype("float64").to_numpy()
    series = pd.Series(values, index=pd.DatetimeIndex(pd.to_datetime(rows["metric_date"])))
    series = series.sort_index()
    duplicated = series.index[series.index.duplicated()]
    if len(duplicated):
        raise Unusable(
            f"{symbol} has {len(duplicated)} duplicated date(s), first "
            f"{duplicated[0]:%Y-%m-%d}; a repeated day would be read as a zero return"
        )
    series = series[series > 0.0]
    if series.empty:
        raise Unusable(f"{symbol} has no positive price")
    return series


def _contiguous(index: pd.DatetimeIndex, label: str) -> None:
    try:
        _require_contiguous_days(pd.DataFrame({"symbol": label, "metric_date": index}))
    except ValueError as exc:
        raise Unusable(str(exc)) from exc


def _enough(n: int, label: str) -> None:
    if n < MIN_RETURNS:
        raise Unusable(
            f"{label}: {n} returns, {MIN_RETURNS} needed for a volatility model on daily data"
        )


def crypto_returns(prices: pd.Series, symbol: str) -> pd.Series:
    _contiguous(prices.index, symbol)
    returns = 100.0 * np.log(prices).diff().dropna()
    _enough(len(returns), symbol)
    return returns


def pair_returns(
    left: pd.Series, right: pd.Series, label: str, *, trading_calendar: bool
) -> pd.DataFrame:
    """Aligned percent log returns, columns ``a`` (left) and ``b`` (right)."""
    if trading_calendar:
        inside = (right.index >= left.index.min()) & (right.index <= left.index.max())
        dates = right.index[inside]
        sampled = left.reindex(dates)
        missing = sampled.index[sampled.isna()]
        if len(missing):
            raise Unusable(
                f"{label}: no crypto price on {len(missing)} trading date(s), "
                f"first {missing[0]:%Y-%m-%d}"
            )
        prices = pd.DataFrame({"a": sampled, "b": right.reindex(dates)})
    else:
        prices = pd.concat({"a": left, "b": right}, axis=1, join="inner")
        _contiguous(prices.index, label)
    returns = 100.0 * np.log(prices).diff().dropna()
    _enough(len(returns), label)
    return returns
