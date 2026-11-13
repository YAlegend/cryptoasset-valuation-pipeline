"""From the creation x spend matrix to one row per day (spec 6.3, 10.6(b); Plan 3b section 5).

unspent(c, d) = created(c) - sum_{s <= d} spent(c, s): nothing about unspent outputs is ever
queried. The loop walks spend days once with a running vector, so no N x N matrix is built.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cryptoval.utxo.extract import GENESIS

SAT = 100_000_000
HALVING = 210_000
STH_DAYS = 155
BANDS = (
    ("0d", 0, 1),
    ("1d_1w", 1, 7),
    ("1w_1m", 7, 30),
    ("1m_3m", 30, 90),
    ("3m_6m", 90, 180),
    ("6m_1y", 180, 365),
    ("1y_2y", 365, 730),
    ("2y_3y", 730, 1095),
    ("3y_5y", 1095, 1825),
    ("5y_7y", 1825, 2555),
    ("7y_10y", 2555, 3650),
    ("10y_plus", 3650, None),
)
COINTIME_COLUMNS = [
    "metric_date",
    "coinblocks_created",
    "coinblocks_destroyed",
    "liveliness",
    "vaultedness",
    "utxo_supply",
    "active_supply",
    "vaulted_supply",
    "utxo_realized_cap",
    *(f"hodl_{b}" for b, _, _ in BANDS),
    *(f"hodl_rc_{b}" for b, _, _ in BANDS),
    "sth_supply",
    "sth_realized_cap",
    "realized_profit_usd",
]
SUPPLY_RTOL, SUPPLY_FROM = 0.005, dt.date(2011, 1, 1)
REALIZED_RTOL, REALIZED_FROM = 0.05, dt.date(2012, 1, 1)


def empty_cointime() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="float64") for c in COINTIME_COLUMNS})
    frame["metric_date"] = pd.Series(dtype="datetime64[ns]")
    return frame


def supply_before(heights: np.ndarray) -> np.ndarray:
    """Sat issued in blocks 0..h-1: the coins that exist when block h is mined."""
    subsidy = np.array([(50 * SAT) >> e for e in range(64)] + [0], dtype=np.int64)
    era_total = np.concatenate([[0], np.cumsum(subsidy * HALVING)])
    h = np.asarray(heights, dtype=np.int64)
    era = np.minimum(h // HALVING, 64)
    return era_total[era] + (h - era * HALVING) * subsidy[era]


def _prices(price: pd.Series, days: list[dt.date]) -> np.ndarray:
    p = price.reindex(days).to_numpy(dtype=float, copy=True)
    valid = np.flatnonzero(~np.isnan(p))
    if valid.size == 0:
        raise ValueError("no BTC price in the derivation span")
    gaps = np.isnan(p[valid[0] :])
    if gaps.any():
        first = days[valid[0] + int(np.argmax(gaps))]
        raise ValueError(
            f"BTC price missing on {first}, after the first price; refusing to value coins at 0"
        )
    p[: valid[0]] = 0.0  # pre-price coins: the realized-cap convention
    return p


def derive(
    spent: pd.DataFrame,
    created: pd.DataFrame,
    blocks: pd.DataFrame,
    price: pd.Series,
    end: dt.date,
    start: dt.date = GENESIS,
) -> pd.DataFrame:
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    n = len(days)
    index = {d: i for i, d in enumerate(days)}

    def pos(series: pd.Series) -> np.ndarray:
        return series.map(index).to_numpy(dtype=np.int64)

    created_sat = np.bincount(
        pos(created["created_date"]), weights=created["value_sat"].astype(float), minlength=n
    )
    cb_created = np.bincount(
        pos(blocks["block_date"]),
        weights=supply_before(blocks["height"].to_numpy()) / SAT,
        minlength=n,
    )
    ci, si = pos(spent["created_date"]), pos(spent["spent_date"])
    value = spent["value_sat"].to_numpy(dtype=float)
    cb_destroyed = np.bincount(
        si, weights=spent["coinblocks_destroyed"].to_numpy(dtype=float), minlength=n
    )
    p = _prices(price, days)
    profit = np.bincount(si, weights=value / SAT * np.maximum(p[si] - p[ci], 0.0), minlength=n)

    order = np.argsort(si, kind="stable")
    ci, si, value = ci[order], si[order], value[order]
    bounds = np.searchsorted(si, np.arange(n + 1))
    edges = np.array([lo for _, lo, _ in BANDS])
    cum_spent = np.zeros(n)
    rows = np.zeros((n, len(COINTIME_COLUMNS) - 1))
    col = {c: i - 1 for i, c in enumerate(COINTIME_COLUMNS)}
    for d in range(n):
        a, b = bounds[d], bounds[d + 1]
        np.add.at(cum_spent, ci[a:b], value[a:b])
        alive_sat = created_sat[: d + 1] - cum_spent[: d + 1]
        if alive_sat.min() < 0:
            c = int(np.argmin(alive_sat))
            raise ValueError(f"negative unspent value for coins created {days[c]} as of {days[d]}")
        by_age = (alive_sat / SAT)[::-1]  # index = age in days
        rc_by_age = by_age * p[: d + 1][::-1]
        starts = edges[edges <= d]
        bands, rc_bands = np.zeros(len(BANDS)), np.zeros(len(BANDS))
        bands[: len(starts)] = np.add.reduceat(by_age, starts)
        rc_bands[: len(starts)] = np.add.reduceat(rc_by_age, starts)
        supply, realized = by_age.sum(), rc_by_age.sum()
        r = rows[d]
        r[col["utxo_supply"]], r[col["utxo_realized_cap"]] = supply, realized
        for k, (name, _, _) in enumerate(BANDS):
            r[col[f"hodl_{name}"]] = bands[k] / supply if supply > 0 else np.nan
            r[col[f"hodl_rc_{name}"]] = rc_bands[k] / realized if realized > 0 else np.nan
        r[col["sth_supply"]] = by_age[:STH_DAYS].sum()
        r[col["sth_realized_cap"]] = rc_by_age[:STH_DAYS].sum()

    out = pd.DataFrame(rows, columns=COINTIME_COLUMNS[1:])
    out.insert(0, "metric_date", days)
    out["coinblocks_created"], out["coinblocks_destroyed"] = cb_created, cb_destroyed
    cum_created = np.cumsum(cb_created)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["liveliness"] = np.where(cum_created > 0, np.cumsum(cb_destroyed) / cum_created, np.nan)
    out["vaultedness"] = 1.0 - out["liveliness"]
    out["active_supply"] = out["utxo_supply"] * out["liveliness"]
    out["vaulted_supply"] = out["utxo_supply"] * out["vaultedness"]
    out["realized_profit_usd"] = profit
    return out[COINTIME_COLUMNS]


def vendor_mismatches(cointime: pd.DataFrame, vendor: pd.DataFrame) -> list[str]:
    """Our supply and realized cap against Coin Metrics', which computes both from the same
    chain independently: the correctness proof of the whole extraction (spec section 5)."""
    both = cointime.merge(vendor, on="metric_date", how="inner")
    out = []
    for ours, theirs, rtol, since in (
        ("utxo_supply", "supply", SUPPLY_RTOL, SUPPLY_FROM),
        ("utxo_realized_cap", "realized_cap", REALIZED_RTOL, REALIZED_FROM),
    ):
        dates = pd.to_datetime(both["metric_date"]).dt.date
        scoped = both[dates >= since].dropna(subset=[ours, theirs])
        rel = (scoped[ours] - scoped[theirs]).abs() / scoped[theirs].abs()
        bad = rel[rel > rtol]
        if len(bad):
            worst = scoped.loc[bad.idxmax(), "metric_date"]
            out.append(
                f"btc_cointime_daily/{ours}: {len(bad)} day(s) since {since} differ from Coin "
                f"Metrics {theirs} by more than {rtol:.1%} (worst {bad.max():.2%} on {worst})"
            )
    return out
