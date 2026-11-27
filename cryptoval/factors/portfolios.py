"""Value-weighted decile sorts and the Liu-Tsyvinski-Wu factors (design 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.factors.settings import FactorConfig


def _vw(group: pd.DataFrame) -> float:
    weight = group["market_cap_usd"]
    return float((group["ret"] * weight).sum() / weight.sum())


def _sorted(week: pd.DataFrame, signal: str) -> pd.DataFrame:
    # asset id breaks ties, so the sort never depends on row order
    return week.dropna(subset=[signal]).sort_values([signal, "asset"], kind="stable")


def decile_returns(panel: pd.DataFrame, signal: str, config: FactorConfig) -> pd.DataFrame:
    names = [f"D{b + 1}" for b in range(config.deciles)]
    rows = {}
    for week, group in panel.groupby("week"):
        ranked = _sorted(group, signal)
        n = len(ranked)
        if n < config.min_assets:
            continue
        bucket = np.arange(n) * config.deciles // n
        rows[week] = {name: _vw(ranked[bucket == b]) for b, name in enumerate(names)}
        rows[week]["n_assets"] = float(n)
    out = pd.DataFrame.from_dict(rows, orient="index", columns=[*names, "n_assets"])
    out.insert(config.deciles, "HL", out[names[-1]] - out[names[0]])
    return out


def ltw_factors(panel: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    """Single 30/40/30 sorts, not LTW's 2x3 (design 6: 50-110 assets make 2x3 legs too thin)."""

    def spread(group: pd.DataFrame, signal: str) -> float:
        ranked = _sorted(group, signal)
        if len(ranked) < config.min_assets:
            return np.nan
        k = int(len(ranked) * config.tail_share)
        return _vw(ranked.tail(k)) - _vw(ranked.head(k))

    rows = {
        week: {
            "CMKT": _vw(group),
            "CSMB": -spread(group, "size"),
            "CMOM": spread(group, "momentum"),
        }
        for week, group in panel.groupby("week")
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["CMKT", "CSMB", "CMOM"])
