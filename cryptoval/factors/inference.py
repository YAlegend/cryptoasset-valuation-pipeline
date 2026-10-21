"""Newey-West means, factor alphas and Fama-MacBeth slopes (design 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cryptoval.econometrics.base import Unusable
from cryptoval.factors.settings import FactorConfig

MIN_WEEKS = 52


def _hac(y: np.ndarray, x: np.ndarray, lags: int):
    if len(y) < MIN_WEEKS:
        raise Unusable(f"{len(y)} weeks, {MIN_WEEKS} needed")
    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def nw_mean(series: pd.Series, lags: int) -> dict[str, float]:
    y = series.dropna().to_numpy(dtype="float64")
    fit = _hac(y, np.ones((len(y), 1)), lags)
    return {
        "mean_return": float(fit.params[0]),
        "t_stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
        "n_weeks": float(len(y)),
    }


def nw_alpha(portfolio: pd.Series, factors: pd.DataFrame, lags: int) -> dict[str, float]:
    data = pd.concat([portfolio.rename("_y"), factors], axis=1).dropna()
    x = sm.add_constant(data[factors.columns].to_numpy(dtype="float64"), has_constant="add")
    fit = _hac(data["_y"].to_numpy(dtype="float64"), x, lags)
    out = {
        "alpha": float(fit.params[0]),
        "t_stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
        "n_weeks": float(len(data)),
    }
    for i, name in enumerate(factors.columns, start=1):
        out[f"beta_{name.lower()}"] = float(fit.params[i])
    return out


def fama_macbeth(
    panel: pd.DataFrame, signals: tuple[str, ...], config: FactorConfig
) -> pd.DataFrame:
    """Weekly OLS of winsorized returns on percentile ranks in [0, 1]: a slope is the
    top-minus-bottom return spread. The winsorizing stops one micro-cap setting the slope."""
    rows = {}
    for week, group in panel.dropna(subset=list(signals)).groupby("week"):
        n = len(group)
        if n < config.min_assets:
            continue
        lo, hi = group["ret"].quantile([config.winsor_quantile, 1 - config.winsor_quantile])
        y = group["ret"].clip(lo, hi).to_numpy(dtype="float64")
        ranks = (group[list(signals)].rank() - 1) / (n - 1)
        x = np.column_stack([np.ones(n), ranks.to_numpy(dtype="float64")])
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        rows[week] = dict(zip(signals, coef[1:], strict=True))
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(signals))
