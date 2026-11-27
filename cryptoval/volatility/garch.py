"""Volatility clustering and asymmetry (spec 11.4): ARCH-LM, GARCH(1,1), EGARCH(1,1,1).

Student-t innovations throughout: crypto returns are fat-tailed, and a normal likelihood
would be misspecified on every asset. ARCH-LM and EGARCH's gamma are HYPOTHESES (in the
Holm family); the GARCH parameters are estimates.

EGARCH's gamma multiplies the signed shock in log variance: gamma < 0 is the equity
leverage effect (falls raise volatility more than rises), gamma > 0 the inverse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

from cryptoval.econometrics.base import Unusable

ARCH_LM_LAGS = 7

# A series flatter than this has no volatility to model and the likelihood is singular.
MIN_VARIANCE = 1e-8


def _check(returns: pd.Series, name: str) -> None:
    values = returns.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise Unusable(f"{name}: {int((~np.isfinite(values)).sum())} non-finite returns")
    variance = float(np.var(values))
    if variance < MIN_VARIANCE:
        raise Unusable(
            f"{name}: return variance {variance:.3g} is below {MIN_VARIANCE:.0e}; "
            f"there is no volatility to model"
        )


def arch_lm(returns: pd.Series, lags: int = ARCH_LM_LAGS) -> dict[str, float]:
    """Engle (1982). H0: no ARCH effects, i.e. volatility does not cluster."""
    _check(returns, "arch_lm")
    values = returns.to_numpy(dtype="float64")
    statistic, p_value, _, _ = het_arch(values - values.mean(), nlags=lags, result_object=False)
    return {"statistic": float(statistic), "p_value": float(p_value), "n": float(len(values))}


def _fit(returns: pd.Series, name: str, **spec: Any) -> Any:
    _check(returns, name)
    result = arch_model(returns, mean="Constant", dist="t", **spec).fit(disp="off")
    if result.convergence_flag != 0:
        raise Unusable(
            f"{name}: optimizer did not converge (flag {result.convergence_flag}: "
            f"{result.optimization_result.message})"
        )
    return result


def fit_garch(returns: pd.Series) -> dict[str, Any]:
    result = _fit(returns, "garch", vol="GARCH", p=1, q=1)
    params = result.params
    persistence = float(params["alpha[1]"] + params["beta[1]"])
    # A shock's effect on variance decays by `persistence` a day; at or above 1 the
    # process is integrated and never halves, so a number there would be read as a speed.
    # A half life longer than the sample is the optimizer on the stationarity bound, not
    # a measurement (live run 2026-10-18: BTC, BNB, DOGE at ~1e11 days).
    n = int(result.nobs)
    half_life, reason = float("nan"), None
    if not 0.0 < persistence < 1.0:
        reason = f"persistence {persistence:.4f} >= 1: an integrated variance has no half life"
    else:
        days = float(np.log(0.5) / np.log(persistence))
        if days > n:
            reason = (
                f"half life {days:.3g} days exceeds the {n}-day sample (persistence "
                f"{persistence:.6f}, at the stationarity bound): not measurable here"
            )
        else:
            half_life = days
    return {
        "omega": float(params["omega"]),
        "alpha": float(params["alpha[1]"]),
        "beta": float(params["beta[1]"]),
        "nu": float(params["nu"]),
        "persistence": persistence,
        "half_life_days": half_life,
        "half_life_reason": reason,
        "n": float(n),
        "std_resid": result.std_resid,
        "cond_vol": result.conditional_volatility,
    }


def fit_egarch(returns: pd.Series) -> dict[str, float]:
    """H0: gamma = 0, a symmetric volatility response. Robust standard errors (arch's default)."""
    result = _fit(returns, "egarch", vol="EGARCH", p=1, o=1, q=1)
    return {
        "gamma": float(result.params["gamma[1]"]),
        "gamma_stderr": float(result.std_err["gamma[1]"]),
        "p_value": float(result.pvalues["gamma[1]"]),
        "n": float(result.nobs),
    }
