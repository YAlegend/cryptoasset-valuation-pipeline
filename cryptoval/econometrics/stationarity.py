"""Unit-root pre-tests (spec 11.4): "are these series stationary? Determines what
regressions are even valid".

PRE-TESTS, not claims. They decide whether a cointegration test has a job, so they are
reported with their p-values and are NOT in the Holm family -- counting a specification
check as a hypothesis would make the claims harder to reject than the evidence warrants.

ADF and KPSS have OPPOSITE null hypotheses: ADF's null is a unit root, KPSS's null is
stationarity. Running both is how a reader tells "evidence of a unit root" from "no
evidence either way", which a single test cannot distinguish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss as _kpss

from cryptoval.econometrics.base import Unusable

# Below this the reference distributions are not worth consulting. The refusal names the
# count rather than returning a statistic nobody should read.
MIN_STAT_ROWS = 60

# A series flatter than this has no variation for a unit-root test to work on, and the
# regression is singular. The refusal carries the measured variance.
MIN_VARIANCE = 1e-12


def _clean(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").astype("float64").dropna().to_numpy()
    if len(values) < MIN_STAT_ROWS:
        raise Unusable(
            f"{name}: {len(values)} usable rows, {MIN_STAT_ROWS} needed for a unit-root test"
        )
    variance = float(np.var(values))
    if variance < MIN_VARIANCE:
        raise Unusable(
            f"{name}: variance {variance:.3g} is below {MIN_VARIANCE:.0e}, so the "
            f"regression is singular on {len(values)} rows"
        )
    return values


def adf(series: pd.Series, lag_criterion: str = "aic") -> dict[str, float]:
    """Augmented Dickey-Fuller. Null: the series HAS a unit root."""
    values = _clean(series, "adf")
    statistic, p_value, lags, _, _, _ = adfuller(
        values, autolag=lag_criterion.upper(), result_object=False
    )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "lags": float(lags),
        "n": float(len(values)),
    }


def kpss(series: pd.Series) -> dict[str, float]:
    """KPSS. Null: the series IS stationary -- the opposite of ADF's."""
    values = _clean(series, "kpss")
    statistic, p_value, lags, _ = _kpss(values, regression="c", nlags="auto", result_object=False)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "lags": float(lags),
        "n": float(len(values)),
    }
