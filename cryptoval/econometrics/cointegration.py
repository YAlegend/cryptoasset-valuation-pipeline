"""Cointegration (spec 11.4): "does price maintain a long-run equilibrium with
fundamentals? ... far stronger than the correlation evidence".

THE COEFFICIENT IS ESTIMATED. A stationary practitioner ratio is the same hypothesis
with the coefficient forced to the value the ratio assumes, so estimating it says whether
the assumed value is the one the data supports. For the Metcalfe pair that converts a
goodness-of-fit statistic into a falsifiable test of the exponent 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from cryptoval.econometrics.base import Unusable

MIN_COINT_ROWS = 120

# A residual this flat means the two series are the same series up to an affine map, so
# the "equilibrium" is an identity. Plan 3 shipped exactly this shape once: velocity_total
# was identically volume/market_cap, and the regression recovered price = market cap /
# supply while looking like an estimate.
MIN_RESIDUAL_VARIANCE = 1e-10


def _aligned(
    y: pd.Series, x: pd.Series, name: str = "engle_granger"
) -> tuple[np.ndarray, np.ndarray]:
    """``name`` prefixes the refusal, so a message raised for the VECM does not claim to
    come from Engle-Granger."""
    if len(y) != len(x):
        raise Unusable(f"{name}: {len(y)} dependent rows against {len(x)} independent")
    frame = pd.DataFrame(
        {"y": pd.to_numeric(y, errors="coerce"), "x": pd.to_numeric(x, errors="coerce")}
    ).dropna()
    if len(frame) < MIN_COINT_ROWS:
        raise Unusable(
            f"{name}: {len(frame)} usable rows, {MIN_COINT_ROWS} needed for the "
            f"reference distribution"
        )
    return frame["y"].to_numpy(dtype="float64"), frame["x"].to_numpy(dtype="float64")


def engle_granger(y: pd.Series, x: pd.Series) -> dict[str, float]:
    """The two-step test, plus the cointegrating coefficient and its standard error."""
    y_values, x_values = _aligned(y, x)

    fitted = sm.OLS(y_values, sm.add_constant(x_values)).fit()
    residual_variance = float(np.var(fitted.resid))
    if residual_variance < MIN_RESIDUAL_VARIANCE:
        raise Unusable(
            f"engle_granger: residual variance {residual_variance:.3g} is below "
            f"{MIN_RESIDUAL_VARIANCE:.0e} on {len(y_values)} rows, so the independent series "
            f"is an affine function of the dependent one and the equilibrium is an identity"
        )

    statistic, p_value, _ = coint(y_values, x_values)
    coefficient, stderr = _dols(y_values, x_values)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "coefficient": coefficient,
        "coefficient_stderr": stderr,
        "n": float(len(y_values)),
    }


# Leads and lags of the differenced regressor in the dynamic regression.
DOLS_LEADS_LAGS = 2


def _dols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Stock-Watson dynamic OLS with a Newey-West standard error.

    The static regression's coefficient is consistent but its OLS standard error is not
    valid: the equilibrium error is autocorrelated and x is endogenous to it, so the naive
    interval is too narrow and would reject a true Metcalfe exponent. Leads and lags of
    dx absorb the endogeneity; the HAC covariance absorbs the autocorrelation.
    shortcut: rule-of-thumb bandwidth understates a near-unit-root residual; report it, and
    move to an automatic (Andrews) bandwidth if an interval's verdict hinges on it.
    """
    k = DOLS_LEADS_LAGS
    dx = np.diff(x, prepend=np.nan)
    columns = [x] + [np.roll(dx, -j) for j in range(-k, k + 1)]
    design = np.column_stack(columns)[k + 1 : len(x) - k]
    target = y[k + 1 : len(y) - k]
    maxlags = int(4 * (len(target) / 100) ** (2 / 9))
    fitted = sm.OLS(target, sm.add_constant(design)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return float(fitted.params[1]), float(fitted.bse[1])


def johansen(y: pd.Series, x: pd.Series, lags: int = 1) -> dict[str, float]:
    """The trace test at rank 0, reported as the second opinion on Engle-Granger.

    Engle-Granger picks one series as the dependent and can give different answers
    depending on which; Johansen is symmetric in the two. Reporting both is how the
    chapter says whether the finding survives that choice.

    NOT IN THE HOLM FAMILY, and not exactly sized. det_order=0 leaves the constant
    unrestricted, whose critical values assume the walks drift. Measured on independent
    walks (n=2500): 11.5% rejections at drift/sd 0, 9.5% at 0.02, 6% at 0.05 and above.
    This panel's median drift/sd is 0.018 for log market cap. The restricted-constant
    alternative is exact at zero drift but rejects 60% at 0.05, so it is worse here.
    Engle-Granger measured 5-7% over the same range and carries the claim.
    """
    y_values, x_values = _aligned(y, x)
    result = coint_johansen(np.column_stack([y_values, x_values]), det_order=0, k_ar_diff=lags)
    trace = float(result.lr1[0])
    critical = float(result.cvt[0, 1])  # column 1 is the 95% critical value
    return {
        "trace_statistic": trace,
        "critical_95": critical,
        "rank": float(1.0 if trace > critical else 0.0),
        "n": float(len(y_values)),
    }
