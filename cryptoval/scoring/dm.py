"""The Diebold-Mariano test (spec 11.2).

Written here because statsmodels ships no DM test, and because the two corrections it
needs are exactly the ones a hand-rolled version omits:

* HAC VARIANCE. h-step-ahead forecast errors are serially correlated up to lag h-1 by
  construction -- overlapping windows share h-1 days of data. Using the plain sample
  variance of the loss differential understates it, sometimes by an order of magnitude at
  h=365, and every model then looks significantly better than the baseline.
* THE HARVEY-LEYBOURNE-NEWBOLD CORRECTION. The uncorrected statistic over-rejects in small
  samples, which is every sample here. HLN scales it and refers it to a t distribution
  with n-1 degrees of freedom rather than the normal.

SIGN CONVENTION, fixed once: d_t = loss(model) - loss(baseline), so a NEGATIVE statistic
means the model loses less than the baseline. "Beat persistence" is statistic < 0 with a
small p-value, and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# Below this the t reference distribution is not worth consulting; the refusal names the
# count so a reader can see which asset-horizon pairs were too short.
MIN_PAIRED_OBSERVATIONS = 10


@dataclass(frozen=True)
class DMResult:
    statistic: float | None
    p_value: float | None
    n: int
    reason: str = ""


def diebold_mariano(errors_model, errors_baseline, horizon: int, power: int = 2) -> DMResult:
    """Test equal predictive accuracy of two forecast error series.

    ``power`` is the loss exponent: 2 for squared error (the default, matching RMSE),
    1 for absolute error (matching MAE).
    """
    a = np.asarray(errors_model, dtype="float64")
    b = np.asarray(errors_baseline, dtype="float64")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = int(a.size)

    if n < MIN_PAIRED_OBSERVATIONS:
        reason = f"only {n} paired observations, minimum {MIN_PAIRED_OBSERVATIONS}"
        return DMResult(None, None, n, reason)

    d = np.abs(a) ** power - np.abs(b) ** power
    d_bar = float(np.mean(d))
    centred = d - d_bar

    # Newey-West over lags 0..h-1: gamma_0 + 2 * sum gamma_k.
    gamma_0 = float(np.mean(centred**2))
    variance = gamma_0
    for lag in range(1, min(horizon, n)):
        gamma_k = float(np.mean(centred[lag:] * centred[:-lag]))
        variance += 2.0 * gamma_k

    if not math.isfinite(variance) or variance <= 0.0:
        # Exactly equal losses (d identically zero) is not a difference, and a negative
        # HAC estimate is the estimator failing rather than evidence.
        if np.allclose(d, 0.0):
            return DMResult(0.0, 1.0, n, "")
        return DMResult(None, None, n, f"non-positive HAC variance estimate {variance:.3e}")

    statistic = d_bar / math.sqrt(variance / n)

    # Harvey, Leybourne and Newbold (1997), equation 9.
    adjustment = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if adjustment <= 0.0:
        reason = f"HLN adjustment {adjustment:.3e} non-positive at horizon {horizon}"
        return DMResult(None, None, n, reason)
    statistic *= math.sqrt(adjustment)

    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1)))
    return DMResult(float(statistic), p_value, n, "")
