"""The vector error-correction model (spec 11.4): "short-run dynamics and the SPEED of
mean reversion toward that equilibrium".

TWO loadings, and the gap's speed is neither alone. With the gap z = beta'(y, x) and
beta normalised to (1, beta_x), each series corrects a share of yesterday's gap: alpha
for market cap, alpha_independent for the fundamental. The gap itself moves by
gap_speed = beta'alpha = alpha + beta_x * alpha_independent a day. A pair where the
fundamental does all the correcting has alpha ~ 0 and still reverts, so the half life is
the GAP's, and reading market cap's alpha alone as "the speed of reversion" is wrong.

beta here is the VECM's own maximum-likelihood estimate, not the Engle-Granger/DOLS
coefficient reported beside it; the two are close but not equal.

Fitted only where cointegration survived the correction. A VECM on a pair with no
equilibrium estimates the speed of return to a relationship that does not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM

from cryptoval.econometrics.cointegration import _aligned


def speed_of_adjustment(y: pd.Series, x: pd.Series, lags: int = 1) -> dict[str, float]:
    y_values, x_values = _aligned(y, x, name="vecm")

    fitted = VECM(
        np.column_stack([y_values, x_values]), k_ar_diff=lags, coint_rank=1, deterministic="ci"
    ).fit()
    gap_speed = float(fitted.beta[:2, 0] @ fitted.alpha[:, 0])

    # The gap's SIZE shrinks by |1 + gap_speed| a day, so half of it closes in
    # log(0.5)/log|1 + gap_speed| days. That exists only while -2 < gap_speed < 0; below -1
    # the gap overshoots and flips sign but still shrinks. Outside the band the pair
    # diverges, and a number there would be read as a reversion speed.
    half_life = float("nan")
    if -2.0 < gap_speed < 0.0:
        shrink = abs(1.0 + gap_speed)
        half_life = 0.0 if shrink == 0.0 else float(np.log(0.5) / np.log(shrink))

    return {
        "alpha": float(fitted.alpha[0, 0]),
        "alpha_stderr": float(fitted.stderr_alpha[0, 0]),
        "alpha_independent": float(fitted.alpha[1, 0]),
        "alpha_independent_stderr": float(fitted.stderr_alpha[1, 0]),
        "gap_speed": gap_speed,
        "half_life_days": half_life,
        "lags": float(lags),
        "n": float(len(y_values)),
    }
