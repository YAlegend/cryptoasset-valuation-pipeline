"""The vector error-correction model (spec 11.4): "short-run dynamics and the SPEED of
mean reversion toward that equilibrium".

Alpha is the answer: the share of yesterday's disequilibrium corrected today. Its SIGN is
the whole finding -- negative is reversion, positive is divergence -- so a half life is
reported only where the sign permits one.

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
    alpha = float(fitted.alpha[0, 0])
    stderr = float(fitted.stderr_alpha[0, 0])

    # log(0.5)/log(1+alpha) is the days for half the gap to close, and it exists only
    # while -2 < alpha < 0. Outside that the pair diverges or oscillates, and a number
    # there would be read as a reversion speed.
    half_life = float("nan")
    if -2.0 < alpha < 0.0:
        half_life = float(np.log(0.5) / np.log1p(alpha))

    return {
        "alpha": alpha,
        "alpha_stderr": stderr,
        "half_life_days": half_life,
        "lags": float(lags),
        "n": float(len(y_values)),
    }
