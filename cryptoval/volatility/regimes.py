"""Two-regime Markov-switching mean and variance (spec 11.4, design 6).

Regimes are labelled by VARIANCE: statsmodels' regime index is arbitrary and changes
between fits, so 'regime 1' means nothing on its own. 'calm' is the lower-variance one.

ESTIMATES ONLY. The likelihood-ratio test of two regimes against one has a non-standard
distribution (a nuisance parameter is unidentified under the null; Hansen 1992), so a
chi-squared p-value from it would be wrong, and none enters the Holm family.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from cryptoval.econometrics.base import Unusable

MIN_REGIME_ROWS = 365

REGIME_SCALARS = (
    "calm_mean",
    "calm_variance",
    "calm_p_stay",
    "calm_duration_days",
    "turbulent_mean",
    "turbulent_variance",
    "turbulent_p_stay",
    "turbulent_duration_days",
    "turbulent_share",
)


def fit_regimes(returns: pd.Series, search_reps: int = 20, seed: int = 0) -> dict[str, Any]:
    if len(returns) < MIN_REGIME_ROWS:
        raise Unusable(f"regimes: {len(returns)} returns, {MIN_REGIME_ROWS} needed")
    values = returns.to_numpy(dtype="float64")
    # The start-value search draws from numpy's global state; seeding it makes the fit
    # reproducible, which a published table needs.
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = MarkovRegression(values, k_regimes=2, trend="c", switching_variance=True).fit(
            search_reps=search_reps
        )
    if not result.mle_retvals.get("converged", False):
        raise Unusable(f"regimes: the likelihood did not converge ({result.mle_retvals})")

    params = dict(zip(result.model.param_names, np.asarray(result.params), strict=True))
    variances = [params["sigma2[0]"], params["sigma2[1]"]]
    calm = int(np.argmin(variances))
    turbulent = 1 - calm
    # statsmodels reports p[i->0] only; p[1->1] = 1 - p[1->0].
    stay = [params["p[0->0]"], 1.0 - params["p[1->0]"]]
    durations = np.asarray(result.expected_durations, dtype="float64")
    probability = pd.Series(
        np.asarray(result.smoothed_marginal_probabilities)[:, turbulent], index=returns.index
    )

    out: dict[str, Any] = {"n": float(len(values)), "turbulent_probability": probability}
    for label, i in (("calm", calm), ("turbulent", turbulent)):
        out[f"{label}_mean"] = float(params[f"const[{i}]"])
        out[f"{label}_variance"] = float(variances[i])
        out[f"{label}_p_stay"] = float(stay[i])
        out[f"{label}_duration_days"] = float(durations[i])
    out["turbulent_share"] = float((probability > 0.5).mean())
    return out
