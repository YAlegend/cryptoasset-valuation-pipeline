import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.volatility.regimes import fit_regimes


def _switching(n=3000, p_stay=0.98, variances=(1.0, 9.0), seed=0):
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < p_stay else 1 - state[t - 1]
    values = rng.standard_normal(n) * np.sqrt(np.asarray(variances)[state])
    index = pd.date_range("2015-01-01", periods=n, freq="D")
    return pd.Series(values, index=index), state


def test_it_recovers_the_variances_in_order_and_the_persistence():
    returns, state = _switching()
    fit = fit_regimes(returns)
    assert fit["calm_variance"] == pytest.approx(1.0, rel=0.2)
    assert fit["turbulent_variance"] == pytest.approx(9.0, rel=0.2)
    assert fit["calm_duration_days"] > 20
    agreement = ((fit["turbulent_probability"] > 0.5).to_numpy() == (state == 1)).mean()
    assert agreement > 0.9


def test_labels_follow_variance_not_the_regime_index():
    returns, _ = _switching(variances=(9.0, 1.0))
    fit = fit_regimes(returns)
    assert fit["calm_variance"] < fit["turbulent_variance"]


def test_it_is_deterministic():
    returns, _ = _switching(n=1500)
    assert fit_regimes(returns)["turbulent_share"] == fit_regimes(returns)["turbulent_share"]


def test_a_short_series_is_refused():
    with pytest.raises(Unusable, match="regimes: 50 returns"):
        fit_regimes(pd.Series(np.ones(50), index=pd.date_range("2015-01-01", periods=50)))


def test_labels_follow_variance_when_statsmodels_puts_the_high_variance_first(monkeypatch):
    """statsmodels happened to put the low variance at index 0 on both simulations above,
    so neither could catch a hard-coded calm = 0. This stub puts it at index 1."""
    from cryptoval.volatility import regimes

    n = 400

    class _Result:
        mle_retvals = {"converged": True}
        params = np.array([0.9, 0.05, 0.3, 0.1, 25.0, 2.0])
        expected_durations = np.array([10.0, 20.0])
        smoothed_marginal_probabilities = np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])

        class model:
            param_names = ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "sigma2[0]", "sigma2[1]"]

    class _Model:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, search_reps):
            return _Result()

    monkeypatch.setattr(regimes, "MarkovRegression", _Model)
    fit = fit_regimes(pd.Series(np.ones(n), index=pd.date_range("2015-01-01", periods=n)))
    assert fit["calm_variance"] == 2.0
    assert fit["turbulent_variance"] == 25.0
    assert fit["turbulent_p_stay"] == pytest.approx(0.9)
    assert fit["calm_p_stay"] == pytest.approx(0.95)
    assert fit["calm_duration_days"] == 20.0
    assert fit["turbulent_share"] == 1.0
