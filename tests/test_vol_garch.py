import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.volatility import garch


def _simulate(n=3000, omega=0.1, alpha=0.1, beta=0.85, leverage=0.0, seed=0):
    """GJR-style: a negative shock adds `leverage` to the variance response."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n)
    out = np.empty(n)
    variance = omega / (1.0 - alpha - beta - leverage / 2.0)
    for t in range(n):
        out[t] = np.sqrt(variance) * shocks[t]
        variance = omega + (alpha + leverage * (out[t] < 0)) * out[t] ** 2 + beta * variance
    return pd.Series(out, index=pd.date_range("2015-01-01", periods=n, freq="D"))


def _iid(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n), index=pd.date_range("2015-01-01", periods=n))


def test_arch_lm_finds_clustering_in_a_garch_process():
    assert garch.arch_lm(_simulate())["p_value"] < 1e-6


def test_arch_lm_finds_none_in_iid_noise():
    assert garch.arch_lm(_iid())["p_value"] > 0.05


def test_garch_recovers_its_parameters():
    fit = garch.fit_garch(_simulate())
    assert fit["alpha"] == pytest.approx(0.1, abs=0.04)
    assert fit["beta"] == pytest.approx(0.85, abs=0.06)
    assert fit["half_life_days"] == pytest.approx(np.log(0.5) / np.log(fit["persistence"]))
    assert fit["std_resid"].index.equals(_simulate().index)


def test_an_integrated_garch_has_no_half_life(monkeypatch):
    class _Res:
        convergence_flag = 0
        nobs = 10
        params = pd.Series({"omega": 0.1, "alpha[1]": 0.2, "beta[1]": 0.8, "nu": 4.0})
        std_resid = conditional_volatility = pd.Series([1.0])

    class _Model:
        def fit(self, disp):
            return _Res()

    monkeypatch.setattr(garch, "arch_model", lambda *a, **k: _Model())
    assert np.isnan(garch.fit_garch(_simulate(100))["half_life_days"])


def test_egarch_recovers_the_leverage_sign():
    fit = garch.fit_egarch(_simulate(leverage=0.15, alpha=0.02))
    assert fit["gamma"] < 0.0
    assert fit["p_value"] < 1e-3


def test_egarch_finds_no_asymmetry_in_a_symmetric_process():
    assert garch.fit_egarch(_simulate())["p_value"] > 0.05


def test_a_flat_series_is_refused_with_its_variance():
    flat = pd.Series(np.zeros(1000), index=pd.date_range("2015-01-01", periods=1000))
    with pytest.raises(Unusable, match="variance 0 is below"):
        garch.fit_garch(flat)


def test_non_convergence_is_refused_with_the_flag(monkeypatch):
    class _Res:
        convergence_flag = 4

        class optimization_result:
            message = "Iteration limit reached"

    class _Model:
        def fit(self, disp):
            return _Res()

    monkeypatch.setattr(garch, "arch_model", lambda *a, **k: _Model())
    with pytest.raises(Unusable, match="flag 4: Iteration limit reached"):
        garch.fit_garch(_simulate(100))
