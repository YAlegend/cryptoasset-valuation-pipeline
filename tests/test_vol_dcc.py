import numpy as np
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.volatility.dcc import engle_sheppard, fit_dcc


def _simulate(n, a, b, rho_bar=0.5, seed=0):
    rng = np.random.default_rng(seed)
    qbar = np.array([[1.0, rho_bar], [rho_bar, 1.0]])
    q = qbar.copy()
    z = np.empty((n, 2))
    for t in range(n):
        if t:
            q = (1 - a - b) * qbar + a * np.outer(z[t - 1], z[t - 1]) + b * q
        r = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
        z[t] = np.linalg.cholesky(r) @ rng.standard_normal(2)
    return z


def test_it_recovers_a_known_dcc():
    fit = fit_dcc(_simulate(3000, 0.05, 0.93))
    assert fit["a"] == pytest.approx(0.05, abs=0.03)
    assert fit["b"] == pytest.approx(0.93, abs=0.05)
    assert len(fit["rho"]) == 3000


def test_engle_sheppard_rejects_a_time_varying_correlation():
    assert engle_sheppard(_simulate(3000, 0.05, 0.93))["p_value"] < 1e-6


def test_engle_sheppard_does_not_reject_a_constant_correlation():
    assert engle_sheppard(_simulate(3000, 0.0, 0.0, seed=3))["p_value"] > 0.05


def test_engle_sheppard_size_is_at_most_nominal():
    rejections = sum(
        engle_sheppard(_simulate(1000, 0.0, 0.0, seed=s))["p_value"] < 0.05 for s in range(60)
    )
    assert rejections <= 6  # 5% of 60 is 3; allow sampling noise, not a broken test


def test_a_constant_correlation_fits_a_near_zero():
    assert fit_dcc(_simulate(3000, 0.0, 0.0, seed=3))["a"] < 0.03


def test_a_short_or_non_finite_input_is_refused():
    with pytest.raises(Unusable, match="100 rows, 250 needed"):
        fit_dcc(np.zeros((100, 2)))
    z = _simulate(500, 0.0, 0.0)
    z[10, 0] = np.nan
    with pytest.raises(Unusable, match="1 non-finite"):
        engle_sheppard(z)
