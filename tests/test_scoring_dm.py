# tests/test_scoring_dm.py
import numpy as np
import pytest

from cryptoval.scoring.dm import diebold_mariano


def _rng():
    return np.random.default_rng(20261017)


def test_two_identical_error_series_are_not_different():
    e = _rng().normal(size=500)
    result = diebold_mariano(e, e.copy(), horizon=1)
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.n == 500


def test_a_clearly_better_model_gets_a_negative_statistic_and_a_small_p():
    rng = _rng()
    baseline = rng.normal(scale=1.0, size=800)
    model = rng.normal(scale=0.3, size=800)
    result = diebold_mariano(model, baseline, horizon=1)
    # Negative: the model's loss is lower than the baseline's.
    assert result.statistic < 0
    assert result.p_value < 0.01


def test_a_clearly_worse_model_gets_a_positive_statistic():
    rng = _rng()
    baseline = rng.normal(scale=0.3, size=800)
    model = rng.normal(scale=1.0, size=800)
    assert diebold_mariano(model, baseline, horizon=1).statistic > 0


def test_a_longer_horizon_widens_the_variance_and_shrinks_the_statistic():
    """h-step errors are autocorrelated to lag h-1; ignoring that over-rejects."""
    rng = _rng()
    # Strongly autocorrelated errors -- exactly what overlapping h-step windows produce,
    # and exactly what a plain sample variance understates. With iid errors the HAC sum
    # is noise and this assertion would be a coin flip.
    baseline = np.convolve(rng.normal(size=849), np.ones(50) / 50, mode="valid") + 1.0
    model = baseline * 0.8
    at_1 = diebold_mariano(model, baseline, horizon=1)
    at_30 = diebold_mariano(model, baseline, horizon=30)
    # Measured: -99.5 at h=1 against -15.0 at h=30. Both p-values underflow to 0.0 on
    # this input, so the statistic is what the assertion can hold onto.
    assert abs(at_30.statistic) < 0.5 * abs(at_1.statistic)


def test_too_few_observations_is_a_refusal_with_its_count_not_a_p_value():
    result = diebold_mariano(np.ones(4), np.zeros(4), horizon=30)
    assert result.statistic is None and result.p_value is None
    assert "4 paired" in result.reason


def test_equal_magnitude_errors_of_opposite_sign_are_a_tie_not_a_division_by_zero():
    e = np.array([0.5] * 50)
    result = diebold_mariano(e, e * -1.0, horizon=1)  # squared loss identical, d == 0
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_nan_pairs_are_dropped_and_counted():
    model = np.array([0.1, np.nan, 0.3, 0.2] * 50)
    baseline = np.array([0.5, 0.5, np.nan, 0.6] * 50)
    result = diebold_mariano(model, baseline, horizon=1)
    assert result.n == 100


def test_hln_correction_shrinks_the_statistic_by_its_factor_where_it_is_large():
    """At n=50, h=30 the HLN factor is sqrt(0.168) = 0.41: a 59% shrink. Deleting it fails this."""
    rng = np.random.default_rng(5)  # fixed seed; positive HAC variance at h=30
    baseline = rng.normal(size=50)
    model = rng.normal(scale=0.5, size=50)
    result = diebold_mariano(model, baseline, horizon=30)
    # Uncorrected statistic: the same Bartlett-weighted Newey-West variance, HLN step removed.
    d = model**2 - baseline**2
    centred = d - d.mean()
    gamma = [np.mean(centred**2)] + [np.mean(centred[k:] * centred[:-k]) for k in range(1, 30)]
    bartlett = sum((1.0 - k / 30) * gamma[k] for k in range(1, 30))
    uncorrected = d.mean() / np.sqrt((gamma[0] + 2.0 * bartlett) / 50)
    assert result.statistic == pytest.approx(uncorrected * 0.168**0.5, rel=1e-6)


def test_hln_adjustment_of_zero_is_a_refusal_naming_the_horizon_and_the_adjustment():
    rng = np.random.default_rng(0)  # fixed seed; positive HAC variance at n=30, h=30
    baseline = rng.normal(size=30)
    model = rng.normal(scale=0.5, size=30)
    # n=30, h=30: (31 - 60 + 29) / 30 == 0 exactly, so the non-positive branch fires.
    result = diebold_mariano(model, baseline, horizon=30)
    assert result.statistic is None and result.p_value is None
    assert "horizon 30" in result.reason
    assert "0.000e+00" in result.reason


def test_hac_lags_are_bartlett_weighted_not_equal_weighted():
    """Hand-derived. d = [4, 0, 4, 0, ...], n=10, h=3: mean 2, centred +-2.
    gamma_0 = 4, gamma_1 = -3.6, gamma_2 = 3.2. Bartlett: V = 4 + 2(2/3 * -3.6 + 1/3 * 3.2)
    = 4/3. HLN adjustment = (11 - 6 + 6/10) / 10 = 0.56. Statistic = 2 / sqrt(4/30) * sqrt(0.56)
    = sqrt(16.8). The rectangular kernel gives V = 3.2 and sqrt(7) = 2.65: a different answer.
    """
    model = np.array([2.0, 0.0] * 5)
    baseline = np.zeros(10)
    result = diebold_mariano(model, baseline, horizon=3)
    assert result.statistic == pytest.approx(16.8**0.5, rel=1e-9)
