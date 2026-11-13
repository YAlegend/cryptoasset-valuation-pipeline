import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.econometrics.stationarity import MIN_STAT_ROWS, adf, kpss


def _random_walk(n=800, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0.0, 1.0, n)))


def _stationary(n=800, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 1.0, n))


def test_adf_fails_to_reject_a_unit_root_on_a_random_walk():
    """Known answer, one direction. A random walk HAS a unit root."""
    assert adf(_random_walk())["p_value"] > 0.10


def test_adf_rejects_a_unit_root_on_white_noise():
    """Known answer, the other direction -- so the test can fail."""
    assert adf(_stationary())["p_value"] < 0.01


def test_kpss_rejects_stationarity_on_a_random_walk_and_not_on_noise():
    assert kpss(_random_walk())["p_value"] < 0.05
    assert kpss(_stationary())["p_value"] > 0.05


def test_both_report_the_sample_size_they_used():
    series = _random_walk(n=500)
    assert adf(series)["n"] == 500
    assert kpss(series)["n"] == 500


def test_a_constant_series_is_refused_with_its_measured_variance():
    """Review Focus 1: a constant series makes the ADF regression singular."""
    flat = pd.Series(np.full(400, 3.0))
    for test in (adf, kpss):
        with pytest.raises(Unusable) as exc:
            test(flat)
        assert "variance" in str(exc.value)
        assert "0" in str(exc.value)


def test_too_few_rows_is_refused_with_the_count():
    short = _random_walk(n=MIN_STAT_ROWS - 1)
    for test in (adf, kpss):
        with pytest.raises(Unusable) as exc:
            test(short)
        assert str(MIN_STAT_ROWS - 1) in str(exc.value)


def test_nan_rows_are_dropped_and_the_reported_n_is_the_count_used():
    series = _random_walk(n=500)
    series.iloc[:50] = np.nan
    assert adf(series)["n"] == 450
