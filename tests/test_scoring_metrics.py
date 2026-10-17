# tests/test_scoring_metrics.py
import numpy as np
import pytest

from cryptoval.scoring.metrics import (
    directional_accuracy,
    mae,
    mase,
    naive_scale,
    rmse,
    score_all,
)


def test_mae_and_rmse_on_a_known_error_vector():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.5, 1.5, 4.5])
    assert mae(y, yhat) == pytest.approx(0.8333333, abs=1e-6)
    assert rmse(y, yhat) == pytest.approx(np.sqrt((0.25 + 0.25 + 2.25) / 3))


def test_a_perfect_forecast_scores_zero_error_and_full_direction():
    y = np.array([0.01, -0.02, 0.03])
    assert mae(y, y) == pytest.approx(0.0)
    accuracy, usable = directional_accuracy(y, y)
    assert accuracy == pytest.approx(1.0)
    assert usable == 3


def test_mase_below_one_means_it_beat_predicting_zero():
    y = np.array([0.02, -0.02, 0.02, -0.02])
    scale = naive_scale(y)  # mean|y| under the zero forecast
    assert scale == pytest.approx(0.02)
    assert mase(y, y * 0.5, scale) == pytest.approx(0.5)
    assert mase(y, np.zeros_like(y), scale) == pytest.approx(1.0)


def test_mase_refuses_a_zero_naive_denominator():
    """Review Focus 4: USDC is a dollar peg; a flat training fold makes the scale 0."""
    assert naive_scale(np.zeros(100)) is None
    assert mase(np.array([0.0, 0.0]), np.array([0.0, 0.0]), None) is None
    assert mase(np.array([0.1]), np.array([0.0]), 0.0) is None


def test_directional_accuracy_reports_its_usable_count_and_refuses_an_all_zero_block():
    """Review Focus 5: a block with no directional variation is not 100% accuracy."""
    accuracy, usable = directional_accuracy(np.zeros(50), np.ones(50))
    assert accuracy is None
    assert usable == 0

    y = np.array([0.01, 0.0, -0.01, 0.0])
    accuracy, usable = directional_accuracy(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert usable == 2  # the two zero-return rows have no direction to get right
    assert accuracy == pytest.approx(0.5)


def test_every_metric_ignores_nan_pairs_rather_than_propagating_them():
    y = np.array([0.01, np.nan, -0.01])
    yhat = np.array([0.01, 0.5, np.nan])
    out = score_all(y, yhat, naive_scale=0.01)
    assert out["n"] == 1
    assert out["mae"] == pytest.approx(0.0)


def test_score_all_returns_none_for_every_metric_on_an_empty_overlap():
    out = score_all(np.array([np.nan, np.nan]), np.array([1.0, 2.0]), naive_scale=0.01)
    assert out["n"] == 0
    assert out["mae"] is None and out["rmse"] is None and out["mase"] is None
