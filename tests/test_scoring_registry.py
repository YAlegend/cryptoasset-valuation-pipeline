# tests/test_scoring_registry.py
import numpy as np
import pytest

import cryptoval.scoring  # noqa: F401 -- import registers every model family
from cryptoval.scoring.base import ModelSpec
from cryptoval.scoring.registry import (
    BASELINE_MODEL_ID,
    MODELS,
    load_model_specs,
    register_instance,
)


def _context(daily=None, horizon=1):
    return {"horizon": horizon, "daily_log_return": np.zeros(100) if daily is None else daily}


def test_every_registered_model_has_a_config_and_vice_versa():
    assert set(MODELS) == set(load_model_specs())


def test_the_baseline_is_registered_like_any_other_model():
    """Spec 11.1: a baseline must be scored by the same code path, not assumed."""
    assert BASELINE_MODEL_ID in MODELS
    assert MODELS[BASELINE_MODEL_ID].spec.family == "baseline"


def test_naive_persistence_predicts_no_return():
    model = MODELS["naive_persistence"]
    model.fit(np.zeros((100, 3)), np.full(100, 0.4), _context())
    assert np.allclose(model.predict(np.zeros((7, 3))), 0.0)


def test_historical_mean_predicts_the_training_mean_of_the_h_horizon_return():
    model = MODELS["historical_mean"]
    y = np.array([0.1, 0.2, 0.3, 0.4])
    model.fit(np.zeros((4, 2)), y, _context(horizon=30))
    assert model.predict(np.zeros((3, 2))) == pytest.approx(np.full(3, 0.25))


def test_random_walk_drift_scales_the_daily_drift_by_the_horizon():
    model = MODELS["random_walk_drift"]
    daily = np.full(500, 0.001)
    model.fit(np.zeros((500, 2)), np.full(500, 999.0), _context(daily=daily, horizon=30))
    # The h-horizon y is deliberately absurd: this model must read the DAILY series.
    assert model.predict(np.zeros((2, 2))) == pytest.approx(np.full(2, 0.03))


def test_the_three_baselines_are_not_the_same_model_twice():
    y = np.array([0.02, 0.04, 0.06, 0.08])
    daily = np.full(4, 0.0005)
    context = _context(daily=daily, horizon=30)
    predictions = []
    for model_id in ("naive_persistence", "random_walk_drift", "historical_mean"):
        model = MODELS[model_id]
        model.fit(np.zeros((4, 1)), y, context)
        predictions.append(float(model.predict(np.zeros((1, 1)))[0]))
    assert len(set(predictions)) == 3


def test_a_baseline_declares_it_needs_no_features():
    assert MODELS["naive_persistence"].spec.uses_features is False


def test_a_duplicate_model_id_is_refused():
    class Duplicate:
        spec = ModelSpec(
            model_id="naive_persistence", family="baseline", params={}, citation="x"
        )

        def fit(self, X, y, context):
            return None

        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(ValueError, match="already registered"):
        register_instance(Duplicate())
