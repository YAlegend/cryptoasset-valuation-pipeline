import numpy as np
import pytest

import cryptoval.scoring  # noqa: F401
from cryptoval.scoring.base import Unfittable
from cryptoval.scoring.baselines import HistoricalMean, NaivePersistence, RandomWalkDrift
from cryptoval.scoring.registry import MODELS, load_model_specs

LINEAR = ("ols", "ridge", "elastic_net")
NONLINEAR = ("svr_rbf", "svr_poly", "gradient_boosting", "feedforward_nn")
# Every model that is a SklearnModel. The guard tests below run over all of them, because a
# guard verified for one model and assumed for the others is the gap the M4 mutant exposed.
SKLEARN = LINEAR + NONLINEAR


def _learnable(n=400, p=3, seed=20261017):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 0.03 * X[:, 0] - 0.01 * X[:, 1] + rng.normal(scale=0.002, size=n)
    return X, y


def test_the_linear_family_is_registered_and_configured():
    specs = load_model_specs()
    for model_id in LINEAR:
        assert model_id in MODELS
        assert specs[model_id].family == "linear"


@pytest.mark.parametrize("model_id", LINEAR)
def test_a_linear_model_recovers_a_learnable_signal_better_than_predicting_zero(model_id):
    X, y = _learnable()
    model = MODELS[model_id]
    model.fit(X[:300], y[:300], {"horizon": 1, "daily_log_return": np.zeros(300)})
    predicted = model.predict(X[300:])
    assert np.mean(np.abs(y[300:] - predicted)) < np.mean(np.abs(y[300:]))


@pytest.mark.parametrize("model_id", LINEAR)
def test_a_model_with_no_features_refuses_rather_than_impersonating_the_baseline(model_id):
    model = MODELS[model_id]
    with pytest.raises(Unfittable, match="no features"):
        context = {"horizon": 1, "daily_log_return": np.zeros(100)}
        model.fit(np.zeros((100, 0)), np.zeros(100), context)


@pytest.mark.parametrize("model_id", LINEAR)
def test_refitting_the_same_model_twice_gives_the_same_prediction(model_id):
    """Reproducibility is a project invariant: a score must not depend on run order."""
    X, y = _learnable()
    model = MODELS[model_id]
    context = {"horizon": 1, "daily_log_return": np.zeros(300)}
    model.fit(X[:300], y[:300], context)
    first = model.predict(X[300:])
    model.fit(X[:300], y[:300], context)
    assert np.allclose(first, model.predict(X[300:]))


def test_features_are_standardised_so_scale_does_not_decide_the_fit():
    X, y = _learnable()
    scaled = X.copy()
    scaled[:, 0] *= 1e6
    model = MODELS["ridge"]
    context = {"horizon": 1, "daily_log_return": np.zeros(300)}
    model.fit(X[:300], y[:300], context)
    plain = model.predict(X[300:])
    model.fit(scaled[:300], y[:300], context)
    assert np.allclose(plain, model.predict(scaled[300:]), atol=1e-6)


# Ruling 16: a baseline must not quietly impersonate the model it is compared against.
# None of these paths is reachable through the runner (one try wraps fit and predict), so
# the baselines are called directly.
STATEFUL = (RandomWalkDrift, HistoricalMean)
X_ONE = np.zeros((3, 1))


@pytest.mark.parametrize("cls", STATEFUL)
def test_predict_before_any_fit_refuses_rather_than_returning_zero(cls):
    with pytest.raises(Unfittable, match="before fit"):
        cls().predict(X_ONE)


@pytest.mark.parametrize("cls", STATEFUL)
def test_an_all_nan_training_fold_refuses_with_its_measurement(cls):
    model = cls()
    context = {"horizon": 5, "daily_log_return": np.full(7, np.nan)}
    with pytest.raises(Unfittable, match=r"7 rows, 0 finite"):
        model.fit(np.zeros((7, 1)), np.full(7, np.nan), context)
    with pytest.raises(Unfittable, match="before fit"):
        model.predict(X_ONE)


@pytest.mark.parametrize("cls", STATEFUL)
def test_a_fit_that_raises_does_not_leave_the_previous_folds_prediction(cls):
    model = cls()
    good = {"horizon": 1, "daily_log_return": np.full(4, 0.1)}
    model.fit(np.zeros((4, 1)), np.full(4, 0.1), good)
    assert model.predict(X_ONE)[0] == pytest.approx(0.1)
    with pytest.raises((KeyError, ValueError)):
        # drift: context has no "horizon"; historical_mean: y is not numeric
        model.fit(np.zeros((4, 1)), np.array(["a", "b", "c", "d"]), {})
    with pytest.raises(Unfittable, match="before fit"):
        model.predict(X_ONE)


def test_naive_persistence_is_exempt_and_predicts_exactly_zero_unfitted():
    assert NaivePersistence().predict(X_ONE).tolist() == [0.0, 0.0, 0.0]


def _fresh(model_id):
    """A new instance of a registered SklearnModel, so the registered singleton is untouched."""
    registered = MODELS[model_id]
    return type(registered)(registered.spec, registered._estimator)


@pytest.mark.parametrize("model_id", SKLEARN)
def test_sklearn_predict_before_any_fit_refuses_with_unfittable(model_id):
    X, y = _learnable()
    model = _fresh(model_id)
    with pytest.raises(Unfittable, match="before fit"):
        model.predict(X[:5])


@pytest.mark.parametrize("model_id", SKLEARN)
@pytest.mark.parametrize("bad_fit", ["unfittable", "sklearn_error"])
def test_sklearn_failed_fit_does_not_leave_the_previous_folds_pipeline(model_id, bad_fit):
    X, y = _learnable()
    model = _fresh(model_id)
    context = {"horizon": 1, "daily_log_return": np.zeros(300)}
    model.fit(X[:300], y[:300], context)
    assert model.predict(X[300:]).shape == (100,)
    if bad_fit == "unfittable":
        with pytest.raises(Unfittable, match="no features"):
            model.fit(np.zeros((300, 0)), y[:300], context)
    else:
        with pytest.raises(ValueError):  # sklearn rejects mismatched lengths, past the guards
            model.fit(X[:300], y[:10], context)
    with pytest.raises(Unfittable, match="before fit"):
        model.predict(X[300:])


def test_the_nonlinear_family_is_registered_and_configured():
    specs = load_model_specs()
    for model_id in NONLINEAR:
        assert model_id in MODELS
        assert specs[model_id].family == "nonlinear"


def test_the_registry_holds_the_ten_models_this_plan_builds():
    # 3 baselines + 3 linear + 4 nonlinear. A count that drifts without a plan entry is a
    # model someone added without a config.
    assert len(MODELS) == 10


# Spread, not error. The failure guarded here is a model that predicts almost nothing (the
# epsilon tube swallowing every observation). Such a model can post a respectable MAE and even
# a high correlation while having learned nothing, so the bar reads the spread of predictions.
# Measured on this fixture, std(predicted) / std(y_test):
#   working: svr_rbf 0.961, svr_poly 0.994, gradient_boosting 0.973, feedforward_nn 1.268
#   broken (epsilon 0.1): svr_rbf 0.053, svr_poly 0.146
# The MLP's MAE ratio to predicting zero is 0.860: weak, not degenerate, and this bar is not
# meant to catch it. A weak model losing to naive_persistence is a result to report.
@pytest.mark.parametrize("model_id", NONLINEAR)
def test_a_nonlinear_model_recovers_a_learnable_signal(model_id):
    X, y = _learnable(n=600)
    model = MODELS[model_id]
    model.fit(X[:450], y[:450], {"horizon": 1, "daily_log_return": np.zeros(450)})
    predicted = model.predict(X[450:])
    assert np.isfinite(predicted).all()
    assert len(predicted) == 150
    assert np.std(predicted) / np.std(y[450:]) > 0.5


@pytest.mark.parametrize("model_id", NONLINEAR)
def test_a_stochastic_model_is_seeded_so_two_runs_agree(model_id):
    X, y = _learnable(n=600)
    context = {"horizon": 1, "daily_log_return": np.zeros(450)}
    model = MODELS[model_id]
    model.fit(X[:450], y[:450], context)
    first = model.predict(X[450:])
    model.fit(X[:450], y[:450], context)
    assert np.allclose(first, model.predict(X[450:]))


@pytest.mark.parametrize("model_id", NONLINEAR)
def test_a_nonlinear_model_with_no_features_refuses(model_id):
    with pytest.raises(Unfittable):
        MODELS[model_id].fit(
            np.zeros((100, 0)), np.zeros(100), {"horizon": 1, "daily_log_return": np.zeros(100)}
        )


@pytest.mark.parametrize("model_id", SKLEARN)
def test_a_sklearn_model_with_too_few_rows_for_its_features_refuses(model_id):
    # 3 rows for 3 features needs at least 5 (features + 2); the refusal must come before sklearn.
    with pytest.raises(Unfittable, match="training rows"):
        _fresh(model_id).fit(
            np.zeros((3, 3)), np.zeros(3), {"horizon": 1, "daily_log_return": np.zeros(3)}
        )


def test_the_mlp_does_not_carve_a_random_validation_split_out_of_its_training_data():
    # scikit-learn's early stopping holds out a RANDOM validation fraction of the training
    # rows, which the plan's Global Constraints forbid anywhere. A seeded MLP is deterministic
    # with the flag on as well, so no behavioural test can see it: the config is the thing to pin.
    assert load_model_specs()["feedforward_nn"].params["early_stopping"] is False
