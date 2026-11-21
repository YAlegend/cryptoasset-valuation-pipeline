"""The nonlinear family (spec 11.3).

The dissertation's SVR (RBF and polynomial) and feed-forward NN, re-run on returns rather
than levels, plus gradient boosting. Every one of them is Task 7's SklearnModel with a
different estimator, so the standardising pipeline and the empty-design refusal are
inherited rather than restated four times.

EVERY STOCHASTIC ESTIMATOR IS SEEDED. The repo's contract is that re-running is a no-op
and that a stored row can be reproduced; an unseeded MLP would put a row in
fct_model_score that changes on every run, and a DM p-value computed from it would not be
a fact about the model.

THE PARAMETERS ARE NOT TUNED. There is no inner cross-validation loop here, because an
honest one would need a nested walk-forward inside each training fold and this universe
does not have the folds to spare -- see Task 2's measurement. The values below are
scikit-learn's defaults except where the target's scale demands otherwise (SVR's epsilon,
which defaults to 0.1 and would swallow a daily log return of 0.01 entirely, predicting a
flat zero and scoring exactly like the baseline). That limitation is reported, not hidden.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from cryptoval.scoring.linear import SklearnModel
from cryptoval.scoring.registry import load_model_specs, register_instance

_SPECS = load_model_specs()

register_instance(SklearnModel(_SPECS["svr_rbf"], SVR(**_SPECS["svr_rbf"].params)))
register_instance(SklearnModel(_SPECS["svr_poly"], SVR(**_SPECS["svr_poly"].params)))
register_instance(
    SklearnModel(
        _SPECS["gradient_boosting"],
        GradientBoostingRegressor(**_SPECS["gradient_boosting"].params),
    )
)
register_instance(
    SklearnModel(_SPECS["feedforward_nn"], MLPRegressor(**_SPECS["feedforward_nn"].params))
)
