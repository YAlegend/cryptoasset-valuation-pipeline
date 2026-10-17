"""The linear family (spec 11.3): OLS retained for continuity, plus ridge and elastic net.

ONE WRAPPER FOR EVERY SCIKIT-LEARN ESTIMATOR, reused by the nonlinear family. It does
exactly two things beyond delegating:

* STANDARDISES. The features are log fair-value ratios across 35 methods whose spreads
  differ by orders of magnitude; ridge and elastic net penalise coefficients, so without
  a scaler the penalty would fall on whichever signal happens to be measured small. The
  scaler lives inside the pipeline, which is what makes it fitted on the training fold
  only -- the same rule the runner applies to imputation.
* REFUSES AN EMPTY DESIGN. A fold where no feature cleared the coverage floor cannot
  carry a regression. Returning zeros would make the model arithmetically identical to
  naive_persistence and it would appear in the table as having tied with the baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cryptoval.scoring.base import ModelSpec, Unfittable
from cryptoval.scoring.registry import load_model_specs, register_instance

_SPECS = load_model_specs()


class SklearnModel:
    """Adapts any scikit-learn regressor to the Model protocol."""

    def __init__(self, spec: ModelSpec, estimator) -> None:
        self.spec = spec
        self._estimator = estimator
        self._pipeline = None  # set only by a fit that succeeds

    def fit(self, X: np.ndarray, y: np.ndarray, context: dict[str, Any]) -> None:
        # Cleared FIRST: a fit that raises must not leave the previous fold's pipeline to
        # predict as if it were this fold's.
        self._pipeline = None
        if X.ndim != 2 or X.shape[1] == 0:
            raise Unfittable(
                f"{self.spec.model_id}: no features cleared the coverage floor on this fold"
            )
        if len(y) < X.shape[1] + 2:
            raise Unfittable(
                f"{self.spec.model_id}: {len(y)} training rows for {X.shape[1]} features"
            )
        # A fresh pipeline per fit, so a fold cannot inherit the previous fold's scaler.
        pipeline = make_pipeline(StandardScaler(), clone(self._estimator))
        pipeline.fit(X, y)
        self._pipeline = pipeline

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise Unfittable(f"{self.spec.model_id}: predict called before fit succeeded")
        return np.asarray(self._pipeline.predict(X), dtype="float64")


register_instance(SklearnModel(_SPECS["ols"], LinearRegression()))
register_instance(SklearnModel(_SPECS["ridge"], Ridge(**_SPECS["ridge"].params)))
register_instance(SklearnModel(_SPECS["elastic_net"], ElasticNet(**_SPECS["elastic_net"].params)))
