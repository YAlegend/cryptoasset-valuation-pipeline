"""The three baselines, established first (spec 11.1).

"Published evidence suggests full-history BTC is statistically indistinguishable from a
random walk, so a model that fails to beat persistence is reported as having failed to
beat it." These three are what that sentence means in code.

They are three estimators, not one wearing three hats:
  naive_persistence   the random walk's own forecast of a future log return: zero
  random_walk_drift   the mean ONE-DAY log return over the training fold, times h
  historical_mean     the mean h-HORIZON log return over the training fold
The second and third converge in the limit and differ in every finite sample, which is
why both are scored rather than one being assumed to stand for the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cryptoval.scoring.base import Unfittable
from cryptoval.scoring.registry import load_model_specs, register

_SPECS = load_model_specs()


def _finite_mean(values, model_id: str) -> float:
    """Mean of the finite values; refuses, with its count, rather than returning 0.0.

    An all-NaN fold averaged to 0.0 would make the model arithmetically identical to
    naive_persistence while appearing in the table as a different model.
    """
    a = np.asarray(values, dtype="float64")
    finite = a[np.isfinite(a)]
    if not finite.size:
        raise Unfittable(f"{model_id}: no finite value to average ({a.size} rows, 0 finite)")
    return float(np.mean(finite))


class _ConstantForecast:
    """A prediction fixed at fit time. Unset means unfitted, and unfitted refuses.

    ``fit`` clears it FIRST, before anything can raise, so a fit that fails cannot leave
    the previous fold's number behind to be predicted as if it were this fold's.
    """

    _prediction: float | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._prediction is None:
            raise Unfittable(f"{self.spec.model_id}: predict called before fit succeeded")
        return np.full(len(X), self._prediction, dtype="float64")


@register
class NaivePersistence:
    spec = _SPECS["naive_persistence"]

    def fit(self, X: np.ndarray, y: np.ndarray, context: dict[str, Any]) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype="float64")


@register
class RandomWalkDrift(_ConstantForecast):
    spec = _SPECS["random_walk_drift"]

    def fit(self, X: np.ndarray, y: np.ndarray, context: dict[str, Any]) -> None:
        self._prediction = None
        # The DAILY series, not y: y is already an h-horizon return, and using it here
        # would make this model arithmetically identical to historical_mean.
        horizon = int(context["horizon"])
        self._prediction = _finite_mean(context["daily_log_return"], self.spec.model_id) * horizon


@register
class HistoricalMean(_ConstantForecast):
    spec = _SPECS["historical_mean"]

    def fit(self, X: np.ndarray, y: np.ndarray, context: dict[str, Any]) -> None:
        self._prediction = None
        self._prediction = _finite_mean(y, self.spec.model_id)
