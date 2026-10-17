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

from cryptoval.scoring.registry import load_model_specs, register

_SPECS = load_model_specs()


def _finite_mean(values) -> float:
    a = np.asarray(values, dtype="float64")
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else 0.0


@register
class NaivePersistence:
    spec = _SPECS["naive_persistence"]

    def fit(self, X: np.ndarray, y: np.ndarray, context: dict[str, Any]) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype="float64")


@register
class RandomWalkDrift:
    spec = _SPECS["random_walk_drift"]

    def __init__(self) -> None:
        self._prediction = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, context: dict[str, Any]) -> None:
        # The DAILY series, not y: y is already an h-horizon return, and using it here
        # would make this model arithmetically identical to historical_mean.
        horizon = int(context["horizon"])
        self._prediction = _finite_mean(context["daily_log_return"]) * horizon

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._prediction, dtype="float64")


@register
class HistoricalMean:
    spec = _SPECS["historical_mean"]

    def __init__(self) -> None:
        self._prediction = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, context: dict[str, Any]) -> None:
        self._prediction = _finite_mean(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._prediction, dtype="float64")
