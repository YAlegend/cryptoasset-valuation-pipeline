# cryptoval/scoring/metrics.py
"""MAE, RMSE, MASE and directional accuracy (spec 11.2).

Every function returns None rather than a number it cannot justify. Two cases in
particular return a confident, wrong answer if left alone, and both are live on this
universe:

* MASE divides by the naive forecast's in-sample error. USDC is a dollar peg whose daily
  log returns are ~1e-6 and can be exactly zero across a flat fold, so the denominator
  goes to zero and MASE comes back as inf or NaN -- which sorts to the top of a league
  table as either the best or the worst model depending on the sort.
* Directional accuracy on a test block whose returns are all one sign is whatever the
  model's constant prediction happens to be: 1.0 or 0.0, neither of them skill. Rows with
  a zero return have no direction to predict and are excluded, and the count that
  survived travels with the number.

THE SCALE FOR MASE. Hyndman's MASE scales by the in-sample one-step naive error. Our
target is ALREADY a difference of logs, so the naive forecast of a log return is zero and
its in-sample error is mean(|y_train|). Scaling by that makes MASE read directly:
below 1.0 beats predicting no return at all, above 1.0 loses to it.
"""

from __future__ import annotations

import numpy as np


def _aligned(y, yhat) -> tuple[np.ndarray, np.ndarray]:
    """The pairs where both are finite. A NaN prediction is a missing one, not a zero."""
    a = np.asarray(y, dtype="float64")
    b = np.asarray(yhat, dtype="float64")
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def mae(y, yhat) -> float | None:
    a, b = _aligned(y, yhat)
    return float(np.mean(np.abs(a - b))) if a.size else None


def rmse(y, yhat) -> float | None:
    a, b = _aligned(y, yhat)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else None


def naive_scale(y_train) -> float | None:
    """mean(|y|) over the TRAINING fold: the error of forecasting no return.

    None when it is zero or non-finite, which is the refusal MASE needs.
    """
    a = np.asarray(y_train, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    scale = float(np.mean(np.abs(a)))
    return scale if np.isfinite(scale) and scale > 0.0 else None


def mase(y, yhat, naive_scale: float | None) -> float | None:
    if naive_scale is None or not np.isfinite(naive_scale) or naive_scale <= 0.0:
        return None
    error = mae(y, yhat)
    return None if error is None else error / naive_scale


def directional_accuracy(y, yhat) -> tuple[float | None, int]:
    """(share of correctly signed predictions, rows that had a direction to get right)."""
    a, b = _aligned(y, yhat)
    usable = a != 0.0
    count = int(usable.sum())
    if count == 0:
        return None, 0
    correct = np.sign(a[usable]) == np.sign(b[usable])
    return float(np.mean(correct)), count


def score_all(y, yhat, naive_scale: float | None) -> dict[str, float | None]:
    a, _ = _aligned(y, yhat)
    accuracy, usable = directional_accuracy(y, yhat)
    return {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "mase": mase(y, yhat, naive_scale),
        "directional_accuracy": accuracy,
        "usable_direction_rows": float(usable),
        "n": float(a.size),
    }
