"""The model contract (spec 11.1-11.3).

One signature for everything, so a baseline and a gradient booster are scored by the same
code and "did it beat persistence" is a comparison between two rows rather than between a
result and an assumption. That is spec 11.1's "baselines, established first", expressed
as a type rather than as a convention someone has to remember.

WHY fit TAKES A CONTEXT. The target handed to a model is the h-horizon log return, and
two of the three baselines can be computed from it, but the random walk WITH DRIFT cannot:
its drift is estimated from one-day returns and scaled by h. Without the context it would
be arithmetically identical to the historical mean, and spec 11.1 would be satisfied by
two models wearing three names. The context carries {"horizon": int,
"daily_log_return": np.ndarray} for the training fold; a model that does not need it
ignores it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str  # baseline | linear | nonlinear | recurrent
    params: dict[str, Any] = field(default_factory=dict)
    citation: str = ""
    # False for a model that ignores X entirely. The runner skips feature imputation for
    # it, so a baseline is scorable on an asset with no usable signals at all.
    uses_features: bool = True


@runtime_checkable
class Model(Protocol):
    spec: ModelSpec

    def fit(self, X: np.ndarray, y: np.ndarray, context: dict[str, Any]) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class Unfittable(Exception):
    """A model cannot be fitted on this fold, with the measurement that says so.

    Distinct from an arbitrary exception on purpose. The runner records this as a
    refusal carrying the reason, and anything else as "model raised", because one is a
    measurement about the data and the other is a defect in the code. A model that
    swallowed this and returned zeros would be scored as -- and indistinguishable from --
    the naive_persistence baseline it is being tested against.
    """
