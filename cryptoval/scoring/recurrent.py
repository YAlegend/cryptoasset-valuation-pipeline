"""The recurrent model (spec 11.3): a GRU over a sliding window of lagged features.

THE RUNNER HANDS A 2-D DESIGN; A GRU NEEDS 3-D. Sample ``i`` is ``X[i-L+1 : i+1]`` with
target ``y[i]`` (L = sequence_length), so the first L-1 training rows form no full window
and are dropped from training. At predict time the first L-1 test rows have no history in
``X_test``, so the last L-1 rows of the TRAINING design are prepended. That leaks nothing:
the features are already lagged a day when the runner sees them. Zero padding would
fabricate history, and dropping rows would break ``len(predict(X)) == len(X)``.

THAT HISTORY IS NOT ADJACENT TO THE TEST BLOCK. The stored rows are the last L-1 rows the model
was allowed to TRAIN on, and the runner's purge ends the training slice h + embargo rows before
the test block (those rows' labels overlap it). So a purge gap of h + embargo days sits between
the history and the first test row, which touches only the first L-1 rows of each test block
(under 4% at h=30) and only their feature history, not their target. The rows inside the gap
would give a continuous history and leak nothing, but fetching them means widening the Model
protocol and the runner for one model of eleven, so the gap is accepted rather than closed.

REFUSES THE HORIZONS THAT LACK INDEPENDENT OBSERVATIONS. Only ``qualifying_horizons`` are
fitted; the rest raise Unfittable carrying Task 2's measured asset count, which the runner
records as a ``refused`` row, a research finding rather than a defect.

DETERMINISTIC. Seeded, deterministic algorithms and one thread: an unseeded net would put
a row in fct_model_score that changes between runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from cryptoval.scoring.base import ModelSpec, Unfittable
from cryptoval.scoring.registry import load_model_specs, register_instance

_SPECS = load_model_specs()


class _Net(nn.Module):
    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _windows(X: np.ndarray, length: int) -> np.ndarray:
    """(n, p) -> (n - length + 1, length, p); window i ends at row i + length - 1."""
    view = np.lib.stride_tricks.sliding_window_view(X, length, axis=0)  # (n-L+1, p, L)
    return np.ascontiguousarray(view.transpose(0, 2, 1))


class GRUModel:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._reset()

    def _reset(self) -> None:
        # set only by a fit that succeeds
        self._net: _Net | None = None
        self._scaler: StandardScaler | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._tail: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, context: dict[str, Any]) -> None:
        self._reset()  # FIRST: a raising fit must not leave the previous fold's net behind
        p = self.spec.params
        horizon = context["horizon"]
        if horizon not in p["qualifying_horizons"]:
            raise Unfittable(
                f"{self.spec.model_id}: h={horizon} lacks the independent observations a "
                f"recurrent model needs; only {p['assets_clearing'].get(horizon, 'no')} of 24 "
                f"assets cleared >=5 folds and >=30 non-overlapping windows (measured "
                f"{p['measured_on']} on the {p['panel_vintage']} panel vintage; qualifying "
                f"horizons: {p['qualifying_horizons']})"
            )
        length = p["sequence_length"]
        if X.ndim != 2 or X.shape[1] == 0:
            raise Unfittable(
                f"{self.spec.model_id}: no features cleared the coverage floor on this fold"
            )
        if len(y) - length + 1 < X.shape[1] + 2:
            raise Unfittable(
                f"{self.spec.model_id}: {len(y)} training rows give {max(len(y) - length + 1, 0)}"
                f" windows of {length} for {X.shape[1]} features"
            )

        torch.manual_seed(p["random_state"])
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

        scaler = StandardScaler().fit(X)
        y_mean, y_std = float(np.mean(y)), float(np.std(y)) or 1.0  # y is ~0.01: scale it up
        windows = torch.tensor(_windows(scaler.transform(X), length), dtype=torch.float32)
        target = torch.tensor((y[length - 1 :] - y_mean) / y_std, dtype=torch.float32)

        net = _Net(X.shape[1], p["hidden_size"])
        optimiser = torch.optim.Adam(net.parameters(), lr=p["learning_rate"])
        for _ in range(p["epochs"]):  # full batch: no shuffling to seed, no validation split
            optimiser.zero_grad()
            nn.functional.mse_loss(net(windows), target).backward()
            optimiser.step()

        self._net, self._scaler, self._y_mean, self._y_std = net.eval(), scaler, y_mean, y_std
        self._tail = X[len(X) - length + 1 :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise Unfittable(f"{self.spec.model_id}: predict called before fit succeeded")
        length = self.spec.params["sequence_length"]
        history = np.vstack([self._tail, X])
        windows = torch.tensor(
            _windows(self._scaler.transform(history), length), dtype=torch.float32
        )
        with torch.no_grad():
            out = self._net(windows).numpy().astype("float64") * self._y_std + self._y_mean
        assert len(out) == len(X), f"gru produced {len(out)} predictions for {len(X)} rows"
        return out


register_instance(GRUModel(_SPECS["gru"]))
