"""The walk-forward loop (spec 11.2).

THE GRAIN is spec 8.1's: (model_id, asset, fold, horizon, metric), one row per metric.
fold = -1 marks a value pooled across every fold, which is where the Diebold-Mariano
statistic lives: pooling the out-of-sample errors gives the test more power than five
separate tests on five short blocks, and spec 11.2 asks for a p-value per model, not per
fold.

TWO LEAKS ARE CLOSED HERE rather than by each model, because a leak closed in seven
places is a leak open in one:

* IMPUTATION IS FITTED ON THE TRAINING SLICE ONLY. A median over the whole series is
  look-ahead through the back door, and it is the leak most likely to survive review
  because it reads as data cleaning.
* THE FEATURE SET IS CHOSEN PER FOLD, from the training slice's own coverage. Most of
  this project's signals start when their method's inputs start -- price_to_tvl in 2020,
  power_law only for the 13 assets with a genesis_date -- so a feature set chosen once
  over the whole series hands a 2019 fold a column that did not exist in 2019.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cryptoval.scoring.base import Model
from cryptoval.scoring.dm import diebold_mariano
from cryptoval.scoring.metrics import naive_scale, score_all
from cryptoval.scoring.registry import BASELINE_MODEL_ID
from cryptoval.scoring.splits import expanding_folds, fold_report

SCORE_COLUMNS = [
    "model_id",
    "symbol",
    "horizon",
    "fold",
    "metric",
    "value",
    "n_test_rows",
    "train_rows",
    "non_overlapping_windows",
    "reason",
    "panel_vintage_ts",
]

# A feature present on less than half the training slice is mostly imputation by the time
# the model sees it, and an imputed column with a real name is worse than no column.
MIN_FEATURE_COVERAGE = 0.5

NON_FEATURE_COLUMNS = frozenset({"symbol", "metric_date", "y", "daily_log_return"})


def _row(**kwargs: Any) -> dict[str, Any]:
    base = dict.fromkeys(SCORE_COLUMNS)
    base.update(kwargs)
    return base


def _design(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    """Train and test matrices, imputed with the TRAINING median only."""
    if not features:
        return np.zeros((len(train), 0)), np.zeros((len(test), 0))
    medians = train[features].median(numeric_only=True)
    filled_train = train[features].fillna(medians).fillna(0.0)
    filled_test = test[features].fillna(medians).fillna(0.0)
    return (
        filled_train.to_numpy(dtype="float64"),
        filled_test.to_numpy(dtype="float64"),
    )


def run_walk_forward(
    frame: pd.DataFrame,
    models: dict[str, Model],
    horizon: int,
    panel_vintage_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Score every model on every symbol in ``frame`` at one horizon."""
    rows: list[dict[str, Any]] = []
    features_used: dict[str, list[str]] = {}
    candidates = [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]

    for symbol, group in frame.groupby("symbol", sort=True):
        group = group.sort_values("metric_date").reset_index(drop=True)
        report = fold_report(len(group), horizon)
        folds = expanding_folds(len(group), horizon)
        if not folds:
            rows.append(
                _row(
                    model_id="*",
                    symbol=str(symbol),
                    horizon=horizon,
                    fold=-1,
                    metric="refused",
                    reason=report["reason"],
                    panel_vintage_ts=panel_vintage_ts,
                )
            )
            continue

        # Pooled out-of-sample errors per model, for the DM test at fold -1.
        pooled: dict[str, list[np.ndarray]] = {model_id: [] for model_id in models}

        for fold in folds:
            train = group.iloc[fold.train].copy()
            test = group.iloc[fold.test].copy()
            labelled = train[train["y"].notna()]
            if labelled.empty or test["y"].notna().sum() == 0:
                continue

            coverage = labelled[candidates].notna().mean() if candidates else pd.Series(dtype=float)
            features = sorted(coverage[coverage >= MIN_FEATURE_COVERAGE].index)
            features_used[f"{symbol}|{fold.index}"] = features
            x_train, x_test = _design(labelled, test, features)
            y_train = labelled["y"].to_numpy(dtype="float64")
            y_test = test["y"].to_numpy(dtype="float64")
            scale = naive_scale(y_train)
            context = {
                "horizon": horizon,
                "daily_log_return": labelled.get(
                    "daily_log_return", pd.Series(np.zeros(len(labelled)))
                ).to_numpy(dtype="float64"),
            }

            for model_id, model in models.items():
                try:
                    model.fit(x_train, y_train, context)
                    predicted = np.asarray(model.predict(x_test), dtype="float64")
                except Exception as exc:
                    # One broken model must not void a whole run, exactly as in the
                    # method registry: the failure becomes a recorded refusal.
                    rows.append(
                        _row(
                            model_id=model_id,
                            symbol=str(symbol),
                            horizon=horizon,
                            fold=fold.index,
                            metric="refused",
                            reason=f"model raised {type(exc).__name__}: {exc}",
                            train_rows=len(labelled),
                            non_overlapping_windows=report["non_overlapping_windows"],
                            panel_vintage_ts=panel_vintage_ts,
                        )
                    )
                    continue

                scored = score_all(y_test, predicted, scale)
                pooled[model_id].append(y_test - predicted)
                for metric, value in scored.items():
                    if metric == "n":
                        continue
                    rows.append(
                        _row(
                            model_id=model_id,
                            symbol=str(symbol),
                            horizon=horizon,
                            fold=fold.index,
                            metric=metric,
                            value=value,
                            n_test_rows=int(scored["n"]),
                            train_rows=len(labelled),
                            non_overlapping_windows=report["non_overlapping_windows"],
                            panel_vintage_ts=panel_vintage_ts,
                        )
                    )

        baseline_errors = (
            np.concatenate(pooled[BASELINE_MODEL_ID])
            if pooled.get(BASELINE_MODEL_ID)
            else np.array([])
        )
        for model_id, chunks in pooled.items():
            if model_id == BASELINE_MODEL_ID or not chunks or baseline_errors.size == 0:
                continue
            result = diebold_mariano(np.concatenate(chunks), baseline_errors, horizon)
            for metric, value in (
                ("dm_statistic", result.statistic),
                ("dm_p_value", result.p_value),
            ):
                rows.append(
                    _row(
                        model_id=model_id,
                        symbol=str(symbol),
                        horizon=horizon,
                        fold=-1,
                        metric=metric,
                        value=value,
                        n_test_rows=result.n,
                        non_overlapping_windows=report["non_overlapping_windows"],
                        reason=result.reason,
                        panel_vintage_ts=panel_vintage_ts,
                    )
                )

    out = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    out.attrs["features_used"] = features_used
    return out
