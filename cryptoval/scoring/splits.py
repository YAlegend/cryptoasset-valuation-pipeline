# cryptoval/scoring/splits.py
"""Expanding-window walk-forward with purging and embargo (spec 11.2).

WHY A PLAIN SPLIT IS WRONG HERE, stated once. The target at row t spans [t, t+h]. A
train/test boundary drawn at row k therefore leaks: the last h training labels were
computed from prices inside the test block, so the model is scored partly on data it was
fitted on. The leak grows with h -- at h=365 a naive boundary hands the model a year of
the answer -- and it is invisible, because every metric improves.

PURGE drops the h rows before the boundary, whose labels reach across it.
EMBARGO drops a further buffer, because daily crypto returns are serially correlated and
adjacency alone carries information past the purge.

No shuffling and no random splitting, anywhere (spec 11.2). Folds are slices into a frame
the caller has already sorted by date, which keeps the time ordering a property of the
data rather than something this module must re-establish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Below this a 5-fold expanding split cannot give any fold a year of training after the
# purge, so the asset is refused rather than folded into degenerate pieces. 730 = two
# years: one to train on, one to spread five test blocks across.
MIN_ROWS_TO_FOLD = 730


@dataclass(frozen=True)
class Fold:
    index: int
    train: slice
    test: slice
    purged: int
    embargoed: int


def expanding_folds(
    n_rows: int,
    horizon: int,
    n_folds: int = 5,
    embargo_days: int = 10,
    min_train_rows: int = 365,
) -> list[Fold]:
    """Time-ordered expanding folds, or [] when the series cannot honestly carry them."""
    if n_rows < MIN_ROWS_TO_FOLD:
        return []

    gap = horizon + embargo_days
    # The usable span is everything a label can be computed for: the last h rows have no
    # forward price, so they can never be tested on.
    usable = n_rows - horizon
    first_test = min_train_rows + gap
    if usable <= first_test:
        return []

    test_size = (usable - first_test) // n_folds
    if test_size <= 0:
        return []

    folds: list[Fold] = []
    for i in range(n_folds):
        test_start = first_test + i * test_size
        test_stop = test_start + test_size if i < n_folds - 1 else usable
        train_stop = test_start - gap
        if train_stop < min_train_rows:
            continue
        folds.append(
            Fold(
                index=i,
                train=slice(0, train_stop),
                test=slice(test_start, test_stop),
                purged=horizon,
                embargoed=embargo_days,
            )
        )
    return folds


def fold_report(
    n_rows: int,
    horizon: int,
    n_folds: int = 5,
    embargo_days: int = 10,
    min_train_rows: int = 365,
) -> dict[str, Any]:
    """What the split yields, including the number that decides whether a model can fit.

    ``non_overlapping_windows`` is the honest sample size. Daily rows at horizon h
    overlap h-fold, so 4,305 rows at h=365 are not 4,305 observations of an annual return
    -- they are about 11.8, and a model fitted on them has seen eleven independent facts.
    Every score row carries this number for exactly that reason.
    """
    folds = expanding_folds(n_rows, horizon, n_folds, embargo_days, min_train_rows)
    if not folds:
        if n_rows < MIN_ROWS_TO_FOLD:
            reason = (
                f"{n_rows} rows at horizon {horizon} cannot carry {n_folds} purged folds "
                f"(minimum {MIN_ROWS_TO_FOLD} rows, {min_train_rows} training rows per fold)"
            )
        else:
            # Enough rows, but the horizon consumes the tail: name the span it had and the
            # row its first test block needed to reach.
            usable = n_rows - horizon
            first_test = min_train_rows + horizon + embargo_days
            reason = (
                f"{n_rows} rows at horizon {horizon} leaves {usable} usable rows; the first "
                f"test row is {first_test} ({min_train_rows} training + {horizon} purge + "
                f"{embargo_days} embargo), and {n_folds} test blocks need "
                f"{first_test + n_folds} usable rows"
            )
        return {
            "folds": 0,
            "reason": reason,
            "non_overlapping_windows": 0.0,
            "smallest_train_rows": 0,
            "test_rows": 0,
        }
    smallest_train = min(f.train.stop - f.train.start for f in folds)
    return {
        "folds": len(folds),
        "reason": "",
        # n_rows / h, not train_rows / h: the question is how many INDEPENDENT
        # observations of an h-day return the series contains at all. Scaling by the
        # smallest training fold would return min_train_rows / h on every input, which
        # is a constant wearing a measurement's name.
        "non_overlapping_windows": round(n_rows / horizon, 1),
        "smallest_train_rows": smallest_train,
        "test_rows": sum(f.test.stop - f.test.start for f in folds),
    }
