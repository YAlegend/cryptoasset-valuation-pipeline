# tests/test_scoring_splits.py
import pytest

from cryptoval.scoring.splits import (
    MIN_ROWS_TO_FOLD,
    expanding_folds,
    fold_report,
)


def test_folds_are_time_ordered_and_never_overlap():
    folds = expanding_folds(n_rows=3000, horizon=30)
    assert len(folds) == 5
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.test.stop <= later.test.start
        # Expanding, not rolling: every train set starts at row 0 and grows.
        assert later.train.start == 0
        assert later.train.stop > earlier.train.stop


def test_the_purge_removes_exactly_the_rows_whose_label_reaches_the_test_block():
    horizon = 30
    embargo = 10
    folds = expanding_folds(n_rows=3000, horizon=horizon, embargo_days=embargo)
    for fold in folds:
        assert fold.train.stop == fold.test.start - horizon - embargo
        assert fold.purged == horizon
        assert fold.embargoed == embargo
        # The decisive property: no training label can reach into the test block.
        assert fold.train.stop + horizon <= fold.test.start


def test_a_longer_horizon_purges_more():
    # The LAST fold: the first one's train always stops at min_train_rows by construction,
    # whatever the horizon, so it cannot show the purge growing.
    short = expanding_folds(n_rows=3000, horizon=1)[-1]
    long = expanding_folds(n_rows=3000, horizon=90)[-1]
    assert long.train.stop < short.train.stop  # 2321 < 2461
    assert long.purged == 90


def test_an_asset_with_too_few_rows_is_refused_with_its_measured_row_count():
    """Review Focus 2: TRX and SOL carry 364 price rows; POL carries 1,085."""
    report = fold_report(n_rows=364, horizon=30)
    assert report["folds"] == 0
    assert "364 rows" in report["reason"]
    assert str(MIN_ROWS_TO_FOLD) in report["reason"]
    assert expanding_folds(n_rows=364, horizon=30) == []


def test_a_fold_that_would_train_on_too_little_is_not_emitted():
    folds = expanding_folds(n_rows=1200, horizon=365, min_train_rows=365)
    for fold in folds:
        assert fold.train.stop - fold.train.start >= 365
        assert fold.test.stop - fold.test.start > 0


def test_the_fold_report_states_non_overlapping_sample_size_per_horizon():
    """Review Focus 3: BTC's 4,305 rows are ~11.8 independent annual windows."""
    report = fold_report(n_rows=4305, horizon=365)
    assert report["non_overlapping_windows"] == pytest.approx(11.0, abs=1.0)
    assert report["non_overlapping_windows"] < 30
    daily = fold_report(n_rows=4305, horizon=1)
    assert daily["non_overlapping_windows"] > 1000


def test_a_refusal_for_too_few_rows_names_the_minimum_not_the_span():
    report = fold_report(n_rows=500, horizon=30)
    assert report["folds"] == 0
    assert "500 rows" in report["reason"]
    assert f"minimum {MIN_ROWS_TO_FOLD} rows" in report["reason"]
    assert "usable" not in report["reason"]


def test_a_refusal_caused_by_the_horizon_names_the_span_and_the_first_test_row():
    # 1000 rows clears the 730 minimum, so the minimum must not be the stated cause.
    report = fold_report(n_rows=1000, horizon=365)
    assert report["folds"] == 0
    assert "1000 rows" in report["reason"]
    assert "635 usable rows" in report["reason"]
    assert "first test row is 740" in report["reason"]
    assert "minimum 730" not in report["reason"]


@pytest.mark.parametrize("n_rows", [730, 1200, 3000, 4305])
@pytest.mark.parametrize("horizon", [1, 7, 30, 90, 365])
@pytest.mark.parametrize("embargo_days", [0, 10])
def test_the_purge_invariant_holds_across_the_grid(n_rows, horizon, embargo_days):
    folds = expanding_folds(n_rows=n_rows, horizon=horizon, embargo_days=embargo_days)
    if (n_rows, horizon) == (730, 365):
        # The usable span of 365 rows never reaches the first test row, so nothing is emitted.
        assert folds == []
        return
    assert len(folds) == 5
    for fold in folds:
        assert fold.train.start == 0
        # The purge: no training label reaches into the test block.
        assert fold.train.stop + horizon <= fold.test.start
        assert fold.test.stop <= n_rows - horizon
        assert fold.test.stop - fold.test.start > 0
