import numpy as np
import pandas as pd

import cryptoval.scoring  # noqa: F401
from cryptoval.scoring.base import ModelSpec
from cryptoval.scoring.registry import MODELS
from cryptoval.scoring.runner import SCORE_COLUMNS, run_walk_forward
from cryptoval.scoring.splits import expanding_folds


def _frame(days: int = 2000, symbols=("BTC",)) -> pd.DataFrame:
    rng = np.random.default_rng(20261017)
    out = []
    for symbol in symbols:
        dates = pd.date_range("2019-01-01", periods=days, freq="D")
        signal = rng.normal(size=days)
        out.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "metric_date": dates,
                    "y": 0.01 * signal + rng.normal(scale=0.002, size=days),
                    "daily_log_return": rng.normal(scale=0.01, size=days),
                    "metcalfe_estimated": signal,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def _baselines() -> dict:
    return {k: MODELS[k] for k in ("naive_persistence", "random_walk_drift", "historical_mean")}


def test_the_runner_emits_the_documented_columns_and_grain():
    rows = run_walk_forward(_frame(), _baselines(), horizon=30)
    assert list(rows.columns) == SCORE_COLUMNS
    keyed = rows[rows["fold"] >= 0]
    assert not keyed.duplicated(subset=["model_id", "symbol", "fold", "horizon", "metric"]).any()


def test_every_model_gets_a_dm_row_against_persistence():
    """Spec 11.2: a difference without a p-value is not reported as a difference."""
    rows = run_walk_forward(_frame(), _baselines(), horizon=30)
    pooled = rows[rows["fold"] == -1]
    for model_id in ("random_walk_drift", "historical_mean"):
        got = set(pooled[pooled["model_id"] == model_id]["metric"])
        assert {"dm_statistic", "dm_p_value"} <= got


def test_the_baseline_is_not_tested_against_itself():
    rows = run_walk_forward(_frame(), _baselines(), horizon=30)
    baseline_dm = rows[(rows["model_id"] == "naive_persistence") & (rows["metric"] == "dm_p_value")]
    assert baseline_dm.empty


def test_every_row_carries_the_non_overlapping_sample_size():
    """Review Focus 3: a h=365 score must not read like a h=1 score."""
    rows = run_walk_forward(_frame(days=2200), _baselines(), horizon=365)
    scored = rows[rows["fold"] >= 0]
    assert (scored["non_overlapping_windows"] > 0).all()
    assert scored["non_overlapping_windows"].max() < 10


def test_an_asset_with_too_few_rows_yields_a_refusal_row_not_a_score():
    rows = run_walk_forward(_frame(days=300), _baselines(), horizon=30)
    assert (rows["metric"] == "refused").all()
    assert rows["reason"].str.contains("300 rows").all()
    assert rows["value"].isna().all()


def test_a_feature_below_the_coverage_floor_is_dropped_for_that_fold():
    frame = _frame()
    frame["sparse_signal"] = np.nan
    frame.loc[frame.index[-50:], "sparse_signal"] = 1.0
    rows = run_walk_forward(frame, _baselines(), horizon=30)
    dropped = rows.attrs["features_used"]
    assert all("sparse_signal" not in used for used in dropped.values())


def test_two_symbols_are_scored_independently():
    rows = run_walk_forward(_frame(symbols=("BTC", "ETH")), _baselines(), horizon=30)
    assert set(rows["symbol"]) == {"BTC", "ETH"}


class _FeatureSpy:
    """Records the matrices the runner hands a feature-consuming model.

    The shipped baselines all ignore X, so no score they produce can depend on imputation;
    only a model that actually receives X can show whether it was imputed from the
    training slice or from the whole series.
    """

    spec = ModelSpec(model_id="spy", family="linear", uses_features=True)

    def __init__(self):
        self.train_x: list[np.ndarray] = []
        self.test_x: list[np.ndarray] = []

    def fit(self, X, y, context):
        self.train_x.append(np.array(X))

    def predict(self, X):
        self.test_x.append(np.array(X))
        return np.zeros(len(X))


def test_imputed_cells_receive_the_training_median_not_the_whole_series_median():
    frame = _frame().drop(columns="metcalfe_estimated")
    # A ramp, so each fold's training median differs from the whole-frame median, with every
    # 4th row missing so imputation fires in every training and test slice.
    ramp = np.arange(len(frame), dtype="float64")
    ramp[::4] = np.nan
    frame["ramp"] = ramp
    whole_median = float(np.nanmedian(ramp))

    spy = _FeatureSpy()
    run_walk_forward(frame, {"spy": spy}, horizon=30)

    folds = expanding_folds(len(frame), 30)
    assert len(spy.train_x) == len(spy.test_x) == len(folds)
    for fold, x_train, x_test in zip(folds, spy.train_x, spy.test_x, strict=True):
        train_median = float(np.nanmedian(ramp[fold.train]))
        assert abs(train_median - whole_median) > 100  # the fixture can tell them apart
        for x, raw in ((x_train, ramp[fold.train]), (x_test, ramp[fold.test])):
            imputed = x[np.isnan(raw), 0]
            assert imputed.size > 0
            assert (imputed == train_median).all()
            assert not (imputed == whole_median).any()


def test_a_late_starting_feature_is_excluded_early_and_included_late():
    frame = _frame()
    # Absent for the first 600 rows: below the 0.5 coverage floor for the early folds'
    # training slices, above it for the late ones. Coverage over the WHOLE frame is 70%.
    frame["late_signal"] = np.nan
    frame.loc[frame.index[600:], "late_signal"] = 1.0
    used = run_walk_forward(frame, _baselines(), horizon=30).attrs["features_used"]

    assert "late_signal" not in used["BTC|0"]  # 0 of 365 training rows
    assert "late_signal" not in used["BTC|2"]  # 391 of 991: 39%
    assert "late_signal" in used["BTC|4"]  # 1017 of 1617: 63%
    assert all("metcalfe_estimated" in feats for feats in used.values())  # not vacuously empty
