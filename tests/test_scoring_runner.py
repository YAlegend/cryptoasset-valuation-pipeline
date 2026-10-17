import numpy as np
import pandas as pd

import cryptoval.scoring  # noqa: F401
from cryptoval.scoring.registry import MODELS
from cryptoval.scoring.runner import SCORE_COLUMNS, run_walk_forward


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


def test_imputation_is_fitted_on_the_training_fold_only():
    """A median over the whole series is look-ahead dressed as data cleaning."""
    frame = _frame()
    # A feature that is NaN early and enormous late: a whole-series median would leak
    # the late regime into every early fold.
    frame.loc[frame.index[:1200], "metcalfe_estimated"] = np.nan
    frame.loc[frame.index[1200:], "metcalfe_estimated"] = 500.0
    rows = run_walk_forward(frame, _baselines(), horizon=30)
    assert not rows.empty
    assert rows["value"].replace([np.inf, -np.inf], np.nan).dropna().abs().max() < 1e6


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
