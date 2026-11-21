# Cryptoasset Valuation — The Statistical Rebuild Implementation Plan (Plan 4 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build spec §11's walk-forward scoring layer — baselines established first, a purged and
embargoed expanding-window protocol, the §11.3 models re-run correctly, Diebold-Mariano against
persistence, and a family-wise correction over the whole set of hypotheses — so that "does any of
this beat a random walk" becomes a SQL query against `fct_model_score` rather than a claim.

**Architecture:** Three layers, mirroring the one Plan 2 built for methods. **The protocol layer**
(Tasks 1-4) turns the panel into a point-in-time feature matrix and a forward log-return target,
cuts it into purged expanding folds, and defines the metrics and the DM test that score a
prediction — none of it knowing what a model is. **The model layer** (Tasks 5-9) is a registry with
one `Model` protocol and one YAML per model, exactly like `cryptoval/methods/registry.py`: baselines
register through the same interface as a gradient-booster, so "did it beat the random walk" is a
comparison between two rows rather than between a result and an assumption. **The discipline layer**
(Tasks 10-12) counts the hypotheses, corrects for them, persists the mart and writes the chapter.

**Tech Stack:** Python 3.14.5, pandas 3.0.6, numpy 2.5.4, DuckDB 1.5.6, dbt-core 1.12.5 with
dbt-duckdb 1.11.0, pyarrow 26.0.0, statsmodels 0.15.0, scipy 1.18.1, pytest 9.1.1, ruff 0.17.0.
**One new dependency: `scikit-learn==1.9.1`** (cp314 wheels confirmed available 2026-10-17), which
supplies every §11.3 model but the recurrent one: OLS, ridge, elastic net, SVR with RBF and
polynomial kernels, gradient boosting, and the feed-forward NN via `MLPRegressor`.

**Spec:** `docs/superpowers/specs/2026-10-11-cryptoasset-valuation-pipeline-design.md` (revised
2026-10-12). This plan implements §11.1, §11.2, §11.3 and §11.6 under the Phase 4 row of §13, and
meets that row's exit criterion in full: *every model has a DM p-value against persistence;
`fct_model_score` populated including baselines.*

**Scope boundary — read this before starting.** Spec §11 contains three subsystems that write three
different tables, and this plan is **one** of them.

- **§11.4 (econometrics: ADF/KPSS, Engle-Granger, Johansen, VECM, GARCH/EGARCH, Markov-switching,
  DCC-GARCH) is NOT in this plan.** It writes `fct_econometric_test`, answers a different question
  ("is there a long-run equilibrium with fundamentals?" rather than "does a forecast beat a random
  walk?"), and needs `arch==8.0.0` plus a hand-rolled DCC estimator that no free package supplies
  well. It depends on nothing here and nothing here depends on it. **Plan 4b.**
- **§11.5 (cross-sectional asset pricing: Liu-Tsyvinski-Wu three-factor, Fama-MacBeth, decile
  portfolios) is NOT in this plan.** It writes `fct_factor_score`, and it carries an unresolved
  data problem this plan should not inherit: §11.5 and §5 both require a **point-in-time top-N
  universe** rather than the curated 24, and no free source serves historical market-cap rankings.
  Plan 4c must open by deciding between ranking within the 24 by point-in-time market cap (with the
  survivorship limitation stated loudly) and declaring the universe requirement a gap. **Plan 4c.**
- **§11.6's correction is applied here over this plan's own hypotheses only.** The count is written
  to the mart as a row, so Plans 4b and 4c can re-correct over the union later. Doing it once over
  all three would mean no plan could ship a corrected number until the last one landed.

## Global Constraints

Copied verbatim from the spec; every task's requirements implicitly include this section.

- **Target is log returns, never price levels.** "A model predicting levels from levels is the
  dissertation's error restated" (§11.2). Horizons h ∈ {1, 7, 30, 90, 365} days.
- **No `train_test_split`, no shuffling, anywhere in the codebase** (§11.2). A test enforces this by
  grepping the source tree.
- **Splitting is expanding-window walk-forward, strictly time-ordered, with purging and embargo**
  around each fold boundary, so overlapping horizon windows cannot leak across the split (§11.2).
- **Feature hygiene:** no feature may be a transformation of the target at or after the prediction
  timestamp; all features point-in-time via §8.2 (§11.2).
- **A difference without a p-value is not reported as a difference** (§11.2). Every model carries a
  Diebold-Mariano statistic against the persistence baseline.
- **Baselines before models** (§11.1). Naive persistence, random walk with drift, historical mean
  are registered models and are scored by the same code path as everything else.
- **The expectation, stated in advance, is that most models will not beat persistence at most
  horizons. That is a result, not a failure** (§11.3).
- **The report records the total number of hypotheses tested** and applies a family-wise correction
  to headline claims (§11.6).
- `line-length = 100`, ruff `select = ["E", "F", "I", "UP", "B"]`, `target-version = "py314"`.
- Every refusal carries the measurement that caused it. A reason without a number is not actionable
  — the rule `cryptoval/methods/base.py` already enforces for methods, enforced here for scores.

## Review Focus

Five failure modes the spec implies that no task's obvious tests would exercise. Each line's test is
assigned to the task that owns the code.

1. **A feature that is the target.** `crypto_qtm_panel_velocity` returns the market price to machine
   epsilon, so its signal `log(fair_value/price)` is identically zero; `mayer_multiple` and
   `puell_multiple` are documented as using an *inclusive* window, so their signal on day *t* has
   seen day *t*'s price. A reasonable person expects a leaked feature to be impossible by
   construction, not caught by inspection. → **Task 1** (`test_every_feature_is_lagged_at_least_one_day`,
   `test_a_zero_variance_signal_is_dropped_with_its_measured_variance`).
2. **An asset with too little history to fold.** TRX and SOL carry 364 price rows (CoinGecko only);
   POL carries 1,085. A reasonable person expects a refusal naming the row count, not three
   degenerate folds and a confident MASE. → **Task 2**
   (`test_an_asset_with_too_few_rows_is_refused_with_its_measured_row_count`).
3. **h=365 on an 11-year series.** BTC's 4,305 rows are ~11.8 *non-overlapping* annual windows. A
   reasonable person expects a h=365 score not to read as comparable to a h=1 score. → **Task 2**
   (`test_the_fold_report_states_non_overlapping_sample_size_per_horizon`).
4. **MASE when the naive in-sample error is zero.** USDC is a dollar peg: its one-day log returns
   are ~1e-6 and can be exactly 0 over a flat training fold, making MASE's denominator zero and the
   metric `inf` or `NaN` silently. → **Task 3** (`test_mase_refuses_a_zero_naive_denominator`).
5. **A test block with no directional variation.** A fold whose test returns are all one sign, or
   all exactly zero, makes directional accuracy 1.0 or undefined while looking like a strong result.
   → **Task 3** (`test_directional_accuracy_reports_its_usable_count_and_refuses_an_all_zero_block`).

## File Structure

```
cryptoval/scoring/__init__.py       importing the package registers every model family
cryptoval/scoring/base.py           ModelSpec, Model protocol, Prediction, Score
cryptoval/scoring/targets.py        forward log-return target + point-in-time feature matrix
cryptoval/scoring/splits.py         purged, embargoed, expanding-window folds + the fold report
cryptoval/scoring/metrics.py        MAE, RMSE, MASE, directional accuracy
cryptoval/scoring/dm.py             Diebold-Mariano with the Harvey-Leybourne-Newbold correction
cryptoval/scoring/registry.py       MODELS, register, load_model_specs, run_walk_forward
cryptoval/scoring/baselines.py      naive_persistence, random_walk_drift, historical_mean
cryptoval/scoring/linear.py         ols, ridge, elastic_net
cryptoval/scoring/nonlinear.py      svr_rbf, svr_poly, gradient_boosting, feedforward_nn
cryptoval/scoring/multiplicity.py   hypothesis count + Holm-Bonferroni FWER
config/models/*.yml                 one per model: id, family, params, citation
dbt/models/marts/fct_model_score.sql
reports/does-anything-beat-a-random-walk.md
tests/test_scoring_targets.py  test_scoring_splits.py  test_scoring_metrics.py
tests/test_scoring_dm.py  test_scoring_registry.py  test_scoring_models.py
tests/test_scoring_multiplicity.py
```

Boundaries follow the method layer's, which has held through two plans: the protocol knows nothing
about models, each model family is one module whose import registers it, and the registry is the
only module that knows both. `cryptoval/scoring/` mirrors `cryptoval/methods/` deliberately — an
executor who has read one can read the other.

---

# Stage 4-1 — The protocol, which knows nothing about models

## Task 1: The forward log-return target and the point-in-time feature matrix

Spec §11.2 fixes the target as log returns at h ∈ {1, 7, 30, 90, 365} and forbids price levels. The
features are this project's own output: for each method, the signal `log(fair_value / price)`, which
is what "does a valuation signal predict returns" actually means. Two hazards live here and both
are Review Focus items — a signal that is algebraically the price, and a signal computed with an
inclusive window that has already seen day *t*.

The blanket defence is one line: **every feature is lagged one day.** A prediction stamped *t* sees
features as of *t-1*. That makes an inclusive-window leak impossible by construction rather than by
audit, costs one day of signal, and needs no per-method special-casing.

**Files:**
- Create: `cryptoval/scoring/__init__.py`, `cryptoval/scoring/targets.py`
- Test: `tests/test_scoring_targets.py`

**Interfaces:**
- Consumes: `fct_asset_metrics_daily` (`symbol`, `metric_date`, `price_usd`) and
  `fct_valuation_estimate` (`symbol`, `metric_date`, `method_id`, `fair_value`, `applicable`).
- Produces: `HORIZONS: tuple[int, ...]`; `forward_log_return(price: pd.Series, horizon: int) ->
  pd.Series`; `signal_matrix(estimates: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame`;
  `build_frame(panel: pd.DataFrame, estimates: pd.DataFrame, horizon: int) -> pd.DataFrame` with
  columns `symbol, metric_date, y, <one column per method_id>`; `FEATURE_LAG_DAYS: int = 1`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring_targets.py
import numpy as np
import pandas as pd
import pytest

from cryptoval.scoring.targets import (
    FEATURE_LAG_DAYS,
    HORIZONS,
    build_frame,
    forward_log_return,
    signal_matrix,
)


def _price_frame(days: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": dates,
            "price_usd": 100.0 * np.exp(np.linspace(0.0, 1.0, days)),
        }
    )


def _estimates(price: pd.DataFrame, method_id: str, ratio) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": price["symbol"],
            "metric_date": price["metric_date"],
            "method_id": method_id,
            "fair_value": price["price_usd"] * ratio,
            "applicable": True,
        }
    )


def test_the_target_is_a_forward_log_return_not_a_level():
    price = pd.Series([100.0, 110.0, 121.0, 133.1])
    y = forward_log_return(price, horizon=1)
    assert y.iloc[0] == pytest.approx(np.log(1.1))
    # The last h rows have no future price, so they are NaN rather than silently reused.
    assert np.isnan(y.iloc[-1])


def test_the_target_at_horizon_h_spans_h_days():
    price = pd.Series(100.0 * np.exp(np.arange(10) * 0.01))
    assert forward_log_return(price, horizon=3).iloc[0] == pytest.approx(0.03)


def test_horizons_are_the_five_the_spec_fixes():
    assert HORIZONS == (1, 7, 30, 90, 365)


def test_every_feature_is_lagged_at_least_one_day():
    """Review Focus 1: an inclusive-window signal must not see the day it predicts."""
    price = _price_frame()
    # A varying ratio: a constant one is dropped for zero variance by the test below.
    estimates = _estimates(price, "mayer_multiple", np.linspace(1.2, 1.8, len(price)))
    frame = build_frame(price, estimates, horizon=1)
    signals = signal_matrix(estimates, price).set_index("metric_date")["mayer_multiple"]
    merged = frame.set_index("metric_date")["mayer_multiple"].dropna()
    assert FEATURE_LAG_DAYS >= 1
    for date, value in merged.items():
        assert value == pytest.approx(signals.loc[date - pd.Timedelta(days=FEATURE_LAG_DAYS)])


def test_a_zero_variance_signal_is_dropped_with_its_measured_variance():
    """Review Focus 1: crypto_qtm_panel_velocity returns the price, so its signal is 0.0."""
    price = _price_frame()
    estimates = pd.concat(
        [
            _estimates(price, "crypto_qtm_panel_velocity", 1.0),  # fair_value == price
            _estimates(price, "metcalfe_estimated", np.linspace(0.5, 2.0, len(price))),
        ],
        ignore_index=True,
    )
    frame = build_frame(price, estimates, horizon=1)
    assert "crypto_qtm_panel_velocity" not in frame.columns
    assert "metcalfe_estimated" in frame.columns
    assert frame.attrs["dropped_features"]["crypto_qtm_panel_velocity"].startswith("variance ")


def test_an_inapplicable_estimate_contributes_no_signal():
    price = _price_frame()
    estimates = _estimates(price, "nvt_reversion", np.linspace(1.2, 1.8, len(price)))
    estimates.loc[estimates.index[:200], "applicable"] = False
    estimates.loc[estimates.index[:200], "fair_value"] = None
    frame = build_frame(price, estimates, horizon=1)
    assert frame["nvt_reversion"].isna().sum() >= 200


def test_a_non_positive_fair_value_yields_no_signal_rather_than_a_nan_log():
    price = _price_frame(60)
    estimates = _estimates(price, "thermocap_floor", 1.0)
    estimates.loc[estimates.index[:10], "fair_value"] = -5.0
    signals = signal_matrix(estimates, price)
    assert signals["thermocap_floor"].iloc[:10].isna().all()
    assert signals["thermocap_floor"].iloc[10:].notna().any()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring_targets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.scoring'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/scoring/__init__.py
"""Walk-forward scoring (spec 11.1-11.3, 11.6).

Importing the package registers every model family. The method layer learned this the
hard way -- a registration list kept in two places drifts -- so there is one copy here,
next to the modules it names.
"""

import cryptoval.scoring.baselines  # noqa: F401 -- import registers the family
import cryptoval.scoring.linear  # noqa: F401
import cryptoval.scoring.nonlinear  # noqa: F401
```

```python
# cryptoval/scoring/targets.py
"""The target and the features (spec 11.2).

TWO RULES, settled here so that seven models do not each decide them:

* The target is a FORWARD LOG RETURN, never a level. log(P[t+h] / P[t]), NaN in the last
  h rows because there is no future price to divide by. Spec 11.2 is explicit that a model
  predicting levels from levels is the dissertation's error restated.
* Every feature is LAGGED ONE DAY. The features are this project's own fair values, and
  two classes of them have already seen day t's price: crypto_qtm_panel_velocity returns
  the market price to machine epsilon, and mayer_multiple and puell_multiple are
  documented as using an inclusive window because their published definitions are
  inclusive. Lagging everything makes that leak impossible by construction instead of by
  per-method audit, at a cost of one day of signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS: tuple[int, ...] = (1, 7, 30, 90, 365)

FEATURE_LAG_DAYS = 1

# A signal flatter than this carries no information a model could use, and at least one
# is flat BY CONSTRUCTION (a circular method's fair value is the price). Dropping it with
# its measured variance beats handing a regression a constant column.
MIN_SIGNAL_VARIANCE = 1e-12


def forward_log_return(price: pd.Series, horizon: int) -> pd.Series:
    """log(P[t+h] / P[t]), with the last h rows NaN.

    Positive prices only: a non-positive or missing price yields NaN rather than a
    complex or infinite return.
    """
    p = pd.to_numeric(price, errors="coerce").astype("float64")
    p = p.where(p > 0.0)
    return np.log(p.shift(-horizon) / p)


def signal_matrix(estimates: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """One column per method: log(fair_value / price), NOT yet lagged.

    The ratio rather than the level, because a fair value in dollars is not comparable
    across assets and a model fitted per asset would otherwise learn the price scale.
    """
    usable = estimates[
        estimates["applicable"].astype("boolean").fillna(False)
        & estimates["fair_value"].notna()
    ].copy()
    usable["fair_value"] = pd.to_numeric(usable["fair_value"], errors="coerce")
    usable = usable[usable["fair_value"] > 0.0]

    wide = usable.pivot_table(
        index=["symbol", "metric_date"],
        columns="method_id",
        values="fair_value",
        aggfunc="median",  # a grid method emits one row per param set; the median is the cell
    ).reset_index()

    merged = price[["symbol", "metric_date", "price_usd"]].merge(
        wide, on=["symbol", "metric_date"], how="left"
    )
    methods = [c for c in wide.columns if c not in ("symbol", "metric_date")]
    base = pd.to_numeric(merged["price_usd"], errors="coerce").where(lambda s: s > 0.0)
    for method_id in methods:
        merged[method_id] = np.log(merged[method_id] / base)
    return merged.drop(columns=["price_usd"])


def build_frame(panel: pd.DataFrame, estimates: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """(symbol, metric_date, y, <feature per method>) for one horizon, lagged and sorted.

    ``frame.attrs["dropped_features"]`` maps a dropped method id to the measurement that
    dropped it, so a missing column is explained rather than merely absent.
    """
    price = panel[["symbol", "metric_date", "price_usd"]].sort_values(
        ["symbol", "metric_date"]
    )
    signals = signal_matrix(estimates, price).sort_values(["symbol", "metric_date"])
    methods = [c for c in signals.columns if c not in ("symbol", "metric_date")]

    # The lag, applied within each symbol so one asset's last row cannot feed the next
    # asset's first. groupby(...).shift is the only correct way to say that.
    signals[methods] = signals.groupby("symbol", sort=False)[methods].shift(FEATURE_LAG_DAYS)

    out = price.merge(signals, on=["symbol", "metric_date"], how="left")
    out["y"] = out.groupby("symbol", sort=False)["price_usd"].transform(
        lambda s: forward_log_return(s, horizon)
    )

    dropped: dict[str, str] = {}
    for method_id in methods:
        variance = float(out[method_id].var(skipna=True)) if out[method_id].notna().any() else 0.0
        if not np.isfinite(variance) or variance < MIN_SIGNAL_VARIANCE:
            dropped[method_id] = f"variance {variance:.3e} below {MIN_SIGNAL_VARIANCE:.0e}"
            out = out.drop(columns=[method_id])

    out = out.drop(columns=["price_usd"]).reset_index(drop=True)
    out.attrs["dropped_features"] = dropped
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scoring_targets.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/scoring tests/test_scoring_targets.py
git commit -m "feat: the forward log-return target and a feature matrix that cannot leak

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Purged, embargoed, expanding-window folds — and the measurement that decides Task 9

Spec §11.2 requires expanding-window walk-forward with purging and embargo. The reason is specific:
a training row at *t* carries a label that spans *[t, t+h]*, so the last *h* training rows have
labels reaching into the test block. Dropping them is the purge. The embargo drops a further buffer
for serial correlation.

This task also carries the measurement that decides Task 9. The question is not how many training
*rows* exist but how many **non-overlapping** target windows they contain: BTC's 4,305 rows are
4,305 sequences at h=365 and about **11.8 independent annual observations**. A model with thousands
of parameters fitted on eleven effective observations memorises them.

**Files:**
- Create: `cryptoval/scoring/splits.py`
- Test: `tests/test_scoring_splits.py`

**Interfaces:**
- Consumes: nothing from Task 1 at runtime; the frame shape it folds is Task 1's output.
- Produces: `@dataclass(frozen=True) Fold(index: int, train: slice, test: slice, purged: int,
  embargoed: int)`; `expanding_folds(n_rows: int, horizon: int, n_folds: int = 5, embargo_days: int
  = 10, min_train_rows: int = 365) -> list[Fold]`; `fold_report(n_rows: int, horizon: int, **kw) ->
  dict[str, Any]`; `MIN_ROWS_TO_FOLD: int = 730`.

- [ ] **Step 1: Write the failing tests**

```python
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
    for earlier, later in zip(folds, folds[1:], strict=True):
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring_splits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.scoring.splits'`.

- [ ] **Step 3: Write the implementation**

```python
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

No shuffling, no train_test_split, anywhere (spec 11.2). Folds are slices into a frame
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
        return {
            "folds": 0,
            "reason": (
                f"{n_rows} rows at horizon {horizon} cannot carry {n_folds} purged folds "
                f"(minimum {MIN_ROWS_TO_FOLD} rows, {min_train_rows} training rows per fold)"
            ),
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scoring_splits.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: MEASURE — the number that decides Task 9**

This is a measurement step, not a placeholder: it names the number to take, where it goes, and what
each outcome means. Run it and **paste the output into the Task 9 brief and the task report.**

```bash
.venv/bin/python - <<'PY'
import duckdb
from cryptoval.scoring.splits import fold_report
from cryptoval.scoring.targets import HORIZONS

con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
rows = con.execute("""
  select symbol, count(*) filter (where price_usd is not null) n
  from fct_asset_metrics_daily group by 1 order by n desc
""").fetchdf()

print(f"{'symbol':8} {'rows':>6} " + " ".join(f"h={h:<4}" for h in HORIZONS))
qualifying = {h: 0 for h in HORIZONS}
for _, r in rows.iterrows():
    cells = []
    for h in HORIZONS:
        rep = fold_report(int(r.n), h)
        cells.append(f"{rep['folds']}f/{rep['non_overlapping_windows']:.0f}w")
        # The threshold, stated in advance: a recurrent net needs folds AND independent
        # observations, not overlapping daily windows of the same decade.
        if rep["folds"] >= 5 and rep["non_overlapping_windows"] >= 30:
            qualifying[h] += 1
    print(f"{r.symbol:8} {int(r.n):6d} " + " ".join(f"{c:<6}" for c in cells))
print("\nassets clearing >=5 folds AND >=30 non-overlapping windows:")
for h, n in qualifying.items():
    print(f"  h={h:<4} {n:2d} of {len(rows)} assets")
PY
```

**What each outcome means.** A horizon where **8 or more assets** clear both thresholds can
honestly carry a recurrent model, and Task 9 implements the GRU for those horizons only. A horizon
where fewer clear becomes a declared-gap row carrying these counts. On the 2026-10-14 vintage the
expected shape is that h=1 and h=7 clear comfortably, h=30 and h=90 clear for the long-history
assets, and **h=365 clears for none** — BTC's 4,305 rows give ~11.8 annual windows against a
threshold of 30. If the measurement contradicts that, follow the measurement, not this paragraph.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/scoring/splits.py tests/test_scoring_splits.py
git commit -m "feat: purged, embargoed expanding folds, and the sample size that decides the GRU

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: The four metrics, each refusing rather than returning a meaningless number

Spec §11.2 fixes the metrics: MAE, RMSE, MASE (scaled against naive), directional accuracy. Two of
them have a degenerate case that returns a confident number instead of an error, and both are
Review Focus items: MASE divides by the naive in-sample error, which is ~0 for a dollar peg; and
directional accuracy is 1.0 on a test block whose returns all share a sign.

**Files:**
- Create: `cryptoval/scoring/metrics.py`
- Test: `tests/test_scoring_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `mae(y, yhat) -> float | None`; `rmse(y, yhat) -> float | None`;
  `mase(y, yhat, naive_scale) -> float | None`; `naive_scale(y_train) -> float | None`;
  `directional_accuracy(y, yhat) -> tuple[float | None, int]`;
  `score_all(y, yhat, naive_scale) -> dict[str, float | None]` returning keys
  `{"mae", "rmse", "mase", "directional_accuracy", "usable_direction_rows", "n"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring_metrics.py
import numpy as np
import pytest

from cryptoval.scoring.metrics import (
    directional_accuracy,
    mae,
    mase,
    naive_scale,
    rmse,
    score_all,
)


def test_mae_and_rmse_on_a_known_error_vector():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.5, 1.5, 4.5])
    assert mae(y, yhat) == pytest.approx(0.8333333, abs=1e-6)
    assert rmse(y, yhat) == pytest.approx(np.sqrt((0.25 + 0.25 + 2.25) / 3))


def test_a_perfect_forecast_scores_zero_error_and_full_direction():
    y = np.array([0.01, -0.02, 0.03])
    assert mae(y, y) == pytest.approx(0.0)
    accuracy, usable = directional_accuracy(y, y)
    assert accuracy == pytest.approx(1.0)
    assert usable == 3


def test_mase_below_one_means_it_beat_predicting_zero():
    y = np.array([0.02, -0.02, 0.02, -0.02])
    scale = naive_scale(y)  # mean|y| under the zero forecast
    assert scale == pytest.approx(0.02)
    assert mase(y, y * 0.5, scale) == pytest.approx(0.5)
    assert mase(y, np.zeros_like(y), scale) == pytest.approx(1.0)


def test_mase_refuses_a_zero_naive_denominator():
    """Review Focus 4: USDC is a dollar peg; a flat training fold makes the scale 0."""
    assert naive_scale(np.zeros(100)) is None
    assert mase(np.array([0.0, 0.0]), np.array([0.0, 0.0]), None) is None
    assert mase(np.array([0.1]), np.array([0.0]), 0.0) is None


def test_directional_accuracy_reports_its_usable_count_and_refuses_an_all_zero_block():
    """Review Focus 5: a block with no directional variation is not 100% accuracy."""
    accuracy, usable = directional_accuracy(np.zeros(50), np.ones(50))
    assert accuracy is None
    assert usable == 0

    y = np.array([0.01, 0.0, -0.01, 0.0])
    accuracy, usable = directional_accuracy(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert usable == 2  # the two zero-return rows have no direction to get right
    assert accuracy == pytest.approx(0.5)


def test_every_metric_ignores_nan_pairs_rather_than_propagating_them():
    y = np.array([0.01, np.nan, -0.01])
    yhat = np.array([0.01, 0.5, np.nan])
    out = score_all(y, yhat, naive_scale=0.01)
    assert out["n"] == 1
    assert out["mae"] == pytest.approx(0.0)


def test_score_all_returns_none_for_every_metric_on_an_empty_overlap():
    out = score_all(np.array([np.nan, np.nan]), np.array([1.0, 2.0]), naive_scale=0.01)
    assert out["n"] == 0
    assert out["mae"] is None and out["rmse"] is None and out["mase"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.scoring.metrics'`.

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scoring_metrics.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/scoring/metrics.py tests/test_scoring_metrics.py
git commit -m "feat: the four walk-forward metrics, each refusing its degenerate case

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: Diebold-Mariano against the baseline

Spec §11.2: *a difference without a p-value is not reported as a difference.* statsmodels ships no
DM test, so it is written here — about forty lines, and the only new statistics in this plan.

The horizon matters to the test itself, not only to the data: *h*-step forecast errors are
serially correlated up to lag *h-1*, so the variance of the loss differential needs a HAC
(Newey-West) estimate over those lags. The Harvey-Leybourne-Newbold correction then adjusts for
the small-sample bias that makes the uncorrected statistic over-reject.

**Files:**
- Create: `cryptoval/scoring/dm.py`
- Test: `tests/test_scoring_dm.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `@dataclass(frozen=True) DMResult(statistic: float | None, p_value: float | None,
  n: int, reason: str)`; `diebold_mariano(errors_model, errors_baseline, horizon: int, power: int =
  2) -> DMResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring_dm.py
import numpy as np
import pytest

from cryptoval.scoring.dm import diebold_mariano


def _rng():
    return np.random.default_rng(20261017)


def test_two_identical_error_series_are_not_different():
    e = _rng().normal(size=500)
    result = diebold_mariano(e, e.copy(), horizon=1)
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.n == 500


def test_a_clearly_better_model_gets_a_negative_statistic_and_a_small_p():
    rng = _rng()
    baseline = rng.normal(scale=1.0, size=800)
    model = rng.normal(scale=0.3, size=800)
    result = diebold_mariano(model, baseline, horizon=1)
    # Negative: the model's loss is lower than the baseline's.
    assert result.statistic < 0
    assert result.p_value < 0.01


def test_a_clearly_worse_model_gets_a_positive_statistic():
    rng = _rng()
    baseline = rng.normal(scale=0.3, size=800)
    model = rng.normal(scale=1.0, size=800)
    assert diebold_mariano(model, baseline, horizon=1).statistic > 0


def test_a_longer_horizon_widens_the_variance_and_shrinks_the_statistic():
    """h-step errors are autocorrelated to lag h-1; ignoring that over-rejects."""
    rng = _rng()
    # Strongly autocorrelated errors -- exactly what overlapping h-step windows produce,
    # and exactly what a plain sample variance understates. With iid errors the HAC sum
    # is noise and this assertion would be a coin flip.
    baseline = np.convolve(rng.normal(size=849), np.ones(50) / 50, mode="valid") + 1.0
    model = baseline * 0.8
    at_1 = diebold_mariano(model, baseline, horizon=1)
    at_30 = diebold_mariano(model, baseline, horizon=30)
    # Measured: -99.5 at h=1 against -15.0 at h=30. Both p-values underflow to 0.0 on
    # this input, so the statistic is what the assertion can hold onto.
    assert abs(at_30.statistic) < 0.5 * abs(at_1.statistic)


def test_too_few_observations_is_a_refusal_with_its_count_not_a_p_value():
    result = diebold_mariano(np.ones(4), np.zeros(4), horizon=30)
    assert result.statistic is None and result.p_value is None
    assert "4 paired" in result.reason


def test_equal_magnitude_errors_of_opposite_sign_are_a_tie_not_a_division_by_zero():
    e = np.array([0.5] * 50)
    result = diebold_mariano(e, e * -1.0, horizon=1)  # squared loss identical, d == 0
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_nan_pairs_are_dropped_and_counted():
    model = np.array([0.1, np.nan, 0.3, 0.2] * 50)
    baseline = np.array([0.5, 0.5, np.nan, 0.6] * 50)
    result = diebold_mariano(model, baseline, horizon=1)
    assert result.n == 100
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring_dm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.scoring.dm'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/scoring/dm.py
"""The Diebold-Mariano test (spec 11.2).

Written here because statsmodels ships no DM test, and because the two corrections it
needs are exactly the ones a hand-rolled version omits:

* HAC VARIANCE. h-step-ahead forecast errors are serially correlated up to lag h-1 by
  construction -- overlapping windows share h-1 days of data. Using the plain sample
  variance of the loss differential understates it, sometimes by an order of magnitude at
  h=365, and every model then looks significantly better than the baseline.
* THE HARVEY-LEYBOURNE-NEWBOLD CORRECTION. The uncorrected statistic over-rejects in small
  samples, which is every sample here. HLN scales it and refers it to a t distribution
  with n-1 degrees of freedom rather than the normal.

SIGN CONVENTION, fixed once: d_t = loss(model) - loss(baseline), so a NEGATIVE statistic
means the model loses less than the baseline. "Beat persistence" is statistic < 0 with a
small p-value, and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# Below this the t reference distribution is not worth consulting; the refusal names the
# count so a reader can see which asset-horizon pairs were too short.
MIN_PAIRED_OBSERVATIONS = 10


@dataclass(frozen=True)
class DMResult:
    statistic: float | None
    p_value: float | None
    n: int
    reason: str = ""


def diebold_mariano(errors_model, errors_baseline, horizon: int, power: int = 2) -> DMResult:
    """Test equal predictive accuracy of two forecast error series.

    ``power`` is the loss exponent: 2 for squared error (the default, matching RMSE),
    1 for absolute error (matching MAE).
    """
    a = np.asarray(errors_model, dtype="float64")
    b = np.asarray(errors_baseline, dtype="float64")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = int(a.size)

    if n < MIN_PAIRED_OBSERVATIONS:
        return DMResult(None, None, n, f"only {n} paired observations, minimum {MIN_PAIRED_OBSERVATIONS}")

    d = np.abs(a) ** power - np.abs(b) ** power
    d_bar = float(np.mean(d))
    centred = d - d_bar

    # Newey-West over lags 0..h-1: gamma_0 + 2 * sum gamma_k.
    gamma_0 = float(np.mean(centred**2))
    variance = gamma_0
    for lag in range(1, min(horizon, n)):
        gamma_k = float(np.mean(centred[lag:] * centred[:-lag]))
        variance += 2.0 * gamma_k

    if not math.isfinite(variance) or variance <= 0.0:
        # Exactly equal losses (d identically zero) is not a difference, and a negative
        # HAC estimate is the estimator failing rather than evidence.
        if np.allclose(d, 0.0):
            return DMResult(0.0, 1.0, n, "")
        return DMResult(None, None, n, f"non-positive HAC variance estimate {variance:.3e}")

    statistic = d_bar / math.sqrt(variance / n)

    # Harvey, Leybourne and Newbold (1997), equation 9.
    adjustment = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if adjustment <= 0.0:
        return DMResult(None, None, n, f"HLN adjustment {adjustment:.3e} non-positive at horizon {horizon}")
    statistic *= math.sqrt(adjustment)

    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1)))
    return DMResult(float(statistic), p_value, n, "")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scoring_dm.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/scoring/dm.py tests/test_scoring_dm.py
git commit -m "feat: Diebold-Mariano with HAC variance and the HLN small-sample correction

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

# Stage 4-2 — The model layer, with baselines going first

## Task 5: The model contract, the registry, and the three baselines

Spec §11.1 is an ordering instruction: *baselines, established first.* The way to honour it in code
is to make a baseline a model — same protocol, same registry, same YAML, same rows in the same
table — so that "did it beat the random walk" is a comparison between two rows and cannot be
quietly assumed. This mirrors `cryptoval/methods/registry.py` deliberately.

The three baselines are genuinely distinct estimators and must not collapse into each other.
`naive_persistence` predicts zero (the random walk's own forecast of a future log return).
`historical_mean` predicts the training mean of the *h*-horizon return. `random_walk_drift`
predicts the training mean of the *one-day* return times *h* — which is why `fit` takes a context:
the h-horizon target alone cannot express it, and the two would otherwise be the same model
registered twice.

**Files:**
- Create: `cryptoval/scoring/base.py`, `cryptoval/scoring/registry.py`,
  `cryptoval/scoring/baselines.py`, `config/models/naive_persistence.yml`,
  `config/models/random_walk_drift.yml`, `config/models/historical_mean.yml`
- Modify: `cryptoval/scoring/__init__.py` (created in Task 1; it already imports `baselines`,
  `linear` and `nonlinear` — create the two empty-for-now modules in Task 7 and Task 8, and until
  then comment out their imports with the task number that restores them)
- Test: `tests/test_scoring_registry.py`

**Interfaces:**
- Consumes: `stable_hash` from `cryptoval.hashing`.
- Produces: `@dataclass(frozen=True) ModelSpec(model_id, family, params, citation, uses_features)`;
  `class Model(Protocol)` with `spec`, `fit(X: np.ndarray, y: np.ndarray, context: dict) -> None`
  and `predict(X: np.ndarray) -> np.ndarray`; `MODELS: dict[str, Model]`; `register(cls) -> type`;
  `register_instance(model) -> Model`; `load_model_specs(config_dir="config/models") ->
  dict[str, ModelSpec]`; `param_set_hash(params) -> str`; `BASELINE_MODEL_ID = "naive_persistence"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring_registry.py
import numpy as np
import pytest

import cryptoval.scoring  # noqa: F401 -- import registers every model family
from cryptoval.scoring.base import ModelSpec
from cryptoval.scoring.registry import (
    BASELINE_MODEL_ID,
    MODELS,
    load_model_specs,
    register_instance,
)


def _context(daily=None, horizon=1):
    return {"horizon": horizon, "daily_log_return": np.zeros(100) if daily is None else daily}


def test_every_registered_model_has_a_config_and_vice_versa():
    assert set(MODELS) == set(load_model_specs())


def test_the_baseline_is_registered_like_any_other_model():
    """Spec 11.1: a baseline must be scored by the same code path, not assumed."""
    assert BASELINE_MODEL_ID in MODELS
    assert MODELS[BASELINE_MODEL_ID].spec.family == "baseline"


def test_naive_persistence_predicts_no_return():
    model = MODELS["naive_persistence"]
    model.fit(np.zeros((100, 3)), np.full(100, 0.4), _context())
    assert np.allclose(model.predict(np.zeros((7, 3))), 0.0)


def test_historical_mean_predicts_the_training_mean_of_the_h_horizon_return():
    model = MODELS["historical_mean"]
    y = np.array([0.1, 0.2, 0.3, 0.4])
    model.fit(np.zeros((4, 2)), y, _context(horizon=30))
    assert model.predict(np.zeros((3, 2))) == pytest.approx(np.full(3, 0.25))


def test_random_walk_drift_scales_the_daily_drift_by_the_horizon():
    model = MODELS["random_walk_drift"]
    daily = np.full(500, 0.001)
    model.fit(np.zeros((500, 2)), np.full(500, 999.0), _context(daily=daily, horizon=30))
    # The h-horizon y is deliberately absurd: this model must read the DAILY series.
    assert model.predict(np.zeros((2, 2))) == pytest.approx(np.full(2, 0.03))


def test_the_three_baselines_are_not_the_same_model_twice():
    y = np.array([0.02, 0.04, 0.06, 0.08])
    daily = np.full(4, 0.0005)
    context = _context(daily=daily, horizon=30)
    predictions = []
    for model_id in ("naive_persistence", "random_walk_drift", "historical_mean"):
        model = MODELS[model_id]
        model.fit(np.zeros((4, 1)), y, context)
        predictions.append(float(model.predict(np.zeros((1, 1)))[0]))
    assert len(set(predictions)) == 3


def test_a_baseline_declares_it_needs_no_features():
    assert MODELS["naive_persistence"].spec.uses_features is False


def test_a_duplicate_model_id_is_refused():
    class Duplicate:
        spec = ModelSpec(
            model_id="naive_persistence", family="baseline", params={}, citation="x"
        )

        def fit(self, X, y, context):
            return None

        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(ValueError, match="already registered"):
        register_instance(Duplicate())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.scoring.base'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/scoring/base.py
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
```

```python
# cryptoval/scoring/registry.py
"""Model discovery and registration (spec 11.3).

Deliberately the same shape as cryptoval/methods/registry.py: a dict keyed by id, a class
decorator that registers an INSTANCE, and specs loaded from one YAML per model. An
executor who has read the method registry has read this one.
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

from cryptoval.hashing import stable_hash
from cryptoval.scoring.base import Model, ModelSpec

# The model every other model is tested against (spec 11.1). Named once so the runner,
# the multiple-testing correction and the report cannot disagree about what "the
# baseline" means.
BASELINE_MODEL_ID = "naive_persistence"

MODELS: dict[str, Model] = {}


def register_instance(model: Model) -> Model:
    if model.spec.model_id in MODELS:
        raise ValueError(f"model id already registered: {model.spec.model_id}")
    MODELS[model.spec.model_id] = model
    return model


def register(cls: type) -> type:
    """Class decorator: instantiate and register, returning the class unchanged."""
    register_instance(cls())
    return cls


def param_set_hash(params: dict[str, Any]) -> str:
    return stable_hash(params)


def load_model_specs(config_dir: str = "config/models") -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    for path in sorted(pathlib.Path(config_dir).glob("*.yml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = ModelSpec(
            model_id=raw["model_id"],
            family=raw["family"],
            params=dict(raw.get("params") or {}),
            citation=raw.get("citation", ""),
            uses_features=bool(raw.get("uses_features", True)),
        )
        if spec.model_id in specs:
            raise ValueError(f"duplicate model_id across configs: {spec.model_id}")
        specs[spec.model_id] = spec
    return specs
```

```python
# cryptoval/scoring/baselines.py
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
```

```yaml
# config/models/naive_persistence.yml
model_id: naive_persistence
family: baseline
uses_features: false
params: {}
citation: "Spec 11.1, naive persistence: the random walk's forecast of a future log return is zero"
```

```yaml
# config/models/random_walk_drift.yml
model_id: random_walk_drift
family: baseline
uses_features: false
params: {}
citation: "Spec 11.1, random walk with drift: the mean one-day log return, scaled by the horizon"
```

```yaml
# config/models/historical_mean.yml
model_id: historical_mean
family: baseline
uses_features: false
params: {}
citation: "Spec 11.1, historical mean of the h-horizon log return over the training fold"
```

Update `cryptoval/scoring/__init__.py` to import only what exists so far:

```python
import cryptoval.scoring.baselines  # noqa: F401 -- import registers the family
# import cryptoval.scoring.linear      -- restored in Task 7
# import cryptoval.scoring.nonlinear   -- restored in Task 8
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scoring_registry.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/scoring config/models tests/test_scoring_registry.py
git commit -m "feat: the model contract, its registry, and three baselines that differ

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: The walk-forward runner, which produces the rows

Everything so far is a part; this assembles them. For each (symbol, horizon, model, fold): fit on
the purged training slice, predict the test slice, score it, and emit one row per metric at the
grain spec §8.1 fixes — `(model_id, asset, fold, horizon, metric)`.

Two decisions are settled here rather than by each model. **Imputation is fitted on the training
fold only** — a median taken over the whole series is look-ahead through the back door, and it is
the leak most likely to survive review because it looks like data cleaning. **The feature set is
chosen per fold** from the training slice's own coverage, so a signal that only exists after 2022
cannot silently become a feature for a 2019 fold.

**Files:**
- Create: `cryptoval/scoring/runner.py`
- Test: `tests/test_scoring_runner.py`

**Interfaces:**
- Consumes: `build_frame`, `HORIZONS` (Task 1); `expanding_folds`, `fold_report` (Task 2);
  `score_all`, `naive_scale` (Task 3); `diebold_mariano` (Task 4); `MODELS`, `BASELINE_MODEL_ID`
  (Task 5).
- Produces: `SCORE_COLUMNS: list[str]`; `run_walk_forward(frame: pd.DataFrame, models: dict[str,
  Model], horizon: int, panel_vintage_ts=None) -> pd.DataFrame`;
  `MIN_FEATURE_COVERAGE: float = 0.5`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring_runner.py
import numpy as np
import pandas as pd
import pytest

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.scoring.runner'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/scoring/runner.py
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
                            non_overlapping_windows=report[
                                "non_overlapping_windows"
                            ],
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
                            non_overlapping_windows=report[
                                "non_overlapping_windows"
                            ],
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scoring_runner.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/scoring/runner.py tests/test_scoring_runner.py
git commit -m "feat: the walk-forward runner, with imputation fitted per fold

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

# Stage 4-3 — The models

## Task 7: scikit-learn arrives, and the linear family

The dissertation's OLS is retained for continuity and re-run correctly (§11.3), joined by ridge and
elastic net. All three go through one wrapper, because the only thing a scikit-learn estimator
needs to satisfy this plan's `Model` protocol is a standardising pipeline and a guard for the fold
where no feature cleared the coverage floor.

This task introduces `Unfittable`. Until now every model could fit anything; a regression on zero
columns cannot, and returning zeros would make it indistinguishable from `naive_persistence` in the
results table — a model silently impersonating the baseline it is being tested against. The runner
must record that as a refusal with its measurement, and not as `model raised`, which it reserves
for defects.

**Files:**
- Create: `cryptoval/scoring/linear.py`, `config/models/ols.yml`, `config/models/ridge.yml`,
  `config/models/elastic_net.yml`
- Modify: `cryptoval/scoring/base.py` (add `Unfittable`), `cryptoval/scoring/runner.py`
  (catch it before the generic handler), `cryptoval/scoring/__init__.py` (restore the `linear`
  import), `requirements.txt`, `tests/test_repo_hygiene.py`
- Test: `tests/test_scoring_models.py`

**Interfaces:**
- Consumes: `ModelSpec`, `Model` (Task 5); `SCORE_COLUMNS` (Task 6).
- Produces: `class Unfittable(Exception)` in `cryptoval.scoring.base`;
  `class SklearnModel` in `cryptoval.scoring.linear` with `__init__(self, spec: ModelSpec,
  estimator)`, used again by Task 8.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring_models.py
import pathlib
import re

import numpy as np
import pytest

import cryptoval.scoring  # noqa: F401
from cryptoval.scoring.base import Unfittable
from cryptoval.scoring.registry import MODELS, load_model_specs

LINEAR = ("ols", "ridge", "elastic_net")


def _learnable(n=400, p=3, seed=20261017):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 0.03 * X[:, 0] - 0.01 * X[:, 1] + rng.normal(scale=0.002, size=n)
    return X, y


def test_the_linear_family_is_registered_and_configured():
    specs = load_model_specs()
    for model_id in LINEAR:
        assert model_id in MODELS
        assert specs[model_id].family == "linear"


@pytest.mark.parametrize("model_id", LINEAR)
def test_a_linear_model_recovers_a_learnable_signal_better_than_predicting_zero(model_id):
    X, y = _learnable()
    model = MODELS[model_id]
    model.fit(X[:300], y[:300], {"horizon": 1, "daily_log_return": np.zeros(300)})
    predicted = model.predict(X[300:])
    assert np.mean(np.abs(y[300:] - predicted)) < np.mean(np.abs(y[300:]))


@pytest.mark.parametrize("model_id", LINEAR)
def test_a_model_with_no_features_refuses_rather_than_impersonating_the_baseline(model_id):
    model = MODELS[model_id]
    with pytest.raises(Unfittable, match="no features"):
        model.fit(np.zeros((100, 0)), np.zeros(100), {"horizon": 1, "daily_log_return": np.zeros(100)})


@pytest.mark.parametrize("model_id", LINEAR)
def test_refitting_the_same_model_twice_gives_the_same_prediction(model_id):
    """Reproducibility is a project invariant: a score must not depend on run order."""
    X, y = _learnable()
    model = MODELS[model_id]
    context = {"horizon": 1, "daily_log_return": np.zeros(300)}
    model.fit(X[:300], y[:300], context)
    first = model.predict(X[300:])
    model.fit(X[:300], y[:300], context)
    assert np.allclose(first, model.predict(X[300:]))


def test_features_are_standardised_so_scale_does_not_decide_the_fit():
    X, y = _learnable()
    scaled = X.copy()
    scaled[:, 0] *= 1e6
    model = MODELS["ridge"]
    context = {"horizon": 1, "daily_log_return": np.zeros(300)}
    model.fit(X[:300], y[:300], context)
    plain = model.predict(X[300:])
    model.fit(scaled[:300], y[:300], context)
    assert np.allclose(plain, model.predict(scaled[300:]), atol=1e-6)


def test_no_shuffling_or_random_split_anywhere_in_the_source_tree():
    """Spec 11.2: no train_test_split, no shuffling, anywhere in the codebase."""
    banned = re.compile(r"train_test_split|KFold|ShuffleSplit|\.sample\(frac|shuffle\s*=\s*True")
    offenders = []
    for path in pathlib.Path("cryptoval").rglob("*.py"):
        if banned.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert not offenders, f"time-ordered validation only (spec 11.2); found in {offenders}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'Unfittable'`.

- [ ] **Step 3: Install the dependency**

```bash
.venv/bin/pip install 'scikit-learn==1.9.1'
printf 'scikit-learn==1.9.1\n' >> requirements.txt
```

Then sort `requirements.txt` to match its existing order if the file is ordered; it is not, so
append is correct.

- [ ] **Step 4: Write the implementation**

Add to `cryptoval/scoring/base.py`:

```python
class Unfittable(Exception):
    """A model cannot be fitted on this fold, with the measurement that says so.

    Distinct from an arbitrary exception on purpose. The runner records this as a
    refusal carrying the reason, and anything else as "model raised", because one is a
    measurement about the data and the other is a defect in the code. A model that
    swallowed this and returned zeros would be scored as -- and indistinguishable from --
    the naive_persistence baseline it is being tested against.
    """
```

In `cryptoval/scoring/runner.py`, catch it before the generic handler. Replace the
`except Exception as exc:` block's opening with:

```python
                try:
                    model.fit(x_train, y_train, context)
                    predicted = np.asarray(model.predict(x_test), dtype="float64")
                except Unfittable as exc:
                    rows.append(
                        _row(
                            model_id=model_id,
                            symbol=str(symbol),
                            horizon=horizon,
                            fold=fold.index,
                            metric="refused",
                            reason=str(exc),
                            train_rows=len(labelled),
                            non_overlapping_windows=report[
                                "non_overlapping_windows"
                            ],
                            panel_vintage_ts=panel_vintage_ts,
                        )
                    )
                    continue
                except Exception as exc:
```

and add `from cryptoval.scoring.base import Model, Unfittable` to its imports.

```python
# cryptoval/scoring/linear.py
"""The linear family (spec 11.3): OLS retained for continuity, plus ridge and elastic net.

ONE WRAPPER FOR EVERY SCIKIT-LEARN ESTIMATOR, reused by the nonlinear family. It does
exactly two things beyond delegating:

* STANDARDISES. The features are log fair-value ratios across 35 methods whose spreads
  differ by orders of magnitude; ridge and elastic net penalise coefficients, so without
  a scaler the penalty would fall on whichever signal happens to be measured small. The
  scaler lives inside the pipeline, which is what makes it fitted on the training fold
  only -- the same rule the runner applies to imputation.
* REFUSES AN EMPTY DESIGN. A fold where no feature cleared the coverage floor cannot
  carry a regression. Returning zeros would make the model arithmetically identical to
  naive_persistence and it would appear in the table as having tied with the baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cryptoval.scoring.base import ModelSpec, Unfittable
from cryptoval.scoring.registry import load_model_specs, register_instance

_SPECS = load_model_specs()


class SklearnModel:
    """Adapts any scikit-learn regressor to the Model protocol."""

    def __init__(self, spec: ModelSpec, estimator) -> None:
        self.spec = spec
        self._estimator = estimator
        self._pipeline = make_pipeline(StandardScaler(), estimator)

    def fit(self, X: np.ndarray, y: np.ndarray, context: dict[str, Any]) -> None:
        if X.ndim != 2 or X.shape[1] == 0:
            raise Unfittable(
                f"{self.spec.model_id}: no features cleared the coverage floor on this fold"
            )
        if len(y) < X.shape[1] + 2:
            raise Unfittable(
                f"{self.spec.model_id}: {len(y)} training rows for {X.shape[1]} features"
            )
        # A fresh pipeline per fit, so a fold cannot inherit the previous fold's scaler.
        from sklearn.base import clone

        self._pipeline = make_pipeline(StandardScaler(), clone(self._estimator))
        self._pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._pipeline.predict(X), dtype="float64")


register_instance(SklearnModel(_SPECS["ols"], LinearRegression()))
register_instance(
    SklearnModel(_SPECS["ridge"], Ridge(**_SPECS["ridge"].params))
)
register_instance(
    SklearnModel(_SPECS["elastic_net"], ElasticNet(**_SPECS["elastic_net"].params))
)
```

```yaml
# config/models/ols.yml
model_id: ols
family: linear
params: {}
citation: "Spec 11.3, the dissertation's linear regression, re-run on returns rather than levels"
```

```yaml
# config/models/ridge.yml
model_id: ridge
family: linear
params:
  alpha: 1.0
citation: "Spec 11.3, ridge regression"
```

```yaml
# config/models/elastic_net.yml
model_id: elastic_net
family: linear
params:
  alpha: 0.001
  l1_ratio: 0.5
  max_iter: 5000
  random_state: 20261017
citation: "Spec 11.3, elastic net"
```

Restore the import in `cryptoval/scoring/__init__.py`:

```python
import cryptoval.scoring.baselines  # noqa: F401 -- import registers the family
import cryptoval.scoring.linear  # noqa: F401
# import cryptoval.scoring.nonlinear   -- restored in Task 8
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scoring_models.py tests/test_scoring_runner.py -v`
Expected: PASS. The runner's tests must still pass — the new `except` clause is ahead of the old
one, not instead of it.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/scoring config/models requirements.txt tests/test_scoring_models.py
git commit -m "feat: scikit-learn, the linear family, and a refusal that cannot impersonate the baseline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: SVR, gradient boosting and the feed-forward net

The dissertation's SVR (RBF and polynomial kernels) and feed-forward NN are retained for continuity
and re-run correctly (§11.3), joined by gradient boosting. All four are the Task 7 wrapper with a
different estimator; the work here is the parameters and the determinism.

Every estimator with a stochastic component gets an explicit `random_state`. This is not a
preference: the repo's whole contract is that re-running a range is a no-op, and a model whose
score changes between runs would put a drifting row in a table other tasks join against.

**Files:**
- Create: `cryptoval/scoring/nonlinear.py`, `config/models/svr_rbf.yml`,
  `config/models/svr_poly.yml`, `config/models/gradient_boosting.yml`,
  `config/models/feedforward_nn.yml`
- Modify: `cryptoval/scoring/__init__.py` (restore the `nonlinear` import)
- Test: `tests/test_scoring_models.py` (append)

**Interfaces:**
- Consumes: `SklearnModel` from `cryptoval.scoring.linear` (Task 7).
- Produces: four more entries in `MODELS`; no new names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring_models.py -- append

NONLINEAR = ("svr_rbf", "svr_poly", "gradient_boosting", "feedforward_nn")


def test_the_nonlinear_family_is_registered_and_configured():
    specs = load_model_specs()
    for model_id in NONLINEAR:
        assert model_id in MODELS
        assert specs[model_id].family == "nonlinear"


def test_the_registry_holds_the_ten_models_this_plan_builds():
    # 3 baselines + 3 linear + 4 nonlinear. A count that drifts without a plan entry is a
    # model someone added without a config.
    assert len(MODELS) == 10


@pytest.mark.parametrize("model_id", NONLINEAR)
def test_a_nonlinear_model_recovers_a_learnable_signal(model_id):
    X, y = _learnable(n=600)
    model = MODELS[model_id]
    model.fit(X[:450], y[:450], {"horizon": 1, "daily_log_return": np.zeros(450)})
    predicted = model.predict(X[450:])
    assert np.isfinite(predicted).all()
    assert len(predicted) == 150


@pytest.mark.parametrize("model_id", NONLINEAR)
def test_a_stochastic_model_is_seeded_so_two_runs_agree(model_id):
    X, y = _learnable(n=600)
    context = {"horizon": 1, "daily_log_return": np.zeros(450)}
    model = MODELS[model_id]
    model.fit(X[:450], y[:450], context)
    first = model.predict(X[450:])
    model.fit(X[:450], y[:450], context)
    assert np.allclose(first, model.predict(X[450:]))


@pytest.mark.parametrize("model_id", NONLINEAR)
def test_a_nonlinear_model_with_no_features_refuses(model_id):
    with pytest.raises(Unfittable):
        MODELS[model_id].fit(
            np.zeros((100, 0)), np.zeros(100), {"horizon": 1, "daily_log_return": np.zeros(100)}
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring_models.py -v`
Expected: FAIL — `KeyError: 'svr_rbf'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/scoring/nonlinear.py
"""The nonlinear family (spec 11.3).

The dissertation's SVR (RBF and polynomial) and feed-forward NN, re-run on returns rather
than levels, plus gradient boosting. Every one of them is Task 7's SklearnModel with a
different estimator, so the standardising pipeline and the empty-design refusal are
inherited rather than restated four times.

EVERY STOCHASTIC ESTIMATOR IS SEEDED. The repo's contract is that re-running is a no-op
and that a stored row can be reproduced; an unseeded MLP would put a row in
fct_model_score that changes on every run, and a DM p-value computed from it would not be
a fact about the model.

THE PARAMETERS ARE NOT TUNED. There is no inner cross-validation loop here, because an
honest one would need a nested walk-forward inside each training fold and this universe
does not have the folds to spare -- see Task 2's measurement. The values below are
scikit-learn's defaults except where the target's scale demands otherwise (SVR's epsilon,
which defaults to 0.1 and would swallow a daily log return of 0.01 entirely, predicting a
flat zero and scoring exactly like the baseline). That limitation is reported, not hidden.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from cryptoval.scoring.linear import SklearnModel
from cryptoval.scoring.registry import load_model_specs, register_instance

_SPECS = load_model_specs()

register_instance(SklearnModel(_SPECS["svr_rbf"], SVR(**_SPECS["svr_rbf"].params)))
register_instance(SklearnModel(_SPECS["svr_poly"], SVR(**_SPECS["svr_poly"].params)))
register_instance(
    SklearnModel(
        _SPECS["gradient_boosting"],
        GradientBoostingRegressor(**_SPECS["gradient_boosting"].params),
    )
)
register_instance(
    SklearnModel(_SPECS["feedforward_nn"], MLPRegressor(**_SPECS["feedforward_nn"].params))
)
```

```yaml
# config/models/svr_rbf.yml
model_id: svr_rbf
family: nonlinear
params:
  kernel: rbf
  C: 1.0
  gamma: scale
  # Default 0.1 is ten times a typical daily log return, so the tube would contain every
  # observation and the fit would be a flat zero -- scoring identically to the baseline.
  epsilon: 0.001
citation: "Spec 11.3, the dissertation's SVR with an RBF kernel, re-run on returns"
```

```yaml
# config/models/svr_poly.yml
model_id: svr_poly
family: nonlinear
params:
  kernel: poly
  degree: 2
  C: 1.0
  gamma: scale
  coef0: 1.0
  epsilon: 0.001
citation: "Spec 11.3, the dissertation's SVR with a polynomial kernel, re-run on returns"
```

```yaml
# config/models/gradient_boosting.yml
model_id: gradient_boosting
family: nonlinear
params:
  n_estimators: 200
  max_depth: 3
  learning_rate: 0.05
  subsample: 0.8
  random_state: 20261017
citation: "Spec 11.3, gradient boosting"
```

```yaml
# config/models/feedforward_nn.yml
model_id: feedforward_nn
family: nonlinear
params:
  hidden_layer_sizes: [64, 32]
  activation: relu
  solver: adam
  alpha: 0.001
  max_iter: 2000
  early_stopping: false
  random_state: 20261017
citation: "Spec 11.3, the dissertation's feed-forward neural network, re-run on returns"
```

Note on `hidden_layer_sizes`: YAML gives a list, scikit-learn wants a tuple but accepts a list in
1.9. If a `TypeError` appears, convert in `nonlinear.py` with
`params | {"hidden_layer_sizes": tuple(params["hidden_layer_sizes"])}` rather than changing the
YAML, so `param_set_hash` stays stable across the change.

`early_stopping: false` is deliberate: scikit-learn's early stopping carves a **random** validation
split out of the training data, which spec §11.2 forbids outright.

Restore the import in `cryptoval/scoring/__init__.py`:

```python
import cryptoval.scoring.baselines  # noqa: F401 -- import registers the family
import cryptoval.scoring.linear  # noqa: F401
import cryptoval.scoring.nonlinear  # noqa: F401
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scoring_models.py -v`
Expected: PASS, with `test_the_registry_holds_the_ten_models_this_plan_builds` at 10.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/scoring config/models tests/test_scoring_models.py
git commit -m "feat: SVR, gradient boosting and the feed-forward net, all seeded

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: The recurrent model — implement it or declare it, from Task 2's measurement

This task has **two outcomes and the measurement decides**, in the same shape Plan 3's Task 6 used
for the 30.2 MB TVL payload. Do not choose by preference. Open Task 2 Step 5's output, which should
be pasted into the task report, and read the line
`assets clearing >=5 folds AND >=30 non-overlapping windows`.

**The threshold, stated in advance:** a horizon qualifies if **8 or more assets** clear both
`folds >= 5` and `non_overlapping_windows >= 30`.

**Outcome A — one or more horizons qualify.** Implement the GRU for those horizons only.
- Add `torch==2.14.1` to `requirements.txt`, and to `.github/workflows/ci.yml` install it from the
  CPU index so ubuntu does not pull ~2GB of CUDA wheels:
  `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`.
  The macOS arm64 wheel measured 121 MB on 2026-10-17; the CPU-only Linux wheel is the comparable
  artefact for CI.
- Create `cryptoval/scoring/recurrent.py` with a `GRUModel(SklearnModel-shaped)` class registering
  as `gru`, a `config/models/gru.yml` carrying `sequence_length`, `hidden_size`, `epochs`,
  `learning_rate` and `random_state`, and a `qualifying_horizons` list taken from the measurement.
- Seed everything: `torch.manual_seed(spec.params["random_state"])`,
  `torch.use_deterministic_algorithms(True)`, and `torch.set_num_threads(1)`. Assert in a test that
  two fits give identical predictions.
- Raise `Unfittable` for any horizon not in `qualifying_horizons`, with the measured window count
  in the message, so the refusal reaches the table rather than being filtered out upstream.
- Update `test_the_registry_holds_the_ten_models_this_plan_builds` to 11.

**Outcome B — no horizon qualifies.** Do not add the dependency. Instead:
- Add the declared-gap row to Task 12's report table, carrying the measured counts verbatim.
- Add a test pinning the reason, so the gap is enforced rather than remembered:

```python
# tests/test_scoring_models.py -- append (Outcome B only)
def test_the_recurrent_model_is_a_declared_gap_with_its_measurement():
    """Spec 11.3 lists an LSTM/GRU; this universe cannot carry one. See Task 2's measurement."""
    assert "gru" not in MODELS
    assert "lstm" not in MODELS
    gap = pathlib.Path("reports/does-anything-beat-a-random-walk.md").read_text(encoding="utf-8")
    assert "non-overlapping" in gap
    assert "recurrent" in gap.lower()
```

**Expected outcome on the 2026-10-14 vintage, for orientation only — follow the measurement, not
this paragraph.** BTC, LTC and DOGE carry 4,305 price rows; ETH 4,086; the next tier (BNB, BCH,
MANA, LINK, ADA, MKR) 3,200-3,400; eleven assets are under 2,400; TRX and SOL have 364. At h=1 and
h=7 most assets clear 30 non-overlapping windows comfortably. At h=365, BTC's 4,305 rows give about
11.8 and no asset in the universe reaches 30, so h=365 is expected to fail for every asset. The
interesting case is h=30 and h=90, where the long-history assets clear and the 2020-vintage DeFi
tokens do not — if that is what the measurement shows, Outcome A applies to those horizons and the
`qualifying_horizons` list is how the model refuses the rest.

- [ ] **Step 1: Read Task 2 Step 5's measurement and record which outcome applies, with the counts**
- [ ] **Step 2: Implement Outcome A or Outcome B as specified above**
- [ ] **Step 3: Run the full suite** — `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`
- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: the recurrent model decided by measured sample size, not preference

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

# Stage 4-4 — Discipline, persistence and the chapter

## Task 10: Counting the hypotheses, and correcting for them

Spec §11.6: *this project evaluates many methods against many assets at many horizons. Some will
look significant by chance. The report therefore records the total number of hypotheses tested and
applies a family-wise correction to headline claims. Without this, the breadth of the method
registry becomes a data-snooping machine.*

The arithmetic is blunt and that is the point. Seven non-baseline models × 22 foldable assets × 5
horizons is up to 770 DM tests. At α=0.05, about 38 of them are expected to look significant with
no skill present whatsoever. A table of "models that beat persistence" without this correction is
a list of coin flips.

Holm-Bonferroni rather than Benjamini-Hochberg: §11.6 asks for a **family-wise** correction, which
controls the probability of *any* false claim, and the headline claim here is existential ("does
*anything* beat a random walk"). FDR would control a different quantity and answer a different
question.

**Files:**
- Create: `cryptoval/scoring/multiplicity.py`
- Test: `tests/test_scoring_multiplicity.py`

**Interfaces:**
- Consumes: `SCORE_COLUMNS` (Task 6).
- Produces: `correct_family_wise(scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame`,
  returning `scores` with `dm_p_value_holm` and `hypotheses_tested` rows appended;
  `ALPHA: float = 0.05`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring_multiplicity.py
import numpy as np
import pandas as pd
import pytest

from cryptoval.scoring.multiplicity import ALPHA, correct_family_wise
from cryptoval.scoring.runner import SCORE_COLUMNS


def _p_rows(p_values, model_ids=None, symbols=None, horizons=None) -> pd.DataFrame:
    n = len(p_values)
    model_ids = model_ids or [f"m{i}" for i in range(n)]
    symbols = symbols or ["BTC"] * n
    horizons = horizons or [1] * n
    rows = []
    for p, model_id, symbol, horizon in zip(p_values, model_ids, symbols, horizons, strict=True):
        row = dict.fromkeys(SCORE_COLUMNS)
        row.update(
            model_id=model_id, symbol=symbol, horizon=horizon, fold=-1,
            metric="dm_p_value", value=p, reason="",
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def test_the_hypothesis_count_is_recorded_as_a_row():
    out = correct_family_wise(_p_rows([0.01, 0.2, 0.5]))
    count = out[out["metric"] == "hypotheses_tested"]
    assert len(count) == 1
    assert count["value"].iloc[0] == 3.0


def test_holm_raises_every_p_value_and_never_lowers_one():
    raw = [0.001, 0.02, 0.3, 0.7]
    out = correct_family_wise(_p_rows(raw))
    adjusted = out[out["metric"] == "dm_p_value_holm"].sort_values("model_id")["value"].to_numpy()
    assert (adjusted >= np.array(raw)).all()


def test_holm_matches_its_definition_on_a_worked_example():
    # Sorted p = 0.01, 0.04; n = 2. Holm: p1 * 2 = 0.02, then max(0.02, 0.04 * 1) = 0.04.
    out = correct_family_wise(_p_rows([0.01, 0.04], model_ids=["a", "b"]))
    adjusted = out[out["metric"] == "dm_p_value_holm"].set_index("model_id")["value"]
    assert adjusted["a"] == pytest.approx(0.02)
    assert adjusted["b"] == pytest.approx(0.04)


def test_a_result_significant_alone_can_stop_being_significant_in_company():
    """The whole point of 11.6: breadth turns a registry into a data-snooping machine."""
    alone = correct_family_wise(_p_rows([0.03]))
    assert alone[alone["metric"] == "dm_p_value_holm"]["value"].iloc[0] < ALPHA

    crowd = correct_family_wise(_p_rows([0.03] + [0.6] * 99))
    survivors = crowd[(crowd["metric"] == "dm_p_value_holm") & (crowd["value"] < ALPHA)]
    assert survivors.empty


def test_a_refused_dm_test_is_not_counted_as_a_hypothesis():
    rows = _p_rows([0.01, 0.02])
    rows.loc[1, "value"] = np.nan
    rows.loc[1, "reason"] = "only 4 paired observations, minimum 10"
    out = correct_family_wise(rows)
    assert out[out["metric"] == "hypotheses_tested"]["value"].iloc[0] == 1.0
    assert len(out[out["metric"] == "dm_p_value_holm"]) == 1


def test_the_original_rows_are_preserved_alongside_the_corrected_ones():
    rows = _p_rows([0.01, 0.2])
    out = correct_family_wise(rows)
    assert len(out[out["metric"] == "dm_p_value"]) == 2
    assert list(out.columns) == SCORE_COLUMNS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring_multiplicity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.scoring.multiplicity'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/scoring/multiplicity.py
"""Multiple-testing discipline (spec 11.6).

"Without this, the breadth of the method registry becomes a data-snooping machine."

Seven non-baseline models against 22 foldable assets at 5 horizons is up to 770
Diebold-Mariano tests. At alpha = 0.05 roughly 38 of them are expected to come back
significant with no skill present at all, so an uncorrected "beat persistence" table is a
list of coin flips with model names attached.

HOLM-BONFERRONI, not Benjamini-Hochberg. Spec 11.6 asks for a FAMILY-WISE correction,
which bounds the probability of making ANY false claim, and the headline question here is
existential: does ANYTHING beat a random walk. FDR bounds the expected share of false
claims among the rejections, which answers a different question and would let a handful
of spurious winners through by design.

The family is this plan's own hypotheses. The count is written to the table as a row so
Plans 4b and 4c can re-correct over the union without re-running anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cryptoval.scoring.runner import SCORE_COLUMNS

ALPHA = 0.05


def correct_family_wise(scores: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Append Holm-adjusted p-values and the hypothesis count to ``scores``.

    A refused DM test (NaN value, reason set) is NOT a hypothesis: counting it would
    inflate the correction and make the surviving claims harder to reject than the
    evidence warrants, which is a different error from the one 11.6 guards against but an
    error all the same.
    """
    tests = scores[(scores["metric"] == "dm_p_value") & scores["value"].notna()].copy()
    n = int(len(tests))

    count_row = dict.fromkeys(SCORE_COLUMNS)
    count_row.update(
        model_id="*",
        symbol="*",
        horizon=-1,
        fold=-1,
        metric="hypotheses_tested",
        value=float(n),
        reason=f"Holm-Bonferroni family-wise correction at alpha={alpha}",
    )

    if n == 0:
        return pd.concat([scores, pd.DataFrame([count_row])], ignore_index=True)[SCORE_COLUMNS]

    _, adjusted, _, _ = multipletests(
        tests["value"].to_numpy(dtype="float64"), alpha=alpha, method="holm"
    )
    corrected = tests.copy()
    corrected["metric"] = "dm_p_value_holm"
    corrected["value"] = np.asarray(adjusted, dtype="float64")
    corrected["reason"] = f"Holm over {n} hypotheses"

    return pd.concat(
        [scores, corrected, pd.DataFrame([count_row])], ignore_index=True
    )[SCORE_COLUMNS]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scoring_multiplicity.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/scoring/multiplicity.py tests/test_scoring_multiplicity.py
git commit -m "feat: Holm-Bonferroni over every hypothesis this plan tests

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 11: The `score` command and the `fct_model_score` mart

Everything exists; nothing is wired. This adds the CLI verb, the gold write, the dbt mart at spec
§8.1's grain, and the one column Task 1 left out — `daily_log_return`, which `random_walk_drift`
needs and which only the panel can supply.

**Files:**
- Create: `dbt/models/marts/fct_model_score.sql`
- Modify: `cryptoval/scoring/targets.py` (emit `daily_log_return`), `cryptoval/cli.py`,
  `dbt/models/marts/_marts.yml`, `Makefile`
- Test: `tests/test_scoring_targets.py` (append), `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `build_frame`, `HORIZONS` (Task 1); `run_walk_forward`, `SCORE_COLUMNS` (Task 6);
  `correct_family_wise` (Task 10); `MODELS` (Task 5); `write_gold` from `cryptoval.io.bronze`.
- Produces: `run_score(settings, assets, model_ids=None, horizons=None) -> dict[str, dict]` in
  `cryptoval.cli`; the `score` subcommand.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring_targets.py -- append

def test_the_frame_carries_the_daily_log_return_the_drift_baseline_needs():
    price = _price_frame(200)
    estimates = _estimates(price, "metcalfe_estimated", np.linspace(0.5, 2.0, 200))
    frame = build_frame(price, estimates, horizon=30)
    assert "daily_log_return" in frame.columns
    # It is BACKWARD looking -- log(P[t]/P[t-1]) -- so it is known at t and leaks nothing.
    expected = np.log(price["price_usd"].iloc[5] / price["price_usd"].iloc[4])
    assert frame["daily_log_return"].iloc[5] == pytest.approx(expected)
    assert np.isnan(frame["daily_log_return"].iloc[0])
```

```python
# tests/test_cli.py -- append
def test_score_writes_the_documented_columns_to_gold(tmp_path, monkeypatch):
    import pandas as pd

    from cryptoval.cli import run_score
    from cryptoval.scoring.runner import SCORE_COLUMNS

    # The command must not require a populated warehouse to be unit-testable: patch the
    # two readers, exactly as the estimate tests do for _read_panel.
    days = 2000
    dates = pd.date_range("2019-01-01", periods=days, freq="D")
    panel = pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": dates,
            "price_usd": 100.0 * np.exp(np.cumsum(np.full(days, 0.0005))),
        }
    )
    estimates = pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": dates,
            "method_id": "metcalfe_estimated",
            "fair_value": panel["price_usd"] * np.linspace(0.5, 2.0, days),
            "applicable": True,
        }
    )
    monkeypatch.setattr("cryptoval.cli._read_score_inputs", lambda s: (panel, estimates))

    settings, assets = _settings(tmp_path)
    report = run_score(settings, assets, horizons=[30])
    written = pd.read_parquet(tmp_path / "gold" / "model_score")
    assert list(written.columns) == SCORE_COLUMNS
    assert (written["metric"] == "hypotheses_tested").any()
    assert report["naive_persistence"]["rows"] > 0
```

(`_settings(tmp_path)` already exists in `tests/test_cli.py`; reuse it rather than writing a new
fixture. If its name differs in the file as it stands, use whatever that file already calls its
settings helper.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring_targets.py tests/test_cli.py -v`
Expected: FAIL — `KeyError: 'daily_log_return'` and `ImportError: cannot import name 'run_score'`.

- [ ] **Step 3: Add the column in `cryptoval/scoring/targets.py`**

Inside `build_frame`, immediately after the `out["y"] = ...` assignment:

```python
    # BACKWARD looking, so it is known at t: random_walk_drift estimates its drift from
    # this and scales it by h. Forward-looking would leak the thing being predicted.
    out["daily_log_return"] = out.groupby("symbol", sort=False)["price_usd"].transform(
        lambda s: np.log(s.where(s > 0.0) / s.where(s > 0.0).shift(1))
    )
```

and drop `price_usd` after it rather than before — move the existing
`out = out.drop(columns=["price_usd"])` below this block.

- [ ] **Step 4: Write the CLI wiring in `cryptoval/cli.py`**

```python
# cryptoval/cli.py -- add near the other imports
import cryptoval.scoring  # noqa: F401 -- import registers every model family
from cryptoval.scoring.multiplicity import correct_family_wise
from cryptoval.scoring.registry import MODELS as SCORING_MODELS
from cryptoval.scoring.runner import SCORE_COLUMNS, run_walk_forward
from cryptoval.scoring.targets import HORIZONS, build_frame
```

```python
# cryptoval/cli.py -- add beside run_estimate
def _read_score_inputs(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The price panel and the estimates the signals are built from.

    Separate from _read_panel because scoring needs the ESTIMATES, which the method layer
    produces and therefore never reads. Patched wholesale in tests.
    """
    import duckdb

    con = duckdb.connect(str(pathlib.Path(settings.data_root) / "cryptoval.duckdb"), read_only=True)
    try:
        panel = con.execute(
            "select symbol, metric_date, price_usd from fct_asset_metrics_daily"
        ).fetchdf()
        estimates = con.execute(
            "select symbol, metric_date, method_id, fair_value, applicable "
            "from fct_valuation_estimate"
        ).fetchdf()
        return panel, estimates
    finally:
        con.close()


def run_score(
    settings: Settings,
    assets: list[Asset],
    model_ids: list[str] | None = None,
    horizons: list[int] | None = None,
) -> dict[str, dict]:
    panel, estimates = _read_score_inputs(settings)
    models = (
        {m: SCORING_MODELS[m] for m in model_ids} if model_ids else dict(SCORING_MODELS)
    )
    vintage = (
        panel["panel_vintage_ts"].max() if "panel_vintage_ts" in panel.columns else None
    )

    frames = []
    for horizon in horizons or list(HORIZONS):
        frame = build_frame(panel, estimates, horizon)
        frames.append(run_walk_forward(frame, models, horizon, panel_vintage_ts=vintage))

    scores = correct_family_wise(
        pd.concat(frames, ignore_index=True)[SCORE_COLUMNS]
        if frames
        else pd.DataFrame(columns=SCORE_COLUMNS)
    )
    write_gold(settings.resolver, "model_score", scores)

    report: dict[str, dict] = {}
    for model_id, group in scores[scores["model_id"] != "*"].groupby("model_id"):
        beaten = group[(group["metric"] == "dm_p_value_holm") & (group["value"] < 0.05)]
        report[str(model_id)] = {
            "rows": int(len(group)),
            "refusals": int((group["metric"] == "refused").sum()),
            "symbols": int(group["symbol"].nunique()),
            "beat_baseline_after_holm": int(len(beaten)),
        }
    return report
```

In `main`, beside the `estimate` parser:

```python
    sc = sub.add_parser("score")
    sc.add_argument("--models", help="comma-separated model ids; default every registered model")
    sc.add_argument("--horizons", help="comma-separated day horizons; default 1,7,30,90,365")
```

and beside the `estimate` branch:

```python
    if args.command == "score":
        ids = [m.strip() for m in args.models.split(",")] if args.models else None
        unknown = [m for m in ids or [] if m not in SCORING_MODELS]
        if unknown:
            print(f"unknown model(s): {unknown}; known: {sorted(SCORING_MODELS)}", file=sys.stderr)
            return 2
        hs = [int(h) for h in args.horizons.split(",")] if args.horizons else None
        for model_id, outcome in sorted(run_score(settings, assets, ids, hs).items()):
            print(model_id, outcome)
        # A model that failed to beat persistence is the expected result (spec 11.3), not
        # a failed run.
        return 0
```

- [ ] **Step 5: Write the mart**

```sql
-- dbt/models/marts/fct_model_score.sql
-- Walk-forward results at spec 8.1's grain (model_id, asset, fold, horizon, metric).
-- The naive baseline is stored as just another model_id, so "did it beat the random
-- walk" is a query rather than a claim.
--
-- fold = -1 marks a value pooled across every fold: the Diebold-Mariano statistic, its
-- p-value, and the Holm-adjusted p-value live there. horizon = -1 with model_id = '*'
-- is the hypothesis count the family-wise correction was taken over (spec 11.6).
--
-- A 'refused' row carries a NULL value and the measurement in reason, the same contract
-- fct_valuation_estimate uses: a refusal with its measurement is a research finding.
select
    model_id,
    symbol,
    horizon,
    fold,
    metric,
    value,
    n_test_rows,
    train_rows,
    -- The honest sample size: daily rows at horizon h overlap h-fold, so 4,305 rows at
    -- h=365 are about 11.8 independent observations, not 4,305. Carried on every row so
    -- a long-horizon score cannot be read as if it were a short-horizon one.
    non_overlapping_windows,
    reason,
    panel_vintage_ts
from read_parquet('{{ var("data_root") }}/gold/model_score/*.parquet')
```

Add to `dbt/models/marts/_marts.yml`:

```yaml
  - name: fct_model_score
    description: >
      Walk-forward scores at grain (model_id, symbol, fold, horizon, metric), spec 8.1 and 11.2.
      Baselines are ordinary rows, so "did it beat the random walk" is a join rather than a claim.
      fold = -1 holds values pooled across folds (the DM statistic, its p-value and the Holm
      adjustment); model_id = '*' with horizon = -1 holds the hypothesis count the family-wise
      correction was taken over. A row whose metric is 'refused' carries a NULL value and the
      measurement that caused it.
    tests:
      - dbt_utils.unique_combination_of_columns:
          arguments:
            combination_of_columns: [model_id, symbol, horizon, fold, metric]
    columns:
      - name: model_id
        tests: [not_null]
      - name: metric
        tests: [not_null]
      - name: value
        description: >
          NULL on a refused row and on a DM test that could not be computed; the reason column
          carries the measurement in both cases.
```

Add to the `Makefile`, beside `estimate`:

```makefile
score:
	$(PY) -m cryptoval.cli score
```

and add `score` to the `.PHONY` line.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`
Expected: whole suite green, lint clean.

- [ ] **Step 7: Commit**

```bash
git add cryptoval dbt Makefile tests
git commit -m "feat: the score command and the fct_model_score mart

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 12: The full run, and the chapter that reports what lost

This is Phase 4's deliverable. The expectation, stated in advance by §11.3, is that most models
will not beat persistence at most horizons — so the chapter is written to report that cleanly
rather than to hunt for a winner. A chapter that finds nothing, stated with its hypothesis count
and its correction, is the honest output of this phase.

**Files:**
- Create: `reports/does-anything-beat-a-random-walk.md`
- Modify: `tests/test_scoring_models.py` (the registry count, if Task 9 chose Outcome A)

- [ ] **Step 1: Run the whole thing**

```bash
.venv/bin/python -m cryptoval.cli score      # expect tens of minutes; run it in the background
make dbt
.venv/bin/python -m pytest -q
.venv/bin/ruff check cryptoval tests
```

- [ ] **Step 2: Measure what to write about, before writing it**

```bash
.venv/bin/python - <<'PY'
import duckdb, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)

print("--- the headline: who beat persistence, before and after Holm ---")
print(con.execute("""
  with p as (select model_id, symbol, horizon,
               max(case when metric='dm_p_value' then value end) raw,
               max(case when metric='dm_p_value_holm' then value end) holm,
               max(case when metric='dm_statistic' then value end) stat
             from fct_model_score where fold = -1 group by 1,2,3)
  select model_id,
         count(*) tests,
         sum(case when raw < 0.05 and stat < 0 then 1 else 0 end) beat_raw,
         sum(case when holm < 0.05 and stat < 0 then 1 else 0 end) beat_holm
  from p group by 1 order by beat_holm desc, beat_raw desc
""").fetchdf().to_string())

print("\n--- hypotheses tested (spec 11.6) ---")
print(con.execute("select value from fct_model_score where metric='hypotheses_tested'").fetchdf().to_string())

print("\n--- MASE by model and horizon: below 1.0 beats forecasting no return ---")
print(con.execute("""
  select model_id, horizon, round(median(value),4) median_mase, count(*) folds
  from fct_model_score where metric='mase' and value is not null and fold >= 0
  group by 1,2 order by 2,3
""").fetchdf().to_string())

print("\n--- directional accuracy, pooled per model ---")
print(con.execute("""
  select model_id, round(median(value),4) median_direction, count(*) folds
  from fct_model_score where metric='directional_accuracy' and value is not null
  group by 1 order by 2 desc
""").fetchdf().to_string())

print("\n--- the honest sample size per horizon ---")
print(con.execute("""
  select horizon, count(distinct symbol) symbols,
         round(min(non_overlapping_windows),1) min_windows,
         round(max(non_overlapping_windows),1) max_windows
  from fct_model_score where fold >= 0 group by 1 order by 1
""").fetchdf().to_string())

print("\n--- refusals, by reason ---")
print(con.execute("""
  select reason, count(*) n, count(distinct symbol) symbols
  from fct_model_score where metric='refused' group by 1 order by n desc limit 15
""").fetchdf().to_string())
PY
```

- [ ] **Step 3: Write `reports/does-anything-beat-a-random-walk.md`**

Six sections. Every number comes from Step 2's queries or from a task report, and every claim that
rests on a parameter says which parameter. Carry the §12 disclaimer at the top, as the other two
reports do.

1. **What Phase 4 tests, and what it cannot.** Ten (or eleven) models, 22 foldable assets, five
   horizons, scored out-of-sample on a purged expanding window. State the two assets that cannot be
   folded at all (TRX and SOL, 364 price rows each) with their counts. State the scope boundary:
   §11.4's econometrics and §11.5's factor models are Plans 4b and 4c, so "does a valuation signal
   *earn returns*" is **not** answered here — this chapter answers "does a valuation signal
   *forecast returns*", which is a weaker and different question.
2. **The protocol, and the three leaks it closes.** The one-day feature lag and why
   (`crypto_qtm_panel_velocity` is the price; Mayer and Puell use inclusive windows). The purge of
   *h* rows plus a 10-day embargo, with the arithmetic for h=365. Per-fold imputation and per-fold
   feature selection. Name the features that were dropped for zero variance, from
   `frame.attrs["dropped_features"]`.
3. **The baselines, and the sample-size caveat.** The three baselines' own scores, and the
   non-overlapping window counts per horizon from Step 2's last-but-one query. State plainly that a
   h=365 result rests on roughly 11 independent annual observations for the longest-history asset
   and fewer for everything else, so it is reported but should not be read as comparable to h=1.
4. **What beat persistence, before and after the correction.** The headline table from Step 2's
   first query: tests, `beat_raw`, `beat_holm` per model. Then the §11.6 sentence with the real
   number: *N hypotheses were tested; at α=0.05 roughly N/20 are expected to look significant with
   no skill present; after Holm-Bonferroni, M survive.* If M is zero, say so in one sentence and do
   not soften it — §11.3 predicted it in advance and a null result stated cleanly is this phase's
   deliverable.
5. **Where the models differ from each other, if anywhere.** Median MASE by model and horizon, and
   median directional accuracy. Note any model whose MASE is systematically above 1.0, which means
   it lost to forecasting no return at all, and say which horizons that happened at.
6. **Declared gaps.** One row per gap, each with its measurement:

   | Spec | What is not here | Why |
   |---|---|---|
   | §11.3 | The recurrent model | **Task 9's outcome, with its measured fold and non-overlapping-window counts.** If Outcome A ran, this row instead names the horizons it was *refused* for and their counts. |
   | §11.2 | Tuned hyperparameters | No inner cross-validation. An honest tuning loop needs a nested walk-forward inside each training fold, and Task 2's measurement shows the folds are not there to spare. Every model runs at the stated config values, so these are scores for *a* parameterisation, not for the best one. |
   | §11.4 | ADF/KPSS, cointegration, VECM, GARCH, Markov-switching, DCC | Plan 4b. Needs `arch` plus a hand-rolled DCC estimator; answers a different question and writes a different table. |
   | §11.5 | Factor models, Fama-MacBeth, decile portfolios | Plan 4c, which must first resolve the point-in-time top-N universe §5 and §11.5 both require and no free source serves. |
   | §11.2 | TRX and SOL | 364 price rows each (CoinGecko only, Coin Metrics serves neither on the free tier). Below the 730-row floor for a five-fold purged split, refused with that count on every horizon. |

- [ ] **Step 4: Verify the chapter against the table it describes**

Re-run two of Step 2's queries and check that every number quoted in section 4 matches. A number
in the prose that does not match the mart is the one defect this task can ship.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`
Expected: whole suite green, lint clean.

- [ ] **Step 6: Commit**

```bash
git add reports/does-anything-beat-a-random-walk.md tests
git commit -m "feat: the walk-forward run and the chapter that reports what lost

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Done criteria for Plan 4

- `make lint && make test` green; roughly 540 unit tests, no network.
- Ten models registered (eleven if Task 9 chose Outcome A), each with a config in `config/models/`
  and a row in `fct_model_score`; `tests/test_scoring_models.py` enforces the count.
- **Phase 4's exit criterion, met in full:** every non-baseline model carries a Diebold-Mariano
  statistic and p-value against `naive_persistence` at `fold = -1`, and `fct_model_score` is
  populated including the three baselines as ordinary rows.
- The target is a forward log return on every row; no model is fitted against a price level, and
  `test_no_shuffling_or_random_split_anywhere_in_the_source_tree` passes over `cryptoval/`.
- Every fold satisfies `train.stop + horizon <= test.start` — the purge is a property the tests
  assert directly, not a comment.
- Every score row carries `non_overlapping_windows`, so a h=365 number cannot be read as
  though it were a h=1 number.
- `hypotheses_tested` appears exactly once in the mart, and every `dm_p_value` has a matching
  `dm_p_value_holm`.
- TRX and SOL produce `refused` rows naming their 364-row count rather than scores.
- MASE is NULL with a reason wherever the naive denominator was zero; directional accuracy is NULL
  with its usable-row count wherever a test block had no directional variation.
- `reports/does-anything-beat-a-random-walk.md` states the hypothesis count, the number surviving
  Holm, and the declared-gap table with a measurement in every row.
- `make dbt` builds `fct_model_score` with its uniqueness test passing.

**Plan 5 starts from:** `fct_model_score` with baselines in it, the `Model` protocol and registry
(which a dashboard can enumerate), the purged splitter (which §11.4's rolling econometrics reuse),
and the declared-gap record so the report does not claim a model that was never built.

---

## Self-review

Run against the spec with fresh eyes after writing the plan, per the writing-plans skill.

**1. Spec coverage.** §11.1 → Task 5 (three distinct baselines, registered through the same
protocol). §11.2 → Tasks 1-4 and 6: target as log returns at the five fixed horizons (Task 1),
expanding-window walk-forward with purging and embargo (Task 2), the four metrics (Task 3),
Diebold-Mariano (Task 4), feature hygiene via the universal one-day lag (Task 1) and per-fold
imputation (Task 6). §11.3 → Tasks 7-9: OLS, SVR with both kernels and the feed-forward NN
retained for continuity, plus ridge, elastic net and gradient boosting; the recurrent model decided
by Task 2's measurement. §11.6 → Task 10. §8.1's `fct_model_score` grain → Task 11.
**Two gaps found and both are deliberate, declared in the scope boundary with the plan that owns
them:** §11.4 → Plan 4b, §11.5 → Plan 4c. §11.5's own precondition — the point-in-time top-N
universe that §5 also requires and no free source serves — is recorded as the first thing Plan 4c
must resolve, rather than being inherited silently.

**2. Placeholder scan.** No "TBD", no "add error handling", no "similar to Task N", no test step
without test code. Two places defer a *decision* to a measurement rather than deferring work: Task
2 Step 5 (the fold and non-overlapping-window counts) and Task 9 (which consumes them). Each names
the measurement, the threshold stated in advance, where the number is recorded, and what each
outcome implies — that is a measurement step, not a placeholder.

**3. Type consistency.** `ModelSpec` and the `Model` protocol are defined once in Task 5 and used
unchanged thereafter. `SklearnModel` is defined in Task 7 and reused by Task 8 rather than
reimplemented. `SCORE_COLUMNS` is defined in Task 6 and is the column list in Tasks 10, 11 and the
mart. `BASELINE_MODEL_ID` is defined in Task 5 and used by Tasks 6, 10 and 12, so no module decides
for itself what "the baseline" means. Two corrections made during this review: **(a)** the File
Structure block above lists `run_walk_forward` under `registry.py`, but it lives in
`cryptoval/scoring/runner.py` — the file list is right, the annotation was not; take `runner.py` as
authoritative. **(b)** `Unfittable` is introduced in Task 7 rather than Task 5, because Task 7 is
the first task whose models can fail to fit; Task 7 therefore modifies `base.py` and `runner.py`
and says so in its Files block.

**4. Review Focus.** All five have a test in the task that owns the code: a feature that is the
target → Task 1 (`test_every_feature_is_lagged_at_least_one_day`,
`test_a_zero_variance_signal_is_dropped_with_its_measured_variance`); too little history → Task 2
(`test_an_asset_with_too_few_rows_is_refused_with_its_measured_row_count`); h=365 on 11 independent
observations → Task 2 (`test_the_fold_report_states_non_overlapping_sample_size_per_horizon`) and
Task 6 (`test_every_row_carries_the_non_overlapping_sample_size`); a zero MASE denominator → Task 3
(`test_mase_refuses_a_zero_naive_denominator`); a test block with no directional variation → Task 3
(`test_directional_accuracy_reports_its_usable_count_and_refuses_an_all_zero_block`).

**One risk the executor must carry.** Every leak this plan closes is invisible when it reopens: a
leaked feature, a whole-series median or an unpurged boundary all make the metrics *better*, so
nothing fails and the result looks like a discovery. There is no downstream check that would catch
it — `fct_model_score` has no identity to violate and no vendor to reconcile against. The defences
are the four tests named in Review Focus 1 and Task 6's
`test_imputation_is_fitted_on_the_training_fold_only`. **If a model comes back beating persistence
after Holm, treat it as a suspected leak first and a finding second**, and re-read those five tests
before writing it up: §11.3 predicts in advance that it should not happen, so it is evidence about
the code before it is evidence about the market.
