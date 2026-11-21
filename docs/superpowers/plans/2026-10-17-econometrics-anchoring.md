# The Anchoring Chain Implementation Plan (Plan 4b of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether market cap holds a long-run equilibrium with a fundamental — realized cap, traded volume, or active addresses — with the cointegrating coefficient estimated rather than assumed, and publish the result as a chapter.

**Architecture:** A new `cryptoval/econometrics/` package mirroring the layering of `cryptoval/scoring/`: a levels-panel builder that knows nothing about tests, three statistic modules that each refuse their degenerate case, a config-driven pair registry, a runner that assembles one long table, and a CLI command that writes gold for a dbt mart. Full-sample and descriptive: no folds, no purge, no embargo, because there is no out-of-sample claim to protect.

**Tech Stack:** Python 3.14, pandas, `statsmodels==0.15.0`'s `adfuller`/`kpss`/`coint`/`coint_johansen`/`VECM`, DuckDB, dbt, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-17-econometrics-anchoring-design.md` (which records the four decisions `docs/superpowers/specs/2026-10-11-cryptoasset-valuation-pipeline-design.md` §11.4 does not contain). Executors read both.

## Global Constraints

- **No new dependency.** `statsmodels==0.15.0` is already pinned in `requirements.txt` and carries every estimator this plan needs. `arch` belongs to the later GARCH plan; do not add it.
- **Every statistic refuses its degenerate case** rather than returning a meaningless number, and the refusal carries the measurement that caused it in `reason`. A refusal with its measurement is a research finding (the contract `fct_valuation_estimate` and `fct_model_score` already use).
- **Full sample, and labelled.** One ADF/KPSS, one Engle-Granger/Johansen, one VECM per pair over the whole available history. The chapter must state that this uses future data by construction, supports no tradability claim, and is **not comparable** to Plan 4's out-of-sample null.
- **The universe is measured at run time, never hard-coded.** 20 assets carried 1,000+ rows on the 2026-10-14 vintage; a re-ingest moves that.
- **The Holm family is the cointegration tests only.** ADF/KPSS are pre-tests that decide which regressions are valid; VECM coefficients are estimates of an already-established relationship. Both are reported, neither is corrected, and the chapter says so.
- **One `TEST_COLUMNS` definition, asserted against the mart's SQL select list.** The Plan 4 review found nothing tied `SCORE_COLUMNS` to `fct_model_score.sql`; do not repeat that.
- **Every row carries `panel_vintage_ts`** and a `params_hash` of the pair's parameters.
- **A test that cannot fail is not coverage.** Every task ends with a mutation proof: change the implementation, watch the new test fail, restore. The Plan 4 review found five vacuous tests this way.
- **Before every commit:** `.venv/bin/python -m pytest -q` green and `.venv/bin/ruff check cryptoval tests` clean.

## Review Focus

Five input classes the spec implies and no task's happy path exercises, most likely to bite first. Each line's test is added to the task that owns the code.

1. **A fundamental that is a true zero or a constant.** `active_addresses = 0` on a dead day and `volume_total_usd = 0` on an illiquid one both make `log` infinite; a constant series makes ADF's regression singular. Expected: the value becomes NaN before the log, and a series with no variance is refused with its measured variance. → Tasks 1 and 3.
2. **A fundamental with interior gaps.** DOT carries 653 rows of MVRV inside 2,605 panel rows, so the two series' overlap is not a contiguous daily grid. Expected: refused with the count of missing days, never silently stitched — a stitched series makes a lag mean something different on either side of the hole. → Task 1.
3. **Two series with different coverage.** `realized_cap` starts later than `price_usd` for most assets. Expected: the test runs on the dates where both are finite, and reports that intersection's length as `n_rows` rather than the panel's. → Task 1.
4. **A perfectly collinear pair.** The design exists to avoid testing price against a price-derived series, so the guard that catches one must itself be tested: a pair whose independent variable is an affine function of the dependent leaves a residual with no variance. Expected: refused with the measured residual variance, not a p-value of 0 reported as a discovery. → Task 4.
5. **A corrected table handed back for correction.** `correct_family_wise` already refuses this for Plan 4's metric names; the generalization must not open a hole. Expected: raises, naming how many already-corrected rows it found, for the econometrics metric names too. → Task 7.

## File Structure

| File | Responsibility |
|---|---|
| `cryptoval/econometrics/__init__.py` | Package docstring; imports the registry so pair specs load on import. |
| `cryptoval/econometrics/base.py` | `PairSpec` dataclass and the `Unusable` exception. The contract, with no estimator in it. |
| `cryptoval/econometrics/series.py` | `build_levels` (the log levels panel) and `pair_rows` (the intersection two series are tested on). Knows nothing about tests. |
| `cryptoval/econometrics/registry.py` | `PAIRS`, `load_pair_specs`. One YAML per pair, same shape as `scoring/registry.py`. |
| `cryptoval/econometrics/stationarity.py` | `adf`, `kpss`. Each refuses too-few rows and zero variance. |
| `cryptoval/econometrics/cointegration.py` | `engle_granger` (statistic, p-value, estimated coefficient and its standard error), `johansen` (trace statistic, 95% critical value, rank). |
| `cryptoval/econometrics/vecm.py` | `speed_of_adjustment` — §11.4's "how fast does reversion happen" column. |
| `cryptoval/econometrics/runner.py` | `TEST_COLUMNS`, `run_pairs`. Assembles the long table and owns the refusal rows. |
| `config/econometrics/*.yml` | Three pairs: `mvrv_anchor`, `liquidity_anchor`, `metcalfe_anchor`. |
| `cryptoval/scoring/multiplicity.py` | Modified: `correct_family_wise` gains keyword-only `p_metric`, `columns`, `count_keys`, all defaulting to Plan 4's behaviour. |
| `cryptoval/cli.py` | Modified: the `econ` subcommand, `_read_econ_inputs`, `run_econ`. |
| `dbt/models/marts/fct_econometric_test.sql` | The mart, at grain (pair_id, symbol, metric). |
| `reports/does-price-anchor-to-fundamentals.md` | The chapter. |

**The ratio needs no new config primitive.** A log ratio is the dependent minus a multiple of the independent: `log(market_cap / realized_cap) = log_market_cap − 1 × log_realized_cap`, and `log(market_cap / active_addresses²) = log_market_cap − 2 × log_addresses`. So `forced_coefficient` generates the constrained series, and no expression parser is needed.

---

# Stage 4b-1 — The series, which knows nothing about tests

## Task 1: The log levels panel and the intersection a pair is tested on

**Files:**
- Create: `cryptoval/econometrics/__init__.py`, `cryptoval/econometrics/base.py`, `cryptoval/econometrics/series.py`
- Test: `tests/test_econ_series.py`

**Interfaces:**
- Consumes: `cryptoval.scoring.targets._require_contiguous_days` (reused, not copied).
- Produces: `PairSpec(pair_id, dependent, independent, forced_coefficient, expected_coefficient, citation, params)`; `Unusable(Exception)`; `LEVEL_SERIES: dict[str, str]`; `MIN_PAIR_ROWS: int`; `build_levels(panel: pd.DataFrame) -> pd.DataFrame`; `pair_rows(levels: pd.DataFrame, symbol: str, spec: PairSpec) -> pd.DataFrame` returning columns `metric_date`, `y`, `x`, `ratio`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_econ_series.py
import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import PairSpec, Unusable
from cryptoval.econometrics.series import LEVEL_SERIES, MIN_PAIR_ROWS, build_levels, pair_rows

MVRV = PairSpec(
    pair_id="mvrv_anchor",
    dependent="log_market_cap",
    independent="log_realized_cap",
    forced_coefficient=1.0,
)


def _panel(days=1200, symbols=("BTC",), **overrides):
    out = []
    for symbol in symbols:
        dates = pd.date_range("2019-01-01", periods=days, freq="D")
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "metric_date": dates,
                "price_usd": np.linspace(100.0, 300.0, days),
                "market_cap_usd": np.linspace(1e9, 3e9, days),
                "realized_cap": np.linspace(5e8, 2e9, days),
                "volume_total_usd": np.linspace(1e7, 5e7, days),
                "active_addresses": np.linspace(1e4, 5e4, days),
                "panel_vintage_ts": pd.Timestamp("2026-10-12 08:30", tz="UTC"),
            }
        )
        for column, value in overrides.items():
            frame[column] = value
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def test_build_levels_logs_every_declared_series():
    levels = build_levels(_panel())
    assert set(LEVEL_SERIES) <= set(levels.columns)
    panel = _panel()
    assert np.allclose(levels["log_market_cap"], np.log(panel["market_cap_usd"]))


def test_a_non_positive_value_becomes_nan_rather_than_minus_infinity():
    """Review Focus 1: active_addresses = 0 on a dead day, volume = 0 on an illiquid one."""
    panel = _panel()
    panel.loc[panel.index[5], "active_addresses"] = 0.0
    panel.loc[panel.index[6], "volume_total_usd"] = -1.0
    levels = build_levels(panel)
    assert levels["log_addresses"].isna().iloc[5]
    assert levels["log_volume"].isna().iloc[6]
    # The point of the guard: NaN is a missing measurement, -inf is a number a regression
    # will happily consume.
    assert not np.isneginf(levels[list(LEVEL_SERIES)].to_numpy()).any()


def test_pair_rows_uses_the_intersection_and_reports_its_length():
    """Review Focus 3: realized_cap starts later than price for most assets."""
    panel = _panel(days=1200)
    panel.loc[panel.index[:300], "realized_cap"] = np.nan
    rows = pair_rows(build_levels(panel), "BTC", MVRV)
    assert len(rows) == 900
    assert rows["metric_date"].min() == panel["metric_date"].iloc[300]
    assert rows[["y", "x"]].notna().all().all()


def test_the_ratio_is_the_dependent_minus_the_forced_multiple_of_the_independent():
    rows = pair_rows(build_levels(_panel()), "BTC", MVRV)
    assert np.allclose(rows["ratio"], rows["y"] - 1.0 * rows["x"])

    metcalfe = PairSpec(
        pair_id="metcalfe_anchor",
        dependent="log_market_cap",
        independent="log_addresses",
        forced_coefficient=2.0,
    )
    squared = pair_rows(build_levels(_panel()), "BTC", metcalfe)
    assert np.allclose(squared["ratio"], squared["y"] - 2.0 * squared["x"])


def test_an_interior_gap_in_the_intersection_is_refused_with_its_count():
    """Review Focus 2: DOT carries 653 MVRV rows inside 2,605 panel rows."""
    panel = _panel(days=1200)
    panel.loc[panel.index[600:610], "realized_cap"] = np.nan
    with pytest.raises(Unusable) as exc:
        pair_rows(build_levels(panel), "BTC", MVRV)
    assert "10 missing day(s)" in str(exc.value)


def test_a_pair_below_the_row_floor_is_refused_with_the_count():
    panel = _panel(days=MIN_PAIR_ROWS - 1)
    with pytest.raises(Unusable) as exc:
        pair_rows(build_levels(panel), "BTC", MVRV)
    assert str(MIN_PAIR_ROWS - 1) in str(exc.value)
    assert str(MIN_PAIR_ROWS) in str(exc.value)


def test_a_symbol_absent_from_the_panel_is_refused_not_empty():
    with pytest.raises(Unusable):
        pair_rows(build_levels(_panel()), "ETH", MVRV)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_econ_series.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.econometrics'`

- [ ] **Step 3: Write the contract**

```python
# cryptoval/econometrics/base.py
"""The pair contract (spec 11.4), with no estimator in it.

A PAIR is the unit of this plan: a dependent series, an independent series, and the
coefficient the practitioner ratio forces on their relationship. Engle-Granger ESTIMATES
that coefficient; the ratio test assumes it. Holding both in one object is what lets the
chapter report them as the unconstrained and constrained versions of one hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Unusable(Exception):
    """The pair cannot be tested on this asset, and the message carries the measurement.

    Deliberately the same shape as scoring.base.Unfittable: the runner turns it into a
    refusal ROW rather than letting it end the run, so a refusal is published with its
    cause instead of disappearing.
    """


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    dependent: str
    independent: str
    # The coefficient the practitioner RATIO forces: 1 for market cap over realized cap,
    # 2 for Metcalfe, whose law states value is proportional to the square of the users.
    forced_coefficient: float
    # The value the law predicts, where a law predicts one. Metcalfe's 2 is testable
    # against the estimated coefficient's interval; MVRV predicts no particular number.
    expected_coefficient: float | None = None
    citation: str = ""
    params: dict[str, Any] = field(default_factory=dict)
```

```python
# cryptoval/econometrics/__init__.py
"""The anchoring chain (spec 11.4): ADF/KPSS, Engle-Granger/Johansen, VECM.

Imported for its side effect of loading the pair registry, the same way
cryptoval.scoring imports its model families.
"""

from cryptoval.econometrics import registry  # noqa: F401
```

- [ ] **Step 4: Write the series builder**

```python
# cryptoval/econometrics/series.py
"""The log levels panel, and the dates a pair is actually testable on.

LEVELS, not returns and not ratios. Cointegration is a statement about two
non-stationary LEVEL series sharing a stochastic trend; differencing them away is
exactly what the test is built to avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.econometrics.base import PairSpec, Unusable
from cryptoval.scoring.targets import _require_contiguous_days

# Log series name -> the panel column it is the log of. Non-positive values become NaN:
# an asset with no transactions that day has no log of zero, and -inf in a regression is
# not a measurement.
LEVEL_SERIES: dict[str, str] = {
    "log_market_cap": "market_cap_usd",
    "log_realized_cap": "realized_cap",
    "log_volume": "volume_total_usd",
    "log_addresses": "active_addresses",
}

# Two years. Below this a cointegration test on daily data is reporting a trend, not an
# equilibrium, and the refusal names the count so a reader sees which assets fell out.
# Measured on the 2026-10-14 vintage: 20 of 24 assets clear it on realized_cap; DOT (653)
# and BNB (647) do not, TRX has no realized_cap at all and SOL has 363 rows.
MIN_PAIR_ROWS = 730


def build_levels(panel: pd.DataFrame) -> pd.DataFrame:
    """(symbol, metric_date, <log series per LEVEL_SERIES>, panel_vintage_ts)."""
    keep = ["symbol", "metric_date"]
    if "panel_vintage_ts" in panel.columns:
        keep.append("panel_vintage_ts")
    out = panel[keep].copy()
    for name, column in LEVEL_SERIES.items():
        values = pd.to_numeric(panel[column], errors="coerce").astype("float64")
        out[name] = np.log(values.where(values > 0.0))
    return out.sort_values(["symbol", "metric_date"]).reset_index(drop=True)


def pair_rows(levels: pd.DataFrame, symbol: str, spec: PairSpec) -> pd.DataFrame:
    """The dates where BOTH series are finite, with y, x and the constrained ratio.

    Raises Unusable, with the measurement, when the symbol is absent, when the
    intersection is shorter than MIN_PAIR_ROWS, or when it has an interior gap. A gap is
    refused rather than stitched: a stitched series makes a one-day lag mean one thing
    before the hole and another after it, and nothing downstream could detect that.
    """
    group = levels[levels["symbol"] == symbol]
    if group.empty:
        raise Unusable(f"{spec.pair_id}: {symbol} is not in the panel")

    rows = group[["metric_date", spec.dependent, spec.independent]].rename(
        columns={spec.dependent: "y", spec.independent: "x"}
    )
    rows = rows[rows["y"].notna() & rows["x"].notna()].reset_index(drop=True)

    if len(rows) < MIN_PAIR_ROWS:
        raise Unusable(
            f"{spec.pair_id}: {symbol} has {len(rows)} rows where both "
            f"{spec.dependent} and {spec.independent} are finite; "
            f"{MIN_PAIR_ROWS} are needed for a cointegration test on daily data"
        )

    # The intersection, not the panel, is what the test reads, so contiguity is checked
    # here rather than inherited from the panel's own guard.
    _require_contiguous_days(rows.assign(symbol=symbol))

    rows["ratio"] = rows["y"] - spec.forced_coefficient * rows["x"]
    return rows
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_econ_series.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Mutation proof**

Change `rows["y"].notna() & rows["x"].notna()` to `rows["y"].notna()` and run the file. `test_pair_rows_uses_the_intersection_and_reports_its_length` must FAIL (it would return 1,200 rows, not 900). Restore with `git checkout cryptoval/econometrics/series.py` and confirm `git diff` is empty.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/econometrics/ tests/test_econ_series.py
git commit -m "feat: the log levels panel, and the intersection a pair is testable on"
```

## Task 2: The pair registry and its three configs

**Files:**
- Create: `cryptoval/econometrics/registry.py`, `config/econometrics/mvrv_anchor.yml`, `config/econometrics/liquidity_anchor.yml`, `config/econometrics/metcalfe_anchor.yml`
- Test: `tests/test_econ_registry.py`

**Interfaces:**
- Consumes: `PairSpec` from Task 1.
- Produces: `PAIRS: dict[str, PairSpec]`; `load_pair_specs(config_dir: str = "config/econometrics") -> dict[str, PairSpec]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_econ_registry.py
import pytest

import cryptoval.econometrics  # noqa: F401 -- loads the registry
from cryptoval.econometrics.registry import PAIRS, load_pair_specs
from cryptoval.econometrics.series import LEVEL_SERIES


def test_the_three_designed_pairs_are_registered():
    assert set(PAIRS) == {"mvrv_anchor", "liquidity_anchor", "metcalfe_anchor"}


def test_every_pair_names_series_the_levels_panel_actually_builds():
    for pair_id, spec in PAIRS.items():
        assert spec.dependent in LEVEL_SERIES, pair_id
        assert spec.independent in LEVEL_SERIES, pair_id


def test_metcalfe_forces_two_and_expects_two():
    """The law states value is proportional to the SQUARE of the users, so the
    coefficient-1 ratio is not Metcalfe's law and must not be labelled as it."""
    spec = PAIRS["metcalfe_anchor"]
    assert spec.forced_coefficient == 2.0
    assert spec.expected_coefficient == 2.0


def test_the_ratio_pairs_force_one_and_predict_no_particular_coefficient():
    for pair_id in ("mvrv_anchor", "liquidity_anchor"):
        assert PAIRS[pair_id].forced_coefficient == 1.0
        assert PAIRS[pair_id].expected_coefficient is None


def test_every_pair_carries_a_citation():
    for pair_id, spec in PAIRS.items():
        assert spec.citation.strip(), pair_id


def test_a_duplicate_pair_id_is_refused(tmp_path):
    for name in ("a.yml", "b.yml"):
        (tmp_path / name).write_text(
            "pair_id: same\ndependent: log_market_cap\n"
            "independent: log_volume\nforced_coefficient: 1.0\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError) as exc:
        load_pair_specs(str(tmp_path))
    assert "same" in str(exc.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_econ_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.econometrics.registry'`

- [ ] **Step 3: Write the registry**

```python
# cryptoval/econometrics/registry.py
"""Pair discovery (spec 11.4).

The same shape as cryptoval/scoring/registry.py and cryptoval/methods/registry.py: a
dict keyed by id, specs loaded from one YAML per pair. A pair is DATA, so adding the
next fundamental is a file, not a branch.
"""

from __future__ import annotations

import pathlib

import yaml

from cryptoval.econometrics.base import PairSpec

PAIRS: dict[str, PairSpec] = {}


def load_pair_specs(config_dir: str = "config/econometrics") -> dict[str, PairSpec]:
    specs: dict[str, PairSpec] = {}
    for path in sorted(pathlib.Path(config_dir).glob("*.yml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        expected = raw.get("expected_coefficient")
        spec = PairSpec(
            pair_id=raw["pair_id"],
            dependent=raw["dependent"],
            independent=raw["independent"],
            forced_coefficient=float(raw["forced_coefficient"]),
            expected_coefficient=None if expected is None else float(expected),
            citation=raw.get("citation", ""),
            params=dict(raw.get("params") or {}),
        )
        if spec.pair_id in specs:
            raise ValueError(f"duplicate pair_id {spec.pair_id} in {path}")
        specs[spec.pair_id] = spec
    return specs


PAIRS.update(load_pair_specs())
```

- [ ] **Step 4: Write the three configs**

```yaml
# config/econometrics/mvrv_anchor.yml
pair_id: mvrv_anchor
dependent: log_market_cap
independent: log_realized_cap
forced_coefficient: 1.0
expected_coefficient: null
params: {lag_criterion: aic, alpha: 0.05}
citation: "Spec 11.4. MVRV is market cap over realized cap, so its stationarity is this pair with the coefficient forced to 1."
```

```yaml
# config/econometrics/liquidity_anchor.yml
pair_id: liquidity_anchor
dependent: log_market_cap
independent: log_volume
forced_coefficient: 1.0
expected_coefficient: null
params: {lag_criterion: aic, alpha: 0.05}
# NOT NVT. volume_total_usd is EXCHANGE TRADING VOLUME, not on-chain transfer value,
# which the nvt_golden_ratio method card already declares as an objection. This pair
# tests market cap against market liquidity, and the chapter must say so.
citation: "Spec 11.4's NVT question, in the form this panel's data supports: market cap against traded volume."
```

```yaml
# config/econometrics/metcalfe_anchor.yml
pair_id: metcalfe_anchor
dependent: log_market_cap
independent: log_addresses
forced_coefficient: 2.0
expected_coefficient: 2.0
params: {lag_criterion: aic, alpha: 0.05}
citation: "Metcalfe (1995) via spec 11.4: value proportional to the square of the users, so the law predicts a coefficient of 2 on log addresses."
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_econ_registry.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Mutation proof**

Set `forced_coefficient: 1.0` in `metcalfe_anchor.yml` and run the file. `test_metcalfe_forces_two_and_expects_two` must FAIL. Restore the file.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/econometrics/registry.py config/econometrics/ tests/test_econ_registry.py
git commit -m "feat: the pair registry, and the three pairs the design settles on"
```

---

# Stage 4b-2 — The statistics, each refusing its degenerate case

## Task 3: ADF and KPSS

**Files:**
- Create: `cryptoval/econometrics/stationarity.py`
- Test: `tests/test_econ_stationarity.py`

**Interfaces:**
- Consumes: `Unusable` from Task 1.
- Produces: `MIN_STAT_ROWS: int`; `adf(series: pd.Series, lag_criterion: str = "aic") -> dict[str, float]` with keys `statistic`, `p_value`, `lags`, `n`; `kpss(series: pd.Series) -> dict[str, float]` with keys `statistic`, `p_value`, `lags`, `n`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_econ_stationarity.py
import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.econometrics.stationarity import MIN_STAT_ROWS, adf, kpss


def _random_walk(n=800, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0.0, 1.0, n)))


def _stationary(n=800, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 1.0, n))


def test_adf_fails_to_reject_a_unit_root_on_a_random_walk():
    """Known answer, one direction. A random walk HAS a unit root."""
    assert adf(_random_walk())["p_value"] > 0.10


def test_adf_rejects_a_unit_root_on_white_noise():
    """Known answer, the other direction -- so the test can fail."""
    assert adf(_stationary())["p_value"] < 0.01


def test_kpss_rejects_stationarity_on_a_random_walk_and_not_on_noise():
    assert kpss(_random_walk())["p_value"] < 0.05
    assert kpss(_stationary())["p_value"] > 0.05


def test_both_report_the_sample_size_they_used():
    series = _random_walk(n=500)
    assert adf(series)["n"] == 500
    assert kpss(series)["n"] == 500


def test_a_constant_series_is_refused_with_its_measured_variance():
    """Review Focus 1: a constant series makes the ADF regression singular."""
    flat = pd.Series(np.full(400, 3.0))
    for test in (adf, kpss):
        with pytest.raises(Unusable) as exc:
            test(flat)
        assert "variance" in str(exc.value)
        assert "0" in str(exc.value)


def test_too_few_rows_is_refused_with_the_count():
    short = _random_walk(n=MIN_STAT_ROWS - 1)
    for test in (adf, kpss):
        with pytest.raises(Unusable) as exc:
            test(short)
        assert str(MIN_STAT_ROWS - 1) in str(exc.value)


def test_nan_rows_are_dropped_and_the_reported_n_is_the_count_used():
    series = _random_walk(n=500)
    series.iloc[:50] = np.nan
    assert adf(series)["n"] == 450
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_econ_stationarity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.econometrics.stationarity'`

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/econometrics/stationarity.py
"""Unit-root pre-tests (spec 11.4): "are these series stationary? Determines what
regressions are even valid".

PRE-TESTS, not claims. They decide whether a cointegration test has a job, so they are
reported with their p-values and are NOT in the Holm family -- counting a specification
check as a hypothesis would make the claims harder to reject than the evidence warrants.

ADF and KPSS have OPPOSITE null hypotheses: ADF's null is a unit root, KPSS's null is
stationarity. Running both is how a reader tells "evidence of a unit root" from "no
evidence either way", which a single test cannot distinguish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss as _kpss

from cryptoval.econometrics.base import Unusable

# Below this the reference distributions are not worth consulting. The refusal names the
# count rather than returning a statistic nobody should read.
MIN_STAT_ROWS = 60

# A series flatter than this has no variation for a unit-root test to work on, and the
# regression is singular. The refusal carries the measured variance.
MIN_VARIANCE = 1e-12


def _clean(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").astype("float64").dropna().to_numpy()
    if len(values) < MIN_STAT_ROWS:
        raise Unusable(
            f"{name}: {len(values)} usable rows, {MIN_STAT_ROWS} needed for a unit-root test"
        )
    variance = float(np.var(values))
    if variance < MIN_VARIANCE:
        raise Unusable(f"{name}: variance {variance:.3g} is below {MIN_VARIANCE:.0e}, so the "
                       f"regression is singular on {len(values)} rows")
    return values


def adf(series: pd.Series, lag_criterion: str = "aic") -> dict[str, float]:
    """Augmented Dickey-Fuller. Null: the series HAS a unit root."""
    values = _clean(series, "adf")
    statistic, p_value, lags, _, _, _ = adfuller(values, autolag=lag_criterion.upper())
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "lags": float(lags),
        "n": float(len(values)),
    }


def kpss(series: pd.Series) -> dict[str, float]:
    """KPSS. Null: the series IS stationary -- the opposite of ADF's."""
    values = _clean(series, "kpss")
    statistic, p_value, lags, _ = _kpss(values, regression="c", nlags="auto")
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "lags": float(lags),
        "n": float(len(values)),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_econ_stationarity.py -v`
Expected: PASS (7 tests). KPSS's p-value is interpolated from a lookup table and
statsmodels warns when it clips; if `InterpolationWarning` surfaces as an error under the
project's pytest settings, assert on the clipped bound rather than silencing the warning.

- [ ] **Step 5: Mutation proof**

Delete the `variance < MIN_VARIANCE` branch from `_clean` and run the file.
`test_a_constant_series_is_refused_with_its_measured_variance` must FAIL. Restore.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/econometrics/stationarity.py tests/test_econ_stationarity.py
git commit -m "feat: ADF and KPSS, with opposite nulls and a refusal for the singular case"
```

## Task 4: Engle-Granger, with the coefficient estimated

**Files:**
- Create: `cryptoval/econometrics/cointegration.py`
- Test: `tests/test_econ_cointegration.py`

**Interfaces:**
- Consumes: `Unusable` from Task 1.
- Produces: `MIN_COINT_ROWS: int`; `engle_granger(y: pd.Series, x: pd.Series) -> dict[str, float]` with keys `statistic`, `p_value`, `coefficient`, `coefficient_stderr`, `n`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_econ_cointegration.py
import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.econometrics.cointegration import MIN_COINT_ROWS, engle_granger


def _cointegrated(n=900, coefficient=2.0, seed=11):
    """x is a random walk; y is `coefficient` times x plus STATIONARY noise, so the two
    share one stochastic trend and y - coefficient*x is stationary by construction."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 1.0, n))
    y = coefficient * x + rng.normal(0.0, 1.0, n)
    return pd.Series(y), pd.Series(x)


def _independent_walks(n=900, seed=11):
    rng = np.random.default_rng(seed)
    return (
        pd.Series(np.cumsum(rng.normal(0.0, 1.0, n))),
        pd.Series(np.cumsum(rng.normal(0.0, 1.0, n))),
    )


def test_a_cointegrated_pair_is_found():
    assert engle_granger(*_cointegrated())["p_value"] < 0.01


def test_two_independent_random_walks_are_not_found():
    """The other direction, so the test can fail. Two unrelated walks are the null case
    this whole chapter is up against."""
    assert engle_granger(*_independent_walks())["p_value"] > 0.10


def test_the_coefficient_is_estimated_not_assumed():
    """The substantive gain over the ratio form: Metcalfe's exponent becomes testable."""
    result = engle_granger(*_cointegrated(coefficient=2.0))
    assert abs(result["coefficient"] - 2.0) < 0.1
    assert result["coefficient_stderr"] > 0.0
    assert abs(result["coefficient"] - 2.0) < 3.0 * result["coefficient_stderr"]


def test_a_different_true_coefficient_is_recovered():
    """Not vacuous: the estimate tracks the truth rather than returning a constant."""
    assert abs(engle_granger(*_cointegrated(coefficient=0.5))["coefficient"] - 0.5) < 0.1


def test_the_reported_n_is_the_rows_used():
    assert engle_granger(*_cointegrated(n=700))["n"] == 700


def test_a_perfectly_collinear_pair_is_refused_with_its_residual_variance():
    """Review Focus 4: this design exists to avoid testing price against a price-derived
    series, so the guard that catches one must itself be tested. A p-value of 0 here
    would be published as a discovery."""
    x = pd.Series(np.cumsum(np.random.default_rng(3).normal(0.0, 1.0, 800)))
    y = 2.0 * x + 5.0
    with pytest.raises(Unusable) as exc:
        engle_granger(y, x)
    assert "residual variance" in str(exc.value)


def test_mismatched_lengths_are_refused():
    y, x = _cointegrated(n=800)
    with pytest.raises(Unusable):
        engle_granger(y, x.iloc[:700])


def test_too_few_rows_is_refused_with_the_count():
    y, x = _cointegrated(n=MIN_COINT_ROWS - 1)
    with pytest.raises(Unusable) as exc:
        engle_granger(y, x)
    assert str(MIN_COINT_ROWS - 1) in str(exc.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_econ_cointegration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.econometrics.cointegration'`

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/econometrics/cointegration.py
"""Cointegration (spec 11.4): "does price maintain a long-run equilibrium with
fundamentals? ... far stronger than the correlation evidence".

THE COEFFICIENT IS ESTIMATED. A stationary practitioner ratio is the same hypothesis
with the coefficient forced to the value the ratio assumes, so estimating it says whether
the assumed value is the one the data supports. For the Metcalfe pair that converts a
goodness-of-fit statistic into a falsifiable test of the exponent 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from cryptoval.econometrics.base import Unusable

MIN_COINT_ROWS = 120

# A residual this flat means the two series are the same series up to an affine map, so
# the "equilibrium" is an identity. Plan 3 shipped exactly this shape once: velocity_total
# was identically volume/market_cap, and the regression recovered price = market cap /
# supply while looking like an estimate.
MIN_RESIDUAL_VARIANCE = 1e-10


def _aligned(y: pd.Series, x: pd.Series, name: str = "engle_granger") -> tuple[np.ndarray, np.ndarray]:
    """``name`` prefixes the refusal, so a message raised for the VECM does not claim to
    come from Engle-Granger."""
    if len(y) != len(x):
        raise Unusable(f"{name}: {len(y)} dependent rows against {len(x)} independent")
    frame = pd.DataFrame(
        {"y": pd.to_numeric(y, errors="coerce"), "x": pd.to_numeric(x, errors="coerce")}
    ).dropna()
    if len(frame) < MIN_COINT_ROWS:
        raise Unusable(
            f"{name}: {len(frame)} usable rows, {MIN_COINT_ROWS} needed for the "
            f"reference distribution"
        )
    return frame["y"].to_numpy(dtype="float64"), frame["x"].to_numpy(dtype="float64")


def engle_granger(y: pd.Series, x: pd.Series) -> dict[str, float]:
    """The two-step test, plus the cointegrating coefficient and its standard error."""
    y_values, x_values = _aligned(y, x)

    fitted = sm.OLS(y_values, sm.add_constant(x_values)).fit()
    residual_variance = float(np.var(fitted.resid))
    if residual_variance < MIN_RESIDUAL_VARIANCE:
        raise Unusable(
            f"engle_granger: residual variance {residual_variance:.3g} is below "
            f"{MIN_RESIDUAL_VARIANCE:.0e} on {len(y_values)} rows, so the independent series "
            f"is an affine function of the dependent one and the equilibrium is an identity"
        )

    statistic, p_value, _ = coint(y_values, x_values)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "coefficient": float(fitted.params[1]),
        "coefficient_stderr": float(fitted.bse[1]),
        "n": float(len(y_values)),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_econ_cointegration.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Mutation proof**

Replace `float(fitted.params[1])` with the constant `1.0` and run the file.
`test_the_coefficient_is_estimated_not_assumed` and
`test_a_different_true_coefficient_is_recovered` must both FAIL — that pair of tests is
what stops a "coefficient" column from being a hard-coded assumption. Restore.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/econometrics/cointegration.py tests/test_econ_cointegration.py
git commit -m "feat: Engle-Granger with the cointegrating coefficient estimated, and the identity case refused"
```

## Task 5: Johansen, as the second opinion

**Files:**
- Modify: `cryptoval/econometrics/cointegration.py`
- Test: `tests/test_econ_cointegration.py` (extend)

**Interfaces:**
- Produces: `johansen(y: pd.Series, x: pd.Series, lags: int = 1) -> dict[str, float]` with keys `trace_statistic`, `critical_95`, `rank`, `n`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_econ_cointegration.py
from cryptoval.econometrics.cointegration import johansen


def test_johansen_finds_one_cointegrating_relationship_in_a_cointegrated_pair():
    result = johansen(*_cointegrated())
    assert result["rank"] == 1
    assert result["trace_statistic"] > result["critical_95"]


def test_johansen_finds_none_between_independent_walks():
    result = johansen(*_independent_walks())
    assert result["rank"] == 0
    assert result["trace_statistic"] < result["critical_95"]


def test_johansen_refuses_too_few_rows_with_the_count():
    y, x = _cointegrated(n=MIN_COINT_ROWS - 1)
    with pytest.raises(Unusable) as exc:
        johansen(y, x)
    assert str(MIN_COINT_ROWS - 1) in str(exc.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_econ_cointegration.py -k johansen -v`
Expected: FAIL — `ImportError: cannot import name 'johansen'`

- [ ] **Step 3: Write the implementation**

```python
# append to cryptoval/econometrics/cointegration.py
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen(y: pd.Series, x: pd.Series, lags: int = 1) -> dict[str, float]:
    """The trace test at rank 0, reported as the second opinion on Engle-Granger.

    Engle-Granger picks one series as the dependent and can give different answers
    depending on which; Johansen is symmetric in the two. Reporting both is how the
    chapter says whether the finding survives that choice.
    """
    y_values, x_values = _aligned(y, x)
    result = coint_johansen(np.column_stack([y_values, x_values]), det_order=0, k_ar_diff=lags)
    trace = float(result.lr1[0])
    critical = float(result.cvt[0, 1])  # column 1 is the 95% critical value
    return {
        "trace_statistic": trace,
        "critical_95": critical,
        "rank": float(1.0 if trace > critical else 0.0),
        "n": float(len(y_values)),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_econ_cointegration.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Mutation proof**

Change `1.0 if trace > critical else 0.0` to `1.0` and run the file.
`test_johansen_finds_none_between_independent_walks` must FAIL — a rank that is always 1
reports an equilibrium between two unrelated random walks, which is the exact false
positive this chapter is up against.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/econometrics/cointegration.py tests/test_econ_cointegration.py
git commit -m "feat: the Johansen trace test, symmetric where Engle-Granger must pick a side"
```

## Task 6: VECM and the speed of adjustment

**Files:**
- Create: `cryptoval/econometrics/vecm.py`
- Test: `tests/test_econ_vecm.py`

**Interfaces:**
- Consumes: `_aligned` and `MIN_COINT_ROWS` from Task 4 (`_aligned` raises the row-count refusal, prefixed `vecm:` via its `name` argument, so this module does not repeat the check).
- Produces: `speed_of_adjustment(y: pd.Series, x: pd.Series, lags: int = 1) -> dict[str, float]` with keys `alpha`, `alpha_stderr`, `half_life_days`, `lags`, `n`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_econ_vecm.py
import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.econometrics.cointegration import MIN_COINT_ROWS
from cryptoval.econometrics.vecm import speed_of_adjustment


def _mean_reverting(n=900, speed=0.1, seed=5):
    """y tracks x with an error-correction term pulling it back at `speed` per day, so
    the fitted alpha has a known sign and rough magnitude."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 1.0, n))
    y = np.zeros(n)
    for t in range(1, n):
        gap = y[t - 1] - x[t - 1]
        y[t] = y[t - 1] + (x[t] - x[t - 1]) - speed * gap + rng.normal(0.0, 0.1)
    return pd.Series(y), pd.Series(x)


def test_the_adjustment_term_is_negative_for_a_mean_reverting_pair():
    """A NEGATIVE alpha is what reversion means: a gap above equilibrium pulls down."""
    assert speed_of_adjustment(*_mean_reverting())["alpha"] < 0.0


def test_a_faster_pull_gives_a_larger_adjustment_and_a_shorter_half_life():
    """Not vacuous: the estimate tracks the simulated speed rather than its own sign."""
    slow = speed_of_adjustment(*_mean_reverting(speed=0.02))
    fast = speed_of_adjustment(*_mean_reverting(speed=0.30))
    assert abs(fast["alpha"]) > abs(slow["alpha"])
    assert fast["half_life_days"] < slow["half_life_days"]


def test_the_half_life_is_none_when_the_pull_is_the_wrong_sign():
    """A positive alpha is divergence, not reversion, and has no half life. Reporting a
    number there would read as a reversion speed."""
    rng = np.random.default_rng(1)
    x = pd.Series(np.cumsum(rng.normal(0.0, 1.0, 900)))
    y = pd.Series(np.cumsum(rng.normal(0.0, 1.0, 900)))
    result = speed_of_adjustment(y, x)
    if result["alpha"] >= 0.0:
        assert np.isnan(result["half_life_days"])


def test_the_reported_n_and_lags_are_what_was_used():
    result = speed_of_adjustment(*_mean_reverting(n=700), lags=2)
    assert result["n"] == 700
    assert result["lags"] == 2


def test_too_few_rows_is_refused_with_the_count():
    y, x = _mean_reverting(n=MIN_COINT_ROWS - 1)
    with pytest.raises(Unusable) as exc:
        speed_of_adjustment(y, x)
    assert str(MIN_COINT_ROWS - 1) in str(exc.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_econ_vecm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.econometrics.vecm'`

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/econometrics/vecm.py
"""The vector error-correction model (spec 11.4): "short-run dynamics and the SPEED of
mean reversion toward that equilibrium".

Alpha is the answer: the share of yesterday's disequilibrium corrected today. Its SIGN is
the whole finding -- negative is reversion, positive is divergence -- so a half life is
reported only where the sign permits one.

Fitted only where cointegration survived the correction. A VECM on a pair with no
equilibrium estimates the speed of return to a relationship that does not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM

from cryptoval.econometrics.cointegration import _aligned


def speed_of_adjustment(y: pd.Series, x: pd.Series, lags: int = 1) -> dict[str, float]:
    y_values, x_values = _aligned(y, x, name="vecm")

    fitted = VECM(
        np.column_stack([y_values, x_values]), k_ar_diff=lags, coint_rank=1, deterministic="ci"
    ).fit()
    alpha = float(fitted.alpha[0, 0])
    stderr = float(fitted.stderr_alpha[0, 0])

    # log(0.5)/log(1+alpha) is the days for half the gap to close, and it exists only
    # while -2 < alpha < 0. Outside that the pair diverges or oscillates, and a number
    # there would be read as a reversion speed.
    half_life = float("nan")
    if -2.0 < alpha < 0.0:
        half_life = float(np.log(0.5) / np.log1p(alpha))

    return {
        "alpha": alpha,
        "alpha_stderr": stderr,
        "half_life_days": half_life,
        "lags": float(lags),
        "n": float(len(y_values)),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_econ_vecm.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Mutation proof**

Remove the `-2.0 < alpha < 0.0` guard so the half life is always computed, and run the
file. `test_the_half_life_is_none_when_the_pull_is_the_wrong_sign` must FAIL when alpha
comes back positive (it raises a domain warning or returns a negative half life). Restore.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/econometrics/vecm.py tests/test_econ_vecm.py
git commit -m "feat: the VECM adjustment term, with a half life only where its sign allows one"
```

---

# Stage 4b-3 — The table, the command, and the chapter

## Task 7: TEST_COLUMNS, the runner, and the generalized Holm correction

**Files:**
- Create: `cryptoval/econometrics/runner.py`
- Modify: `cryptoval/scoring/multiplicity.py`
- Test: `tests/test_econ_runner.py`, `tests/test_scoring_multiplicity.py` (extend)

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: `TEST_COLUMNS: list[str]`; `run_pairs(levels: pd.DataFrame, specs: dict[str, PairSpec], panel_vintage_ts: pd.Timestamp | None = None) -> pd.DataFrame`. `correct_family_wise` gains keyword-only `p_metric: str = "dm_p_value"`, `columns: list[str] | None = None`, `count_keys: dict[str, Any] | None = None`.

**The metric vocabulary**, globally unique so one correction can key on `metric` alone:
`adf_statistic`, `adf_p_value`, `kpss_statistic`, `kpss_p_value`,
`independent_adf_statistic`, `independent_adf_p_value`, `independent_kpss_statistic`,
`independent_kpss_p_value`, `ratio_adf_statistic`,
`ratio_adf_p_value`, `ratio_kpss_statistic`, `ratio_kpss_p_value`, `cointegration_statistic`,
`cointegration_p_value`, `cointegration_p_value_holm`, `cointegration_coefficient`,
`cointegration_coefficient_stderr`, `johansen_trace_statistic`, `johansen_critical_95`,
`johansen_rank`, `vecm_alpha`, `vecm_alpha_stderr`, `vecm_half_life_days`,
`hypotheses_tested`, `refused`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_econ_runner.py
import numpy as np
import pandas as pd

from cryptoval.econometrics.base import PairSpec
from cryptoval.econometrics.runner import TEST_COLUMNS, run_pairs
from cryptoval.econometrics.series import build_levels

SPECS = {
    "mvrv_anchor": PairSpec(
        pair_id="mvrv_anchor",
        dependent="log_market_cap",
        independent="log_realized_cap",
        forced_coefficient=1.0,
        params={"lag_criterion": "aic"},
    )
}


def _panel(days=1200, symbols=("BTC",), seed=4):
    rng = np.random.default_rng(seed)
    out = []
    for symbol in symbols:
        realized = np.cumsum(rng.normal(0.002, 0.01, days)) + 20.0
        out.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "metric_date": pd.date_range("2019-01-01", periods=days, freq="D"),
                    "market_cap_usd": np.exp(realized + rng.normal(0.0, 0.05, days)),
                    "realized_cap": np.exp(realized),
                    "volume_total_usd": np.exp(realized - 3.0),
                    "active_addresses": np.exp(realized / 2.0),
                    "panel_vintage_ts": pd.Timestamp("2026-10-12 08:30", tz="UTC"),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def test_the_runner_emits_the_documented_columns_and_grain():
    rows = run_pairs(build_levels(_panel()), SPECS)
    assert list(rows.columns) == TEST_COLUMNS
    assert not rows.duplicated(subset=["pair_id", "symbol", "metric"]).any()


def test_every_statistic_the_chain_produces_reaches_the_table():
    rows = run_pairs(build_levels(_panel()), SPECS)
    metrics = set(rows["metric"])
    assert {
        "adf_statistic",
        "kpss_p_value",
        "independent_adf_p_value",
        "ratio_adf_p_value",
        "cointegration_p_value",
        "cointegration_coefficient",
        "johansen_rank",
        "vecm_alpha",
    } <= metrics


def test_the_holm_row_counts_only_the_cointegration_tests():
    """Global Constraint: ADF/KPSS are pre-tests and VECM coefficients are estimates.
    Counting them would inflate the correction."""
    rows = run_pairs(build_levels(_panel(symbols=("BTC", "ETH"))), SPECS)
    count = rows[rows["metric"] == "hypotheses_tested"]["value"].iloc[0]
    assert count == 2.0  # one pair, two symbols
    assert (rows["metric"] == "cointegration_p_value_holm").sum() == 2


def test_a_refused_pair_is_a_row_with_the_measurement_not_a_missing_asset():
    panel = _panel(days=1200, symbols=("BTC", "SHORT"))
    panel = panel[(panel["symbol"] != "SHORT") | (panel.groupby("symbol").cumcount() < 100)]
    rows = run_pairs(build_levels(panel), SPECS)
    refused = rows[(rows["symbol"] == "SHORT") & (rows["metric"] == "refused")]
    assert len(refused) == 1
    assert refused["value"].isna().all()
    assert "100 rows" in refused["reason"].iloc[0]


def test_every_row_carries_the_vintage_and_the_params_hash():
    vintage = pd.Timestamp("2026-10-12 08:30", tz="UTC")
    rows = run_pairs(build_levels(_panel()), SPECS, panel_vintage_ts=vintage)
    assert (rows["panel_vintage_ts"] == vintage).all()
    assert rows["params_hash"].notna().all()
    assert rows["params_hash"].nunique() == 1
```

```python
# append to tests/test_scoring_multiplicity.py
def test_the_correction_generalizes_to_another_metric_name_and_column_set():
    columns = ["pair_id", "symbol", "metric", "value", "reason"]
    scores = pd.DataFrame(
        [
            {"pair_id": "p", "symbol": "BTC", "metric": "cointegration_p_value",
             "value": 0.001, "reason": None},
            {"pair_id": "p", "symbol": "ETH", "metric": "cointegration_p_value",
             "value": 0.040, "reason": None},
        ]
    )
    out = correct_family_wise(
        scores,
        p_metric="cointegration_p_value",
        columns=columns,
        count_keys={"pair_id": "*", "symbol": "*"},
    )
    assert list(out.columns) == columns
    assert (out["metric"] == "cointegration_p_value_holm").sum() == 2
    assert out[out["metric"] == "hypotheses_tested"]["value"].iloc[0] == 2.0


def test_a_second_pass_is_refused_for_a_generalized_metric_too():
    """Review Focus 5: the generalization must not open a hole in the guard."""
    columns = ["pair_id", "symbol", "metric", "value", "reason"]
    scores = pd.DataFrame(
        [{"pair_id": "p", "symbol": "BTC", "metric": "cointegration_p_value",
          "value": 0.01, "reason": None}]
    )
    once = correct_family_wise(
        scores, p_metric="cointegration_p_value", columns=columns,
        count_keys={"pair_id": "*", "symbol": "*"},
    )
    with pytest.raises(ValueError) as exc:
        correct_family_wise(
            once, p_metric="cointegration_p_value", columns=columns,
            count_keys={"pair_id": "*", "symbol": "*"},
        )
    assert "already" in str(exc.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_econ_runner.py tests/test_scoring_multiplicity.py -v`
Expected: FAIL — no `cryptoval.econometrics.runner`, and `correct_family_wise` rejects the
unexpected `p_metric` keyword.

- [ ] **Step 3: Generalize the correction**

```python
# cryptoval/scoring/multiplicity.py -- replace the signature and the three hardcoded names
def correct_family_wise(
    scores: pd.DataFrame,
    alpha: float = ALPHA,
    *,
    p_metric: str = "dm_p_value",
    columns: list[str] | None = None,
    count_keys: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Append Holm-adjusted p-values and the hypothesis count to ``scores``.

    A refused test (NaN value, reason set) is NOT a hypothesis: counting it would inflate
    the correction and make the surviving claims harder to reject than the evidence
    warrants, which is a different error from the one 11.6 guards against but an error all
    the same.

    The keyword arguments exist so Plan 4b's econometrics table shares this
    implementation -- and with it the refusal of a second pass -- rather than carrying a
    second copy of Holm. Their defaults are Plan 4's behaviour exactly.
    """
    columns = list(columns if columns is not None else SCORE_COLUMNS)
    holm_metric = f"{p_metric}_holm"
    keys = dict(
        count_keys
        if count_keys is not None
        else {"model_id": "*", "symbol": "*", "horizon": -1, "fold": -1}
    )

    n_holm = int((scores["metric"] == holm_metric).sum())
    n_count = int((scores["metric"] == "hypotheses_tested").sum())
    if n_holm or n_count:
        raise ValueError(
            f"input already has {n_holm} {holm_metric} and {n_count} hypotheses_tested rows; "
            f"the correction was already applied, so re-correct from the raw {p_metric} rows"
        )

    tests = scores[(scores["metric"] == p_metric) & scores["value"].notna()].copy()
    n = int(len(tests))

    count_row = dict.fromkeys(columns)
    count_row.update(keys)
    count_row.update(
        metric="hypotheses_tested",
        value=float(n),
        reason=f"Holm-Bonferroni family-wise correction at alpha={alpha}",
    )

    if n == 0:
        return pd.concat([scores, pd.DataFrame([count_row])], ignore_index=True)[columns]

    _, adjusted, _, _ = multipletests(
        tests["value"].to_numpy(dtype="float64"), alpha=alpha, method="holm"
    )
    corrected = tests.copy()
    corrected["metric"] = holm_metric
    corrected["value"] = np.asarray(adjusted, dtype="float64")
    corrected["reason"] = f"Holm over {n} hypotheses"

    return pd.concat([scores, corrected, pd.DataFrame([count_row])], ignore_index=True)[columns]
```

Add `from typing import Any` to that module's imports.

- [ ] **Step 4: Write the runner**

```python
# cryptoval/econometrics/runner.py
"""One long table for the whole anchoring chain (spec 11.4), at grain
(pair_id, symbol, metric).

The same contract as fct_model_score: a refusal is a ROW carrying the measurement that
caused it, not a silently absent asset, because which assets could not be tested is a
finding this chapter has to report.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptoval.econometrics.base import PairSpec, Unusable
from cryptoval.econometrics.cointegration import engle_granger, johansen
from cryptoval.econometrics.series import pair_rows
from cryptoval.econometrics.stationarity import adf, kpss
from cryptoval.econometrics.vecm import speed_of_adjustment
from cryptoval.hashing import stable_hash
from cryptoval.scoring.multiplicity import correct_family_wise

TEST_COLUMNS = [
    "pair_id",
    "symbol",
    "metric",
    "value",
    "n_rows",
    "lags",
    "reason",
    "params_hash",
    "panel_vintage_ts",
]

COUNT_KEYS = {"pair_id": "*", "symbol": "*"}


def _row(**values: Any) -> dict[str, Any]:
    row = dict.fromkeys(TEST_COLUMNS)
    row.update(values)
    return row


def run_pairs(
    levels: pd.DataFrame,
    specs: dict[str, PairSpec],
    panel_vintage_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Every pair against every symbol in ``levels``, full sample."""
    rows: list[dict[str, Any]] = []

    for pair_id, spec in sorted(specs.items()):
        params_hash = stable_hash(spec.params)
        common = {"pair_id": pair_id, "params_hash": params_hash,
                  "panel_vintage_ts": panel_vintage_ts}

        for symbol in sorted(levels["symbol"].unique()):
            try:
                pair = pair_rows(levels, str(symbol), spec)
            except Unusable as exc:
                rows.append(_row(symbol=str(symbol), metric="refused", reason=str(exc), **common))
                continue

            n = float(len(pair))
            # BOTH series, not just the dependent one: cointegration is a statement about
            # two I(1) series, so a reader cannot judge the finding without the
            # independent series' unit-root evidence too.
            for name, series in (
                ("", pair["y"]),
                ("independent_", pair["x"]),
                ("ratio_", pair["ratio"]),
            ):
                for test, fn in (("adf", adf), ("kpss", kpss)):
                    try:
                        result = fn(series)
                    except Unusable as exc:
                        rows.append(_row(symbol=str(symbol), metric=f"{name}{test}_statistic",
                                         reason=str(exc), n_rows=n, **common))
                        continue
                    for key in ("statistic", "p_value"):
                        rows.append(_row(symbol=str(symbol), metric=f"{name}{test}_{key}",
                                         value=result[key], n_rows=result["n"],
                                         lags=result["lags"], **common))

            try:
                eg = engle_granger(pair["y"], pair["x"])
            except Unusable as exc:
                rows.append(_row(symbol=str(symbol), metric="cointegration_statistic",
                                 reason=str(exc), n_rows=n, **common))
                continue

            for key, metric in (
                ("statistic", "cointegration_statistic"),
                ("p_value", "cointegration_p_value"),
                ("coefficient", "cointegration_coefficient"),
                ("coefficient_stderr", "cointegration_coefficient_stderr"),
            ):
                rows.append(_row(symbol=str(symbol), metric=metric, value=eg[key],
                                 n_rows=eg["n"], **common))

            jo = johansen(pair["y"], pair["x"])
            for key, metric in (
                ("trace_statistic", "johansen_trace_statistic"),
                ("critical_95", "johansen_critical_95"),
                ("rank", "johansen_rank"),
            ):
                rows.append(_row(symbol=str(symbol), metric=metric, value=jo[key],
                                 n_rows=jo["n"], **common))

            vecm = speed_of_adjustment(pair["y"], pair["x"])
            for key, metric in (
                ("alpha", "vecm_alpha"),
                ("alpha_stderr", "vecm_alpha_stderr"),
                ("half_life_days", "vecm_half_life_days"),
            ):
                rows.append(_row(symbol=str(symbol), metric=metric, value=vecm[key],
                                 n_rows=vecm["n"], lags=vecm["lags"], **common))

    table = pd.DataFrame(rows, columns=TEST_COLUMNS)
    return correct_family_wise(
        table,
        p_metric="cointegration_p_value",
        columns=TEST_COLUMNS,
        count_keys=COUNT_KEYS,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_econ_runner.py tests/test_scoring_multiplicity.py tests/test_cli.py -v`
Expected: PASS. `tests/test_cli.py` must pass unchanged — the generalization's defaults are
Plan 4's behaviour, and that file is the proof.

- [ ] **Step 6: Mutation proof**

Change `p_metric="cointegration_p_value"` in `run_pairs` to `p_metric="adf_p_value"` and run
`tests/test_econ_runner.py`. `test_the_holm_row_counts_only_the_cointegration_tests` must
FAIL (the count becomes 4, not 2). Restore.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/econometrics/runner.py cryptoval/scoring/multiplicity.py tests/test_econ_runner.py tests/test_scoring_multiplicity.py
git commit -m "feat: the econometrics table, and one Holm implementation serving both plans"
```

## Task 8: The `econ` command, the mart, and the column-list test

**Files:**
- Modify: `cryptoval/cli.py`, `cryptoval/scoring/registry.py`, `Makefile`
- Create: `dbt/models/marts/fct_econometric_test.sql`, and the `fct_econometric_test` block in `dbt/models/marts/_marts.yml`
- Test: `tests/test_cli.py` (extend), `tests/test_marts_contract.py`

**Interfaces:**
- Consumes: `TEST_COLUMNS`, `run_pairs` from Task 7; `PAIRS` from Task 2.
- Produces: `_read_econ_inputs(settings) -> pd.DataFrame`; `run_econ(settings, assets, pair_ids=None) -> dict[str, dict]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_marts_contract.py
"""The mart's select list must agree with the constant that writes gold.

The Plan 4 whole-branch review found nothing tied SCORE_COLUMNS to fct_model_score.sql, so
a column added to the constant would reach gold and never reach the mart, with no test
failing. This is that test, for both marts.
"""

import pathlib
import re

from cryptoval.econometrics.runner import TEST_COLUMNS
from cryptoval.scoring.runner import SCORE_COLUMNS


def _selected_columns(sql_path: str) -> list[str]:
    text = pathlib.Path(sql_path).read_text(encoding="utf-8")
    body = text[text.lower().index("select") : text.lower().rindex("from")]
    out = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--") or stripped.lower() == "select":
            continue
        match = re.fullmatch(r"([a-z_][a-z0-9_]*),?", stripped)
        if match:
            out.append(match.group(1))
    return out


def test_the_score_mart_selects_exactly_score_columns():
    assert _selected_columns("dbt/models/marts/fct_model_score.sql") == SCORE_COLUMNS


def test_the_econometrics_mart_selects_exactly_test_columns():
    assert _selected_columns("dbt/models/marts/fct_econometric_test.sql") == TEST_COLUMNS
```

```python
# append to tests/test_cli.py
def test_econ_writes_the_documented_columns_to_gold(tmp_path, monkeypatch):
    from cryptoval.econometrics.runner import TEST_COLUMNS

    settings = Settings(data_root=str(tmp_path))
    levels_panel = _econ_inputs()
    monkeypatch.setattr(cli, "_read_econ_inputs", lambda _settings: levels_panel)
    report = cli.run_econ(settings, [])

    written = pd.read_parquet(
        pathlib.Path(settings.resolver.table("gold", "econometric_test")) / "part-0.parquet"
    )
    assert list(written.columns) == TEST_COLUMNS
    assert (written["metric"] == "hypotheses_tested").any()
    assert report["mvrv_anchor"]["rows"] > 0


def test_econ_refuses_an_unknown_pair_at_the_cli(capsys):
    assert cli.main(["econ", "--pairs", "not_a_pair"]) == 2
    assert "unknown pair" in capsys.readouterr().err
```

Add the fixture next to `_score_inputs`:

```python
def _econ_inputs(days=1200, symbols=("BTC",), seed=4):
    rng = np.random.default_rng(seed)
    out = []
    for symbol in symbols:
        trend = np.cumsum(rng.normal(0.002, 0.01, days)) + 20.0
        out.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "metric_date": pd.date_range("2019-01-01", periods=days, freq="D"),
                    "market_cap_usd": np.exp(trend + rng.normal(0.0, 0.05, days)),
                    "realized_cap": np.exp(trend),
                    "volume_total_usd": np.exp(trend - 3.0),
                    "active_addresses": np.exp(trend / 2.0),
                    "panel_vintage_ts": pd.Timestamp("2026-10-12 08:30", tz="UTC"),
                }
            )
        )
    return pd.concat(out, ignore_index=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_marts_contract.py tests/test_cli.py -k "econ or mart" -v`
Expected: FAIL — no `fct_econometric_test.sql`, no `run_econ`.

- [ ] **Step 3: Write the CLI command**

```python
# cryptoval/cli.py -- additions
from cryptoval.econometrics.registry import PAIRS
from cryptoval.econometrics.runner import TEST_COLUMNS, run_pairs
from cryptoval.econometrics.series import LEVEL_SERIES, build_levels


def _read_econ_inputs(settings: Settings) -> pd.DataFrame:
    """The levels panel. Separate from _read_score_inputs: this chapter needs the
    FUNDAMENTALS in levels and no estimates at all. Patched wholesale in tests."""
    import duckdb

    con = duckdb.connect(str(pathlib.Path(settings.data_root) / "cryptoval.duckdb"), read_only=True)
    try:
        columns = ", ".join(["symbol", "metric_date", "panel_vintage_ts", *LEVEL_SERIES.values()])
        return con.execute(f"select {columns} from fct_asset_metrics_daily").fetchdf()
    finally:
        con.close()


def run_econ(
    settings: Settings, assets: list[Asset], pair_ids: list[str] | None = None
) -> dict[str, dict]:
    panel = _read_econ_inputs(settings)
    specs = {p: PAIRS[p] for p in pair_ids} if pair_ids else dict(PAIRS)
    vintage = panel["panel_vintage_ts"].max() if "panel_vintage_ts" in panel.columns else None

    levels = build_levels(panel)
    table = run_pairs(levels, specs, panel_vintage_ts=vintage)
    write_gold(settings.resolver, "econometric_test", table)

    report: dict[str, dict] = {}
    for pair_id, group in table[table["pair_id"] != "*"].groupby("pair_id"):
        holm = group[(group["metric"] == "cointegration_p_value_holm") & (group["value"] < 0.05)]
        report[str(pair_id)] = {
            "rows": int(len(group)),
            "refusals": int((group["metric"] == "refused").sum()),
            "symbols": int(group["symbol"].nunique()),
            "cointegrated_after_holm": int(len(holm)),
        }
    return report
```

Unlike `score`, a `--pairs` subset needs no carry-forward of unrun pairs: this plan's first
run writes the whole table, and a subset run is a development convenience. Say so in a
comment rather than silently differing from `score`:

```python
    # No _keep_unrun equivalent: unlike score, a --pairs subset REPLACES the table. The
    # pairs are three, a full run takes seconds, and a partial table nobody can detect is
    # worse than one that is obviously partial.
```

Wire the subcommand in `main`, beside `score`:

```python
    ec = sub.add_parser("econ")
    ec.add_argument("--pairs", help="comma-separated pair ids; default every registered pair")
```

```python
    if args.command == "econ":
        ids = [p.strip() for p in args.pairs.split(",")] if args.pairs else None
        unknown = [p for p in ids or [] if p not in PAIRS]
        if unknown:
            print(f"unknown pair(s): {unknown}; known: {sorted(PAIRS)}", file=sys.stderr)
            return 2
        for pair_id, outcome in sorted(run_econ(settings, assets, ids).items()):
            print(pair_id, outcome)
        # No cointegration found is the expected result, not a failed run.
        return 0
```

- [ ] **Step 4: Write the mart**

```sql
-- dbt/models/marts/fct_econometric_test.sql
-- The anchoring chain at grain (pair_id, symbol, metric), spec 11.4.
--
-- FULL SAMPLE, by design. Every statistic here is computed over the whole available
-- history and therefore uses future data; this table supports no forecasting claim and is
-- not comparable to fct_model_score, which is strictly out of sample.
--
-- pair_id = '*' holds the hypothesis count the family-wise correction was taken over, and
-- that family is the cointegration tests ONLY: ADF/KPSS are pre-tests that decide which
-- regressions are valid, and VECM coefficients estimate an already-established
-- relationship.
--
-- A 'refused' row carries a NULL value and the measurement in reason, the same contract
-- fct_valuation_estimate and fct_model_score use.
select
    pair_id,
    symbol,
    metric,
    value,
    n_rows,
    lags,
    reason,
    params_hash,
    panel_vintage_ts
from read_parquet('{{ var("data_root") }}/gold/econometric_test/*.parquet')
```

```yaml
# dbt/models/marts/_marts.yml -- new block beside fct_model_score
  - name: fct_econometric_test
    description: >
      The anchoring chain at grain (pair_id, symbol, metric), spec 11.4. Full sample and
      therefore descriptive: it uses future data by construction, supports no tradability
      claim, and is not comparable to fct_model_score. pair_id = '*' holds the hypothesis
      count, taken over the cointegration tests only. A row whose metric is 'refused'
      carries a NULL value and the measurement that caused it.
    tests:
      - dbt_utils.unique_combination_of_columns:
          arguments:
            combination_of_columns: [pair_id, symbol, metric]
    columns:
      - name: pair_id
        tests: [not_null]
      - name: metric
        tests: [not_null]
      - name: params_hash
        description: >
          stable_hash of the pair's params. Re-running after editing a pair's YAML writes
          rows a reader can tell apart from the old ones -- the provenance fct_model_score
          lacks.
        tests: [not_null]
```

- [ ] **Step 5: Delete the dead `param_set_hash`**

```bash
grep -rn "param_set_hash" cryptoval/ tests/
```

It appears only at `cryptoval/scoring/registry.py:39-40`. Nothing calls it, and
`fct_econometric_test.params_hash` now proves the pattern it was reserved for. Delete the
function. If the grep shows any caller, stop and wire it instead.

- [ ] **Step 6: Add `econ` to the Makefile**

```makefile
econ:
	$(PY) -m cryptoval.cli econ
```

and in the `all` recipe, after `score` and before the final `dbt`:

```makefile
	$(MAKE) econ
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, with the two new mart-contract tests and the two new CLI tests included.

- [ ] **Step 8: Mutation proof**

Append `, "an_extra_column"` to `TEST_COLUMNS` and run `tests/test_marts_contract.py`.
`test_the_econometrics_mart_selects_exactly_test_columns` must FAIL — that is the whole
point of the test. Restore. Then delete `panel_vintage_ts` from the mart's select list and
confirm the same test FAILS from the other side.

- [ ] **Step 9: Commit**

```bash
git add cryptoval/cli.py cryptoval/scoring/registry.py dbt/models/marts/ Makefile tests/
git commit -m "feat: the econ command, the fct_econometric_test mart, and a test that ties both marts to their column constants"
```

## Task 9: The live run and the chapter

**Files:**
- Create: `reports/does-price-anchor-to-fundamentals.md`
- Test: none new — this task runs the pipeline and writes prose.

- [ ] **Step 1: Run the chain against the real warehouse**

```bash
.venv/bin/python -m cryptoval.cli econ
cd dbt && ../.venv/bin/dbt build --select fct_econometric_test --profiles-dir . && cd ..
```

Record the printed report per pair. Expect refusals: TRX and SOL on `realized_cap`, DOT and
BNB near the 730-row floor.

- [ ] **Step 2: Query the measurements the chapter needs**

```bash
.venv/bin/python -I -c "
import duckdb
con = duckdb.connect('data/cryptoval.duckdb', read_only=True)
q = lambda s: con.execute(s).fetchdf()
print(q('''select pair_id, count(distinct symbol) as symbols,
             sum(case when metric = 'refused' then 1 else 0 end) as refusals
      from fct_econometric_test where pair_id <> '*' group by 1 order by 1'''))
print(q('''select pair_id, symbol, value as holm_p
      from fct_econometric_test
      where metric = 'cointegration_p_value_holm' and value < 0.05
      order by 3'''))
print(q('''select pair_id, symbol, value as coefficient
      from fct_econometric_test where metric = 'cointegration_coefficient'
      order by 1, 2'''))
print(q('''select symbol, value as alpha from fct_econometric_test
      where pair_id = 'metcalfe_anchor' and metric = 'vecm_alpha' order by 2'''))
"
```

- [ ] **Step 3: Write the chapter**

Sections, each carrying its own measurements — no claim without the number behind it:

1. **What this tests and what it cannot.** The three pairs, the estimated-coefficient form
   and why (the circularity argument from the design doc, including Plan 3's
   `velocity_total` precedent). State in this section, not a footnote: full sample, uses
   future data by construction, no tradability claim, **not** comparable to Plan 4's
   out-of-sample null, and a finding here does not soften that null.
2. **The universe, measured.** Assets tested per pair and every refusal with its count.
3. **Stationarity, as a pre-test.** ADF and KPSS on **both** series' levels and on the
   constrained ratios, with the two tests' opposite nulls spelled out. Say plainly that
   cointegration presupposes both series are I(1), and name any asset where that
   presupposition fails — a cointegration p-value computed on a pair that fails it is
   reported with that caveat attached, not quietly alongside the others. State that these
   pre-tests are not in the Holm family and why.
4. **Cointegration.** The headline table: statistic, raw p-value, Holm-adjusted p-value and
   estimated coefficient per pair and asset, with the hypothesis count. Johansen beside
   Engle-Granger, and where the two disagree, say so rather than picking the agreeable one.
5. **Metcalfe's exponent.** The estimated coefficient against the law's 2, with the
   interval. This is the one place a number is predicted in advance.
6. **Speed of adjustment.** VECM alpha and the half life, only for pairs that survived the
   correction, and only where alpha's sign permits a half life.
7. **Declared gaps.** All four from the design doc §9, each as a row with its measurement:
   structural breaks with Gregory-Hansen named; `volume_total_usd` is exchange volume and
   this pair is not NVT; bivariate only; lag order from AIC, not searched.

- [ ] **Step 4: Verify every number in the chapter against the warehouse**

Re-run each query in Step 2 and check the figures in the prose against the output. A figure
that cannot be reproduced by a query in the chapter's own reproduction section does not go
in the chapter.

- [ ] **Step 5: Commit**

```bash
git add reports/does-price-anchor-to-fundamentals.md
git commit -m "feat: the anchoring run, and the chapter that reports what it found"
```

---

## Self-review notes

Checked against the design doc, section by section:

- §1 the question → Task 9's chapter section 1.
- §2 fundamental form, ratios as the constrained case → Tasks 1 (the `ratio` column, derived
  from `forced_coefficient`), 2 (Metcalfe forces 2), 4 (the coefficient is estimated), 7
  (both `adf_*` and `ratio_adf_*` reach the table).
- §3 full sample and labelled → Global Constraints, the mart's header comment, Task 9's
  section 1. No splits module is imported anywhere in this plan.
- §4 measured universe → Task 1's `MIN_PAIR_ROWS` refusal, Task 7's refusal rows, Task 9
  section 2.
- §5 architecture → the File Structure table; every module in it has a task.
- §6 table shape and the SQL assertion → Tasks 7 and 8, with `tests/test_marts_contract.py`
  covering `fct_model_score` too.
- §7 multiple testing → Task 7's `p_metric="cointegration_p_value"` and its mutation proof.
- §8 carry-overs → `param_set_hash` deleted in Task 8 Step 5; `SCORE_COLUMNS`-to-SQL fixed in
  Task 8; `features_used` deferred, as the design states.
- §9 declared gaps → Task 9 section 7.
- §10 testing strategy → known-answer tests in both directions in Tasks 3–6, a mutation
  proof step in every task.
- §11 out of scope → no task touches GARCH, Markov-switching, DCC or §11.5.

Six problems the self-review found in this plan's own first draft, all fixed above: the
Tech Stack line pinned the wrong statsmodels version; `_aligned` raised refusals labelled
`engle_granger` when the VECM called it; the stationarity loop tested only the dependent
series when cointegration presupposes **both** are I(1); one assertion was an unreadable
nested-numpy expression; and two mutation proofs were written so that they might not fail.

**One design-doc correction this plan makes:** the design says `correct_family_wise` is
"reused **unchanged**". It cannot be — it hardcodes `dm_p_value` and `SCORE_COLUMNS`. Task 7
generalizes it with keyword-only arguments defaulting to Plan 4's exact behaviour, which is
still one implementation rather than two, and `tests/test_cli.py` passing unchanged is the
proof. The design doc should be amended to say "reused, generalized by three keyword
arguments whose defaults are Plan 4's behaviour".
