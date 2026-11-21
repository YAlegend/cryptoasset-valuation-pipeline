# UTXO Extraction and Family E(b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract Bitcoin's creation × spend coin-age matrix from the BigQuery sandbox once, then derive
the Cointime, HODL-wave and dissertation signals from it locally. That flips Family E(b) live.

**Architecture:**
- **`cryptoval/utxo/bigquery.py`** holds the SQL and the only module that imports
  `google.cloud.bigquery`.
- **`cryptoval/utxo/extract.py`** stages a spend-date range behind the existing `ByteBudgetGate`
  and writes three bronze tables per run. It also writes a fourth with BTC's full Coin Metrics
  history.
- **`cryptoval/utxo/derive.py`** turns all complete runs into one silver table,
  `btc_cointime_daily`.
- **dbt** exposes that table as `fct_btc_cointime_daily`. `_widen_panel` joins it in, and six
  methods in `cryptoval/methods/cointime.py` read it.

**Tech Stack:** Python 3.14, pandas, numpy, google-cloud-bigquery 3.46.2, db-dtypes 1.8.0, DuckDB/dbt.

**Spec:** `docs/superpowers/specs/2026-10-18-utxo-extraction-design.md`. The parent spec is
`2026-10-11-cryptoasset-valuation-pipeline-design.md` §6.3, §10.6(b).

## Global Constraints

- Free only. Every BigQuery query dry-runs first and goes through `ByteBudgetGate.guard`
  (0.9 TB/month). A refusal raises `BudgetExceeded` before anything executes.
- The project is `settings.bigquery_project` (`cryptoval-sandbox-3141`), never the gcloud default.
  If it is empty, the run refuses.
- The REST download path is `create_bqstorage_client=False`; the Storage Read API is never used.
- `make utxo` (extraction) is NOT in `make all`; `make utxo-derive` is.
- Cumulative from genesis: derive refuses unless the complete runs cover `[2009-01-03, end]` with
  no gap and no overlap.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Test command: `.venv/bin/python -m pytest -q`. Lint: `.venv/bin/ruff check cryptoval tests`.

## Review Focus

1. **A crash mid-run** (spent written, created not) leaves a partial run. It must be ignored, and
   the next run must redo the same range (Task 2 test
   `test_a_partial_run_is_ignored_and_its_range_redone`).
2. **The three dry runs together** exceed the month's remaining budget: nothing runs, so no query
   is half-spent (Task 2, `test_the_gate_sees_the_total_of_all_three_queries`).
3. **A re-run when already up to date** issues no query at all (Task 2,
   `test_up_to_date_issues_no_query`).
4. **A price gap after the first price**, which would silently value coins at 0, must raise
   (Task 3, `test_a_price_gap_after_the_first_price_raises`).
5. **Unspent value going negative**, a sign of a double-counted spend, must raise rather than
   publish (Task 3, `test_negative_unspent_raises`).

---

### Task 1: Settings, dependency, SQL and the BigQuery client

**Files:**
- Modify: `requirements.txt`, `cryptoval/config.py` (Settings), `config/settings.yml`
- Create: `cryptoval/utxo/__init__.py`, `cryptoval/utxo/bigquery.py`
- Test: `tests/test_utxo_bigquery.py`

**Interfaces:**
- Produces:
  - `SQL_BUILDERS: dict[str, Callable[[dt.date, dt.date], str]]`, keyed `btc_utxo_spent`,
    `btc_utxo_created`, `btc_blocks`. Both dates are inclusive.
  - `BigQueryClient(project: str)` with `.dry_run(sql) -> int` and
    `.run(sql, max_bytes: int) -> tuple[pd.DataFrame, int]`.
  - `Settings.bigquery_project: str = ""`.

- [ ] **Step 1: Install dependencies**

Append to `requirements.txt`:
```
google-cloud-bigquery==3.46.2
db-dtypes==1.8.0
```
Run: `.venv/bin/pip install -q google-cloud-bigquery==3.46.2 db-dtypes==1.8.0`
Expected: exit 0.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_utxo_bigquery.py
import datetime as dt

import pytest

from cryptoval.config import load_config
from cryptoval.utxo.bigquery import SQL_BUILDERS, BigQueryClient

START, END = dt.date(2026, 9, 1), dt.date(2026, 9, 30)


def test_every_query_is_bounded_to_the_range_end_exclusive():
    for sql in (b(START, END) for b in SQL_BUILDERS.values()):
        assert "'2026-09-01'" in sql and "'2026-10-01'" in sql


def test_spent_joins_inputs_to_deduplicated_transactions():
    sql = SQL_BUILDERS["btc_utxo_spent"](START, END)
    assert "crypto_bitcoin.inputs" in sql and "crypto_bitcoin.transactions" in sql
    assert "crypto_bitcoin.outputs" not in sql  # the 37% dearer join (spec Ruling 1)
    assert "GROUP BY `hash`" in sql  # BIP30 duplicate hashes collapse to one row
    for col in ("created_date", "spent_date", "value_sat", "coinblocks_destroyed", "n_inputs"):
        assert col in sql


def test_no_query_selects_a_wide_column():
    for sql in (b(START, END) for b in SQL_BUILDERS.values()):
        for wide in ("script_hex", "script_asm", "addresses", "SELECT *"):
            assert wide not in sql


def test_an_unconfigured_project_is_refused_before_any_client_is_built():
    with pytest.raises(ValueError, match="bigquery_project"):
        BigQueryClient("")


def test_settings_name_the_sandbox_project():
    settings, _ = load_config()
    assert settings.bigquery_project == "cryptoval-sandbox-3141"
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_utxo_bigquery.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'cryptoval.utxo'`.

- [ ] **Step 4: Implement**

In `cryptoval/config.py` `Settings`, after `bigquery_monthly_byte_budget`:
```python
    bigquery_project: str = ""
```
In `config/settings.yml`, after the budget line:
```yaml
bigquery_project: cryptoval-sandbox-3141   # no billing account; never the gcloud default project
```
`cryptoval/utxo/__init__.py`:
```python
"""Bitcoin coin-age extraction from the BigQuery sandbox (spec 6.3, Plan 3b)."""
```
`cryptoval/utxo/bigquery.py`:
```python
"""The three extraction queries and the only module that talks to BigQuery (spec 6.3).

Columns are pruned to the narrow fields each query needs and every query carries a
block_timestamp range, which prunes partitions through the public views (318 GB -> 5 GB for
one month of inputs, measured 2026-10-18).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import pandas as pd

DATASET = "bigquery-public-data.crypto_bitcoin"


def _bounds(start: dt.date, end: dt.date) -> tuple[str, str]:
    """Inclusive dates to a half-open timestamp interval."""
    return start.isoformat(), (end + dt.timedelta(days=1)).isoformat()


def spent_sql(start: dt.date, end: dt.date) -> str:
    lo, hi = _bounds(start, end)
    # Through transactions, not outputs: inputs carry the spent value themselves, so only
    # the creating block is needed, and this join bills 466 GB against 740 GB (Ruling 1).
    # GROUP BY hash collapses BIP30's two duplicated coinbase hashes to the later block.
    return f"""
WITH tx AS (
  SELECT `hash`, MAX(block_number) AS block_number, MAX(block_timestamp) AS block_timestamp
  FROM `{DATASET}.transactions`
  WHERE block_timestamp < TIMESTAMP('{hi}')
  GROUP BY `hash`
)
SELECT
  DATE(tx.block_timestamp) AS created_date,
  DATE(i.block_timestamp) AS spent_date,
  CAST(SUM(i.value) AS INT64) AS value_sat,
  CAST(SUM(i.value * (i.block_number - tx.block_number)) / 1e8 AS FLOAT64) AS coinblocks_destroyed,
  COUNT(*) AS n_inputs
FROM `{DATASET}.inputs` AS i
JOIN tx ON tx.`hash` = i.spent_transaction_hash
WHERE i.block_timestamp >= TIMESTAMP('{lo}') AND i.block_timestamp < TIMESTAMP('{hi}')
GROUP BY created_date, spent_date
"""


def created_sql(start: dt.date, end: dt.date) -> str:
    lo, hi = _bounds(start, end)
    return f"""
SELECT
  DATE(block_timestamp) AS created_date,
  CAST(SUM(value) AS INT64) AS value_sat,
  COUNT(*) AS n_outputs
FROM `{DATASET}.outputs`
WHERE block_timestamp >= TIMESTAMP('{lo}') AND block_timestamp < TIMESTAMP('{hi}')
GROUP BY created_date
"""


def blocks_sql(start: dt.date, end: dt.date) -> str:
    lo, hi = _bounds(start, end)
    # Per block, not per day: block timestamps are not monotonic in height, so a day's heights
    # are not a contiguous run and a (min, max) pair would misstate its coinblocks created.
    return f"""
SELECT number AS height, DATE(timestamp) AS block_date
FROM `{DATASET}.blocks`
WHERE timestamp >= TIMESTAMP('{lo}') AND timestamp < TIMESTAMP('{hi}')
"""


SQL_BUILDERS: dict[str, Callable[[dt.date, dt.date], str]] = {
    "btc_utxo_spent": spent_sql,
    "btc_utxo_created": created_sql,
    "btc_blocks": blocks_sql,
}


class BigQueryClient:
    """Dry-run and run against one named project, never the gcloud default."""

    def __init__(self, project: str) -> None:
        if not project or not project.strip():
            raise ValueError(
                "bigquery_project is not configured in config/settings.yml; refusing to fall "
                "back to the gcloud default project, which may carry a billing account"
            )
        from google.cloud import bigquery

        self._bigquery = bigquery
        self._client = bigquery.Client(project=project)

    def dry_run(self, sql: str) -> int:
        config = self._bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        return int(self._client.query(sql, job_config=config).total_bytes_processed or 0)

    def run(self, sql: str, max_bytes: int) -> tuple[pd.DataFrame, int]:
        # maximum_bytes_billed is BigQuery's own refusal, a third guard behind the sandbox
        # and our gate: a query whose real scan exceeds the dry run fails instead of billing.
        config = self._bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes)
        job = self._client.query(sql, job_config=config)
        frame = job.result().to_dataframe(create_bqstorage_client=False)
        return frame, int(job.total_bytes_billed or 0)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_utxo_bigquery.py tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt cryptoval/config.py config/settings.yml cryptoval/utxo tests/test_utxo_bigquery.py
git commit -m "feat: BigQuery client and the three coin-age extraction queries"
```

---

### Task 2: The staged extractor

**Files:**
- Create: `cryptoval/utxo/extract.py`
- Test: `tests/test_utxo_extract.py`

**Interfaces:**
- Consumes: `SQL_BUILDERS` (Task 1); `ByteBudgetGate`, `BudgetExceeded` (`cryptoval.budget`);
  `RunManifest` (`cryptoval.manifest`); `write_bronze` (`cryptoval.io.bronze`).
- Produces:
  - Constants: `GENESIS = dt.date(2009, 1, 3)`,
    `RANGE_SOURCES = ("btc_utxo_spent", "btc_utxo_created", "btc_blocks")`,
    `HISTORY_SOURCE = "btc_coinmetrics_history"`.
  - `Range(start, end)`, a frozen dataclass with inclusive ends.
  - `completed_runs(resolver) -> list[tuple[Range, str]]`, sorted by start, raising on overlap.
  - `assert_contiguous(runs) -> dt.date`, which returns the last end and raises unless the runs
    start at GENESIS with no gap.
  - `next_range(resolver, today) -> Range | None`.
  - `extract(resolver, client, gate, manifest, today, now, history_fetch, dry_run=False) -> dict`.
  - `read_runs(resolver, runs) -> dict[str, pd.DataFrame]`.
  - `read_history(resolver) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_utxo_extract.py
import datetime as dt

import pandas as pd
import pytest

from cryptoval.budget import BudgetExceeded, ByteBudgetGate
from cryptoval.io.bronze import write_bronze
from cryptoval.manifest import RunManifest
from cryptoval.paths import PathResolver
from cryptoval.utxo.extract import (
    GENESIS,
    HISTORY_SOURCE,
    Range,
    assert_contiguous,
    completed_runs,
    extract,
    next_range,
    read_history,
)

TODAY = dt.date(2009, 1, 10)
NOW = dt.datetime(2009, 1, 10, 12, tzinfo=dt.UTC)

FRAMES = {
    "inputs": pd.DataFrame(
        {"created_date": [GENESIS], "spent_date": [dt.date(2009, 1, 5)], "value_sat": [10],
         "coinblocks_destroyed": [1.0], "n_inputs": [1]}
    ),
    "outputs": pd.DataFrame({"created_date": [GENESIS], "value_sat": [5_000_000_000], "n_outputs": [1]}),
    "blocks": pd.DataFrame({"height": [0], "block_date": [GENESIS]}),
}


class FakeClient:
    def __init__(self, estimate=100, fail_on=None):
        self.estimate, self.fail_on = estimate, fail_on
        self.dry, self.ran = [], []

    def dry_run(self, sql):
        self.dry.append(sql)
        return self.estimate

    def run(self, sql, max_bytes):
        key = next(k for k in FRAMES if f"crypto_bitcoin.{k}" in sql)
        if key == self.fail_on:
            raise RuntimeError("boom")
        self.ran.append((key, max_bytes))
        return FRAMES[key].copy(), self.estimate


def _history(start, end):
    return pd.DataFrame({"asset": ["btc"], "metric_date": [pd.Timestamp(end, tz="UTC")],
                         "PriceUSD": [1.0], "SplyCur": [50.0], "realized_cap": [1.0]})


def _run(tmp_path, client, budget=10_000, today=TODAY, dry_run=False):
    resolver = PathResolver(str(tmp_path))
    manifest = RunManifest(resolver)
    gate = ByteBudgetGate(manifest, budget)
    return resolver, manifest, extract(resolver, client, gate, manifest, today, NOW, _history,
                                       dry_run=dry_run)


def test_the_first_run_covers_genesis_to_two_days_ago(tmp_path):
    resolver, manifest, report = _run(tmp_path, FakeClient())
    assert report["status"] == "extracted"
    [(rng, _)] = completed_runs(resolver)
    assert rng == Range(GENESIS, dt.date(2009, 1, 8))
    ledger = manifest.read()
    assert ledger["bytes_scanned"].sum() == 300  # three queries, actual billed bytes recorded
    assert not read_history(resolver).empty


def test_the_next_run_starts_the_day_after_the_last_end(tmp_path):
    _run(tmp_path, FakeClient())
    resolver, _, _ = _run(tmp_path, FakeClient(), today=dt.date(2009, 1, 15))
    ranges = [r for r, _ in completed_runs(resolver)]
    assert ranges[1] == Range(dt.date(2009, 1, 9), dt.date(2009, 1, 13))
    assert assert_contiguous(completed_runs(resolver)) == dt.date(2009, 1, 13)


def test_up_to_date_issues_no_query(tmp_path):
    _run(tmp_path, FakeClient())
    client = FakeClient()
    _, _, report = _run(tmp_path, client)
    assert report["status"] == "up_to_date" and client.dry == [] and client.ran == []


def test_the_gate_sees_the_total_of_all_three_queries(tmp_path):
    client = FakeClient(estimate=400)  # each fits in 1000, the three together do not
    with pytest.raises(BudgetExceeded):
        _run(tmp_path, client, budget=1000)
    assert client.ran == []


def test_dry_run_prices_the_range_and_runs_nothing(tmp_path):
    client = FakeClient()
    resolver, _, report = _run(tmp_path, client, dry_run=True)
    assert report["status"] == "dry_run" and report["estimated_bytes"] == 300
    assert client.ran == [] and completed_runs(resolver) == []


def test_a_partial_run_is_ignored_and_its_range_redone(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        _run(tmp_path, FakeClient(fail_on="outputs"))
    resolver = PathResolver(str(tmp_path))
    assert completed_runs(resolver) == []
    assert next_range(resolver, TODAY) == Range(GENESIS, dt.date(2009, 1, 8))
    assert RunManifest(resolver).read()["status"].tolist() == ["success", "failed"]


def test_the_query_cap_is_what_remains_of_the_month(tmp_path):
    client = FakeClient()
    _run(tmp_path, client, budget=10_000)
    assert [cap for _, cap in client.ran] == [10_000, 9_900, 9_800]


def test_overlapping_runs_are_refused(tmp_path):
    resolver = PathResolver(str(tmp_path))
    for run_id, (a, b) in {"r1": (GENESIS, dt.date(2009, 1, 8)),
                           "r2": (dt.date(2009, 1, 5), dt.date(2009, 1, 9))}.items():
        for source, key in (("btc_utxo_spent", "inputs"), ("btc_utxo_created", "outputs"),
                            ("btc_blocks", "blocks")):
            frame = FRAMES[key].assign(range_start=a, range_end=b)
            write_bronze(resolver, source, b, frame, {}, run_id)
    with pytest.raises(ValueError, match="overlap"):
        completed_runs(resolver)


def test_a_gap_is_not_contiguous():
    runs = [(Range(GENESIS, dt.date(2009, 1, 8)), "a"), (Range(dt.date(2009, 1, 10), dt.date(2009, 1, 12)), "b")]
    with pytest.raises(ValueError, match="gap"):
        assert_contiguous(runs)
    with pytest.raises(ValueError, match="genesis"):
        assert_contiguous([(Range(dt.date(2009, 1, 4), dt.date(2009, 1, 8)), "a")])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_utxo_extract.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'cryptoval.utxo.extract'`.

- [ ] **Step 3: Implement `cryptoval/utxo/extract.py`**

```python
"""Staged extraction of Bitcoin's coin-age matrix (spec 6.3, Plan 3b).

A run covers one spend-date range and lands three bronze tables under ONE run_id. A range
counts as done only when all three exist for that run_id, so a run that dies half-way is
ignored and its range is redone; bronze is append-only, so the debris stays as a record.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from cryptoval.budget import ByteBudgetGate
from cryptoval.hashing import stable_hash
from cryptoval.io.bronze import write_bronze
from cryptoval.manifest import RunManifest
from cryptoval.paths import PathResolver
from cryptoval.utxo.bigquery import SQL_BUILDERS

GENESIS = dt.date(2009, 1, 3)
RANGE_SOURCES = ("btc_utxo_spent", "btc_utxo_created", "btc_blocks")
HISTORY_SOURCE = "btc_coinmetrics_history"
# The dataset follows the tip, and a just-mined day can still reorg.
LAG_DAYS = 2


@dataclass(frozen=True)
class Range:
    start: dt.date
    end: dt.date


def _partitions(resolver: PathResolver, source: str) -> dict[str, pathlib.Path]:
    root = pathlib.Path(resolver.layer("bronze")) / f"source={source}"
    return {
        p.parent.name.removeprefix("run_id="): p
        for p in root.glob("ingest_date=*/run_id=*/data.parquet")
    }


def completed_runs(resolver: PathResolver) -> list[tuple[Range, str]]:
    parts = {s: _partitions(resolver, s) for s in RANGE_SOURCES}
    complete = set.intersection(*(set(p) for p in parts.values()))
    runs = []
    for run_id in complete:
        bounds = pd.read_parquet(parts[RANGE_SOURCES[0]][run_id], columns=["range_start", "range_end"])
        runs.append((Range(bounds["range_start"].iloc[0], bounds["range_end"].iloc[0]), run_id))
    runs.sort(key=lambda r: r[0].start)
    for (prev, a), (cur, b) in zip(runs, runs[1:]):
        if cur.start <= prev.end:
            raise ValueError(f"extracted ranges overlap: {prev} (run {a}) and {cur} (run {b})")
    return runs


def assert_contiguous(runs: list[tuple[Range, str]]) -> dt.date:
    """The last day covered, when the runs tile [GENESIS, end] exactly; otherwise raise.

    Cointime is cumulative from genesis, so a hole yields wrong values, not partial ones.
    """
    if not runs:
        raise ValueError("no complete extraction runs")
    if runs[0][0].start != GENESIS:
        raise ValueError(f"extraction starts {runs[0][0].start}, not at genesis {GENESIS}")
    for (prev, _), (cur, _) in zip(runs, runs[1:]):
        if cur.start != prev.end + dt.timedelta(days=1):
            raise ValueError(f"gap between {prev.end} and {cur.start} in the extracted ranges")
    return runs[-1][0].end


def next_range(resolver: PathResolver, today: dt.date) -> Range | None:
    runs = completed_runs(resolver)
    start = assert_contiguous(runs) + dt.timedelta(days=1) if runs else GENESIS
    end = today - dt.timedelta(days=LAG_DAYS)
    return Range(start, end) if start <= end else None


def read_runs(resolver: PathResolver, runs: list[tuple[Range, str]]) -> dict[str, pd.DataFrame]:
    out = {}
    for source in RANGE_SOURCES:
        parts = _partitions(resolver, source)
        out[source] = pd.concat([pd.read_parquet(parts[r]) for _, r in runs], ignore_index=True)
    return out


def read_history(resolver: PathResolver) -> pd.DataFrame:
    """The newest BTC Coin Metrics history, or an empty frame."""
    parts = sorted(_partitions(resolver, HISTORY_SOURCE).values(), key=lambda p: p.stat().st_mtime)
    return pd.read_parquet(parts[-1]) if parts else pd.DataFrame()


def extract(
    resolver: PathResolver,
    client,
    gate: ByteBudgetGate,
    manifest: RunManifest,
    today: dt.date,
    now: dt.datetime,
    history_fetch: Callable[[dt.date, dt.date], pd.DataFrame],
    dry_run: bool = False,
) -> dict:
    rng = next_range(resolver, today)
    if rng is None:
        return {"status": "up_to_date", "through": str(today - dt.timedelta(days=LAG_DAYS))}
    sqls = {s: SQL_BUILDERS[s](rng.start, rng.end) for s in RANGE_SOURCES}
    estimates = {s: client.dry_run(sql) for s, sql in sqls.items()}
    total = sum(estimates.values())
    report = {"range": f"{rng.start}..{rng.end}", "estimated_bytes": total,
              "per_query": estimates}
    if dry_run:
        return {"status": "dry_run", **report}
    # The whole range at once: three queries that each fit could still overrun together,
    # and a refusal after the first one ran would have spent bytes on nothing usable.
    gate.guard(total, now)

    run_id = str(uuid.uuid4())
    for source, sql in sqls.items():
        params = {"range_start": rng.start, "range_end": rng.end, "sql_hash": stable_hash(sql)}
        record = manifest.start(source, params, started_at=now)
        billed = 0
        try:
            frame, billed = client.run(sql, max_bytes=gate.remaining(now))
            frame = frame.assign(range_start=rng.start, range_end=rng.end)
            write_bronze(resolver, source, rng.end, frame, params, run_id)
        except Exception as exc:
            manifest.finish(record, 0, 0, "failed", bytes_scanned=billed, error=str(exc))
            raise
        manifest.finish(record, len(frame), len(frame), "success", bytes_scanned=billed)

    # Free (HTTP, not BigQuery): realized-cap weighting needs prices from 2010, and the panel
    # starts in 2015.
    history = history_fetch(GENESIS, rng.end)
    write_bronze(resolver, HISTORY_SOURCE, rng.end, history, {"end": rng.end}, run_id)
    return {"status": "extracted", "run_id": run_id, **report}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_utxo_extract.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add cryptoval/utxo/extract.py tests/test_utxo_extract.py
git commit -m "feat: staged coin-age extraction behind the byte gate, resumable by run"
```

---

### Task 3: Derivation of the daily coin-age table

**Files:**
- Create: `cryptoval/utxo/derive.py`
- Test: `tests/test_utxo_derive.py`

**Interfaces:**
- Consumes: the frames from `read_runs` (Task 2); `GENESIS`.
- Produces:
  - `BANDS`, a tuple of `(name, lo_days, hi_days | None)`.
  - `COINTIME_COLUMNS: list[str]`.
  - `empty_cointime() -> pd.DataFrame`.
  - `supply_before(heights: np.ndarray) -> np.ndarray`, in sat.
  - `derive(spent, created, blocks, price: pd.Series, end: dt.date, start: dt.date = GENESIS) -> pd.DataFrame`.
    `price` is indexed by `datetime.date`, in USD.
  - `vendor_mismatches(cointime, vendor) -> list[str]`, where `vendor` has
    `metric_date, supply, realized_cap`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_utxo_derive.py
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.utxo.derive import COINTIME_COLUMNS, derive, empty_cointime, supply_before, vendor_mismatches

D0 = dt.date(2009, 1, 3)
DAYS = [D0 + dt.timedelta(days=i) for i in range(4)]
SAT = 100_000_000


def _toy(price=(np.nan, 10.0, 20.0, 40.0), spent_c1=100):
    blocks = pd.DataFrame({"height": [0, 1, 2, 3, 4], "block_date": [DAYS[0], DAYS[1], DAYS[1], DAYS[2], DAYS[3]]})
    created = pd.DataFrame({"created_date": DAYS, "value_sat": [50 * SAT, 100 * SAT, 50 * SAT, 50 * SAT]})
    spent = pd.DataFrame({
        "created_date": [DAYS[0], DAYS[1]],
        "spent_date": [DAYS[2], DAYS[3]],
        "value_sat": [20 * SAT, spent_c1 * SAT],
        "coinblocks_destroyed": [60.0, 250.0],
    })
    return derive(spent, created, blocks, pd.Series(price, index=DAYS), DAYS[-1], start=D0).set_index("metric_date")


def test_subsidy_schedule_halves_at_210000():
    got = supply_before(np.array([0, 1, 210_000, 210_001]))
    assert got.tolist() == [0, 50 * SAT, 210_000 * 50 * SAT, 210_000 * 50 * SAT + 25 * SAT]


def test_cointime_primitives_on_a_toy_chain():
    t = _toy()
    assert t["coinblocks_created"].tolist() == [0.0, 150.0, 150.0, 200.0]
    assert t["coinblocks_destroyed"].tolist() == [0.0, 0.0, 60.0, 250.0]
    assert np.isnan(t["liveliness"].iloc[0])
    assert t["liveliness"].iloc[1:].tolist() == pytest.approx([0.0, 0.2, 0.62])
    assert (t["liveliness"] + t["vaultedness"]).iloc[1:].tolist() == pytest.approx([1, 1, 1])
    assert t["utxo_supply"].tolist() == [50.0, 150.0, 180.0, 130.0]
    assert t["active_supply"].iloc[-1] == pytest.approx(130 * 0.62)


def test_realized_cap_values_each_coin_at_its_creation_price_and_pre_price_coins_at_zero():
    t = _toy()
    assert t["utxo_realized_cap"].tolist() == [0.0, 1000.0, 2000.0, 3000.0]


def test_hodl_waves_by_supply_and_by_realized_cap():
    last = _toy().iloc[-1]
    assert last["hodl_0d"] == pytest.approx(50 / 130)
    assert last["hodl_1d_1w"] == pytest.approx(80 / 130)
    assert last["hodl_rc_0d"] == pytest.approx(2000 / 3000)
    assert last["hodl_rc_1d_1w"] == pytest.approx(1000 / 3000)
    assert last["hodl_6m_1y"] == 0.0
    bands = [c for c in COINTIME_COLUMNS if c.startswith("hodl_") and not c.startswith("hodl_rc_")]
    assert last[bands].sum() == pytest.approx(1.0)


def test_short_term_holders_and_realized_profit():
    t = _toy()
    assert t["sth_supply"].iloc[-1] == 130.0 and t["sth_realized_cap"].iloc[-1] == 3000.0
    # day 2 spends 20 BTC bought at 0 for 20 each; day 3 spends 100 bought at 10 for 40 each
    assert t["realized_profit_usd"].tolist() == [0.0, 0.0, 400.0, 3000.0]


def test_negative_unspent_raises():
    with pytest.raises(ValueError, match="negative unspent"):
        _toy(spent_c1=101)


def test_a_price_gap_after_the_first_price_raises():
    with pytest.raises(ValueError, match="price"):
        _toy(price=(np.nan, 10.0, np.nan, 40.0))


def test_empty_cointime_has_every_column():
    assert list(empty_cointime().columns) == COINTIME_COLUMNS and empty_cointime().empty


def test_vendor_mismatches_name_the_breach():
    days = pd.to_datetime(["2012-01-01", "2012-01-02"]).date
    ours = pd.DataFrame({"metric_date": days, "utxo_supply": [100.0, 100.0], "utxo_realized_cap": [10.0, 10.0]})
    vendor = pd.DataFrame({"metric_date": days, "supply": [100.0, 101.0], "realized_cap": [10.0, 10.2]})
    found = vendor_mismatches(ours, vendor)
    assert len(found) == 1 and "utxo_supply" in found[0] and "1 day" in found[0]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_utxo_derive.py -q`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `cryptoval/utxo/derive.py`**

```python
"""From the creation x spend matrix to one row per day (spec 6.3, 10.6(b); Plan 3b section 5).

unspent(c, d) = created(c) - sum_{s <= d} spent(c, s): nothing about unspent outputs is ever
queried. The loop walks spend days once with a running vector, so no N x N matrix is built.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cryptoval.utxo.extract import GENESIS

SAT = 100_000_000
HALVING = 210_000
STH_DAYS = 155
BANDS = (
    ("0d", 0, 1), ("1d_1w", 1, 7), ("1w_1m", 7, 30), ("1m_3m", 30, 90), ("3m_6m", 90, 180),
    ("6m_1y", 180, 365), ("1y_2y", 365, 730), ("2y_3y", 730, 1095), ("3y_5y", 1095, 1825),
    ("5y_7y", 1825, 2555), ("7y_10y", 2555, 3650), ("10y_plus", 3650, None),
)
COINTIME_COLUMNS = [
    "metric_date", "coinblocks_created", "coinblocks_destroyed", "liveliness", "vaultedness",
    "utxo_supply", "active_supply", "vaulted_supply", "utxo_realized_cap",
    *(f"hodl_{b}" for b, _, _ in BANDS), *(f"hodl_rc_{b}" for b, _, _ in BANDS),
    "sth_supply", "sth_realized_cap", "realized_profit_usd",
]
SUPPLY_RTOL, SUPPLY_FROM = 0.005, dt.date(2011, 1, 1)
REALIZED_RTOL, REALIZED_FROM = 0.05, dt.date(2012, 1, 1)


def empty_cointime() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="float64") for c in COINTIME_COLUMNS})
    frame["metric_date"] = pd.Series(dtype="datetime64[ns]")
    return frame


def supply_before(heights: np.ndarray) -> np.ndarray:
    """Sat issued in blocks 0..h-1: the coins that exist when block h is mined."""
    subsidy = np.array([(50 * SAT) >> e for e in range(64)] + [0], dtype=np.int64)
    era_total = np.concatenate([[0], np.cumsum(subsidy * HALVING)])
    h = np.asarray(heights, dtype=np.int64)
    era = np.minimum(h // HALVING, 64)
    return era_total[era] + (h - era * HALVING) * subsidy[era]


def _prices(price: pd.Series, days: list[dt.date]) -> np.ndarray:
    p = price.reindex(days).to_numpy(dtype=float)
    valid = np.flatnonzero(~np.isnan(p))
    if valid.size == 0:
        raise ValueError("no BTC price in the derivation span")
    gaps = np.isnan(p[valid[0]:])
    if gaps.any():
        first = days[valid[0] + int(np.argmax(gaps))]
        raise ValueError(f"BTC price missing on {first}, after the first price; refusing to value coins at 0")
    p[: valid[0]] = 0.0  # pre-price coins: the realized-cap convention
    return p


def derive(
    spent: pd.DataFrame,
    created: pd.DataFrame,
    blocks: pd.DataFrame,
    price: pd.Series,
    end: dt.date,
    start: dt.date = GENESIS,
) -> pd.DataFrame:
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    n = len(days)
    index = {d: i for i, d in enumerate(days)}

    def pos(series: pd.Series) -> np.ndarray:
        return series.map(index).to_numpy(dtype=np.int64)

    created_sat = np.bincount(pos(created["created_date"]), weights=created["value_sat"].astype(float), minlength=n)
    cb_created = np.bincount(
        pos(blocks["block_date"]), weights=supply_before(blocks["height"].to_numpy()) / SAT, minlength=n
    )
    ci, si = pos(spent["created_date"]), pos(spent["spent_date"])
    value = spent["value_sat"].to_numpy(dtype=float)
    cb_destroyed = np.bincount(si, weights=spent["coinblocks_destroyed"].to_numpy(dtype=float), minlength=n)
    p = _prices(price, days)
    profit = np.bincount(si, weights=value / SAT * np.maximum(p[si] - p[ci], 0.0), minlength=n)

    order = np.argsort(si, kind="stable")
    ci, si, value = ci[order], si[order], value[order]
    bounds = np.searchsorted(si, np.arange(n + 1))
    edges = np.array([lo for _, lo, _ in BANDS])
    cum_spent = np.zeros(n)
    rows = np.zeros((n, len(COINTIME_COLUMNS) - 1))
    col = {c: i - 1 for i, c in enumerate(COINTIME_COLUMNS)}
    for d in range(n):
        a, b = bounds[d], bounds[d + 1]
        np.add.at(cum_spent, ci[a:b], value[a:b])
        alive_sat = created_sat[: d + 1] - cum_spent[: d + 1]
        if alive_sat.min() < 0:
            c = int(np.argmin(alive_sat))
            raise ValueError(f"negative unspent value for coins created {days[c]} as of {days[d]}")
        by_age = (alive_sat / SAT)[::-1]  # index = age in days
        rc_by_age = by_age * p[: d + 1][::-1]
        starts = edges[edges <= d]
        bands, rc_bands = np.zeros(len(BANDS)), np.zeros(len(BANDS))
        bands[: len(starts)] = np.add.reduceat(by_age, starts)
        rc_bands[: len(starts)] = np.add.reduceat(rc_by_age, starts)
        supply, realized = by_age.sum(), rc_by_age.sum()
        r = rows[d]
        r[col["utxo_supply"]], r[col["utxo_realized_cap"]] = supply, realized
        for k, (name, _, _) in enumerate(BANDS):
            r[col[f"hodl_{name}"]] = bands[k] / supply if supply > 0 else np.nan
            r[col[f"hodl_rc_{name}"]] = rc_bands[k] / realized if realized > 0 else np.nan
        r[col["sth_supply"]] = by_age[:STH_DAYS].sum()
        r[col["sth_realized_cap"]] = rc_by_age[:STH_DAYS].sum()

    out = pd.DataFrame(rows, columns=COINTIME_COLUMNS[1:])
    out.insert(0, "metric_date", days)
    out["coinblocks_created"], out["coinblocks_destroyed"] = cb_created, cb_destroyed
    cum_created = np.cumsum(cb_created)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["liveliness"] = np.where(cum_created > 0, np.cumsum(cb_destroyed) / cum_created, np.nan)
    out["vaultedness"] = 1.0 - out["liveliness"]
    out["active_supply"] = out["utxo_supply"] * out["liveliness"]
    out["vaulted_supply"] = out["utxo_supply"] * out["vaultedness"]
    out["realized_profit_usd"] = profit
    return out[COINTIME_COLUMNS]


def vendor_mismatches(cointime: pd.DataFrame, vendor: pd.DataFrame) -> list[str]:
    """Our supply and realized cap against Coin Metrics', which computes both from the same
    chain independently: the correctness proof of the whole extraction (spec section 5)."""
    both = cointime.merge(vendor, on="metric_date", how="inner")
    out = []
    for ours, theirs, rtol, since in (
        ("utxo_supply", "supply", SUPPLY_RTOL, SUPPLY_FROM),
        ("utxo_realized_cap", "realized_cap", REALIZED_RTOL, REALIZED_FROM),
    ):
        scoped = both[pd.to_datetime(both["metric_date"]).dt.date >= since].dropna(subset=[ours, theirs])
        rel = (scoped[ours] - scoped[theirs]).abs() / scoped[theirs].abs()
        bad = rel[rel > rtol]
        if len(bad):
            worst = scoped.loc[bad.idxmax(), "metric_date"]
            out.append(
                f"btc_cointime_daily/{ours}: {len(bad)} day(s) since {since} differ from Coin "
                f"Metrics {theirs} by more than {rtol:.1%} (worst {bad.max():.2%} on {worst})"
            )
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_utxo_derive.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add cryptoval/utxo/derive.py tests/test_utxo_derive.py
git commit -m "feat: derive Cointime, HODL waves and realized profit from the coin-age matrix"
```

---

### Task 4: CLI, Makefile and validation

**Files:**
- Modify: `cryptoval/cli.py` (new `run_utxo_extract`, `run_utxo_derive`, `_cointime_failures`;
  the `utxo` subparser; a hook in `run_validate`), `Makefile`
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces:
  - `run_utxo_derive(settings, assets) -> dict`, which writes silver `btc_cointime_daily`.
  - `run_utxo_extract(settings, assets, dry_run) -> dict`.
  - `_cointime_failures(settings) -> list[str]`.
  - CLI `cryptoval utxo extract [--dry-run]` and `cryptoval utxo derive`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cli.py`)

```python
def test_utxo_derive_without_a_backfill_writes_an_empty_table(tmp_path):
    from cryptoval.cli import run_utxo_derive
    from cryptoval.config import Settings
    from cryptoval.utxo.derive import COINTIME_COLUMNS

    settings = Settings(data_root=str(tmp_path))
    report = run_utxo_derive(settings, [])
    assert report["status"] == "awaiting_backfill"
    table = pd.read_parquet(tmp_path / "silver" / "btc_cointime_daily" / "part-0.parquet")
    assert table.empty and list(table.columns) == COINTIME_COLUMNS


def test_cointime_failures_flag_the_complement_and_vendor_breaches(tmp_path, monkeypatch):
    import datetime as dt

    import cryptoval.cli as cli
    from cryptoval.config import Settings
    from cryptoval.io.bronze import write_silver

    settings = Settings(data_root=str(tmp_path))
    day = dt.date(2013, 1, 1)
    write_silver(settings.resolver, "btc_cointime_daily", pd.DataFrame({
        "metric_date": [day], "liveliness": [0.4], "vaultedness": [0.5],
        "utxo_supply": [100.0], "utxo_realized_cap": [10.0]}))
    monkeypatch.setattr(cli, "read_history", lambda resolver: pd.DataFrame({
        "asset": ["btc"], "metric_date": [pd.Timestamp(day, tz="UTC")], "PriceUSD": [1.0],
        "SplyCur": [100.0], "realized_cap": [20.0]}))
    found = cli._cointime_failures(settings)
    assert any("identity_cointime" in f for f in found)
    assert any("utxo_realized_cap" in f for f in found)


def test_cointime_failures_are_silent_before_any_backfill(tmp_path):
    from cryptoval.cli import _cointime_failures
    from cryptoval.config import Settings

    assert _cointime_failures(Settings(data_root=str(tmp_path))) == []


def test_utxo_subcommand_parses(monkeypatch):
    import cryptoval.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "run_utxo_extract", lambda s, a, dry_run: seen.update(dry=dry_run) or {"status": "dry_run"})
    assert cli.main(["utxo", "extract", "--dry-run"]) == 0 and seen == {"dry": True}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q -k "utxo or cointime"`
Expected: FAIL, `ImportError: cannot import name 'run_utxo_derive'`.

- [ ] **Step 3: Implement in `cryptoval/cli.py`**

Imports, at the top with the others:
```python
from cryptoval.budget import ByteBudgetGate
from cryptoval.contracts.identities import check_cointime_complement
from cryptoval.manifest import RunManifest
from cryptoval.utxo.derive import derive, empty_cointime, vendor_mismatches
from cryptoval.utxo.extract import assert_contiguous, completed_runs, extract, read_history, read_runs
```
(Skip any of these that are already imported. `CoinMetricsSource` and `httpx` already are.)

Functions, placed after `run_vol`:
```python
def _btc_history(resolver) -> pd.DataFrame:
    """BTC's full Coin Metrics history as (metric_date, price_usd, supply, realized_cap)."""
    raw = read_history(resolver)
    if raw.empty:
        return pd.DataFrame(columns=["metric_date", "price_usd", "supply", "realized_cap"])
    btc = raw[raw["asset"] == "btc"]
    return pd.DataFrame({
        "metric_date": pd.to_datetime(btc["metric_date"], utc=True).dt.date,
        "price_usd": pd.to_numeric(btc["PriceUSD"], errors="coerce"),
        "supply": pd.to_numeric(btc["SplyCur"], errors="coerce"),
        "realized_cap": pd.to_numeric(btc["realized_cap"], errors="coerce"),
    })


def run_utxo_extract(settings: Settings, assets: list[Asset], dry_run: bool) -> dict:
    """One staged BigQuery extraction (spec 6.3). Spends budget; never part of `make all`."""
    from cryptoval.utxo.bigquery import BigQueryClient

    resolver = settings.resolver
    manifest = RunManifest(resolver)
    btc = [a for a in assets if a.symbol == "BTC"]

    def history(start: dt.date, end: dt.date) -> pd.DataFrame:
        with httpx.Client(timeout=settings.request_timeout_s) as client:
            return CoinMetricsSource(max_retries=settings.max_retries).fetch(start, end, btc, client).frame

    return extract(
        resolver,
        BigQueryClient(settings.bigquery_project),
        ByteBudgetGate(manifest, settings.bigquery_monthly_byte_budget),
        manifest,
        dt.datetime.now(dt.UTC).date(),
        dt.datetime.now(dt.UTC),
        history,
        dry_run=dry_run,
    )


def run_utxo_derive(settings: Settings, assets: list[Asset]) -> dict:
    """Silver btc_cointime_daily from every complete run; empty until the backfill is whole."""
    resolver = settings.resolver
    runs = completed_runs(resolver)
    if not runs:
        write_silver(resolver, "btc_cointime_daily", empty_cointime())
        return {"status": "awaiting_backfill"}
    end = assert_contiguous(runs)
    frames = read_runs(resolver, runs)
    history = _btc_history(resolver)
    table = derive(
        frames["btc_utxo_spent"], frames["btc_utxo_created"], frames["btc_blocks"],
        history.set_index("metric_date")["price_usd"], end,
    )
    write_silver(resolver, "btc_cointime_daily", table)
    last = table.iloc[-1]
    return {"status": "derived", "rows": len(table), "through": str(end),
            "liveliness": round(float(last["liveliness"]), 4),
            "utxo_supply": round(float(last["utxo_supply"]), 2)}


def _cointime_failures(settings: Settings) -> list[str]:
    cointime = _read_silver(settings, "btc_cointime_daily")
    if cointime.empty:
        return []  # awaiting backfill: the family refuses, there is nothing to check
    out = [] if (r := check_cointime_complement(cointime)).passed else [f"btc_cointime_daily/{r.name}: {r.detail}"]
    vendor = _btc_history(settings.resolver)
    cointime = cointime.assign(metric_date=pd.to_datetime(cointime["metric_date"]).dt.date)
    return out + vendor_mismatches(cointime, vendor)
```
In `run_validate`, before `return Validation(failures, warnings)`:
```python
    failures.extend(_cointime_failures(settings))
```
Subparser, after `sub.add_parser("vol")`:
```python
    ux = sub.add_parser("utxo")
    ux.add_argument("action", choices=["extract", "derive"])
    ux.add_argument("--dry-run", action="store_true", help="price the next range, run nothing")
```
Dispatch, before the `vol` branch:
```python
    if args.command == "utxo":
        if args.action == "extract":
            outcome = run_utxo_extract(settings, assets, args.dry_run)
        else:
            outcome = run_utxo_derive(settings, assets)
        print(outcome)
        return 0
```
`Makefile`: add `utxo utxo-derive` to `.PHONY`, and add these targets:
```make
# Spends BigQuery sandbox budget (spec 6.3): run on purpose, monthly, never from `all`.
utxo:
	$(PY) -m cryptoval.cli utxo extract

utxo-derive:
	$(PY) -m cryptoval.cli utxo derive
```
In `all`, after `$(MAKE) lint test ingest`, insert `$(MAKE) utxo-derive`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/cli.py Makefile tests/test_cli.py
git commit -m "feat: utxo extract/derive commands and Cointime checks in validate"
```

---

### Task 5: dbt mart and panel join

**Files:**
- Create: `dbt/models/staging/stg_btc_cointime.sql`, `dbt/models/marts/fct_btc_cointime_daily.sql`
- Modify: `dbt/models/marts/_marts.yml`, `cryptoval/cli.py` (`_widen_panel`, `_read_panel`)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Produces:
  - `PANEL_COINTIME_COLUMNS` (in `cli.py`).
  - `_widen_panel(panel, revenue, tvl, assets, cointime=None)`.
  - The mart `fct_btc_cointime_daily(symbol, metric_date, ...COINTIME_COLUMNS)`.

- [ ] **Step 1: Write the failing test**

```python
def test_widen_panel_joins_cointime_onto_btc_only():
    from cryptoval.cli import _widen_panel

    days = pd.to_datetime(["2020-01-01"])
    panel = pd.DataFrame({"symbol": ["BTC", "ETH"], "metric_date": [days[0], days[0]]})
    cointime = pd.DataFrame({"symbol": ["BTC"], "metric_date": days, "liveliness": [0.6]})
    empty = pd.DataFrame()
    wide = _widen_panel(panel, empty, empty, empty, cointime)
    assert wide.set_index("symbol")["liveliness"].to_dict()["BTC"] == 0.6
    assert pd.isna(wide.set_index("symbol")["liveliness"]["ETH"])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q -k widen_panel_joins_cointime`
Expected: FAIL, `TypeError: _widen_panel() takes 4 positional arguments but 5 were given`.

- [ ] **Step 3: Implement**

`dbt/models/staging/stg_btc_cointime.sql`:
```sql
-- Bitcoin coin-age series from the BigQuery UTXO extraction (Plan 3b). Empty until the
-- backfill from genesis is complete: Cointime is cumulative, so it is all-or-nothing (spec 6.3).
select
    'BTC' as symbol,
    cast(metric_date as date) as metric_date,
    * exclude (metric_date)
from read_parquet('{{ var("data_root") }}/silver/btc_cointime_daily/*.parquet')
```
`dbt/models/marts/fct_btc_cointime_daily.sql`:
```sql
-- Family E(b)'s inputs at grain (symbol, metric_date): Cointime primitives, HODL waves by
-- supply and by realized cap, short-term-holder cohort and realized profit. BTC only.
select * from {{ ref('stg_btc_cointime') }}
```
In `_marts.yml`, add a model entry beside `fct_protocol_tvl_daily`:
```yaml
  - name: fct_btc_cointime_daily
    description: >
      Bitcoin coin-age series derived from the BigQuery creation x spend matrix (Plan 3b).
      Empty until the extraction covers genesis to its cutoff without a gap. utxo_supply and
      utxo_realized_cap are checked against Coin Metrics in `make validate`.
    tests:
      - dbt_utils.unique_combination_of_columns:
          arguments:
            combination_of_columns: [symbol, metric_date]
```
In `cli.py`, beside `PANEL_TVL_COLUMNS`:
```python
PANEL_COINTIME_COLUMNS = [
    "symbol", "metric_date", "liveliness", "active_supply", "utxo_supply", "hodl_rc_0d",
    "hodl_rc_6m_1y", "sth_supply", "sth_realized_cap", "realized_profit_usd",
]
```
Change the `_widen_panel` signature to add `cointime: pd.DataFrame | None = None`, and its last
line to:
```python
    wide = _join(wide, assets, ["symbol"], "dim_asset")
    if cointime is None:
        return wide
    return _join(wide, cointime, ["symbol", "metric_date"], "btc cointime")
```
In `_read_panel`'s `_widen_panel(...)` call, add the argument
`maybe("fct_btc_cointime_daily", PANEL_COINTIME_COLUMNS)`.

- [ ] **Step 4: Run the tests and parse dbt**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q && .venv/bin/python -m cryptoval.cli utxo derive && cd dbt && ../.venv/bin/dbt parse --profiles-dir . | tail -2`
Expected: PASS. Derive prints `{'status': 'awaiting_backfill'}`. dbt parse succeeds.

- [ ] **Step 5: Commit**

```bash
git add dbt/models cryptoval/cli.py tests/test_cli.py
git commit -m "feat: fct_btc_cointime_daily mart, joined onto BTC in the method panel"
```

---

### Task 6: The six Family E(b) methods

**Files:**
- Create: `cryptoval/methods/cointime.py`, and `config/methods/{slrv,rpv,sthpl,aviv,active_price,true_market_mean}.yml`
- Modify: `cryptoval/methods/__init__.py`, `docs/methods/` (via `make cards`), `dbt/seeds/` (via `make seed`)
- Test: `tests/test_methods_cointime.py`

**Interfaces:**
- Consumes: panel columns from Task 5, plus `market_cap_usd`, `price_usd`, `realized_cap` and
  `investor_cap_usd`.
- Produces: registered method ids `slrv`, `rpv`, `sthpl`, `aviv`, `active_price`,
  `true_market_mean`. Signals put their value in `diagnostics[method_id]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_methods_cointime.py
import json

import pandas as pd
import pytest

from cryptoval.methods.registry import METHODS, run_methods

import cryptoval.methods  # noqa: F401  registers every family


def _panel(**over):
    row = dict(symbol="BTC", metric_date=pd.Timestamp("2024-01-01"), market_cap_usd=1000.0,
               price_usd=50.0, realized_cap=400.0, investor_cap_usd=300.0, liveliness=0.6,
               active_supply=12.0, utxo_supply=20.0, hodl_rc_0d=0.02, hodl_rc_6m_1y=0.1,
               sth_supply=4.0, sth_realized_cap=160.0, realized_profit_usd=5.0,
               panel_vintage_ts=pd.Timestamp("2024-01-02", tz="UTC"))
    row.update(over)
    return pd.DataFrame([row] * 40).assign(metric_date=pd.date_range("2024-01-01", periods=40))


def _one(method_id, panel):
    out = run_methods(panel, METHODS, [method_id])
    return out.iloc[-1]


@pytest.mark.parametrize("method_id,expected", [
    ("slrv", 0.2), ("rpv", 0.005), ("sthpl", 50 / 40), ("aviv", 1000 * 0.6 / 300),
])
def test_signals_carry_their_value_in_diagnostics(method_id, expected):
    row = _one(method_id, _panel())
    assert row["applicable"] and pd.isna(row["fair_value"])
    assert json.loads(row["diagnostics"])[method_id] == pytest.approx(expected)


@pytest.mark.parametrize("method_id,expected", [("active_price", 400 / 12), ("true_market_mean", 300 / 12)])
def test_cointime_prices_are_fair_values(method_id, expected):
    assert _one(method_id, _panel())["fair_value"] == pytest.approx(expected)


def test_a_non_positive_denominator_refuses_the_day():
    row = _one("true_market_mean", _panel(investor_cap_usd=-1.0))
    assert not row["applicable"] and "investor_cap_usd" in row["reason"]


def test_an_asset_without_coin_age_data_refuses_with_its_density():
    row = _one("slrv", _panel(symbol="ETH", hodl_rc_0d=None, hodl_rc_6m_1y=None))
    assert not row["applicable"] and "0.0% dense" in row["reason"]


def test_every_cointime_method_is_flagged_circular_where_it_reads_market_cap():
    out = run_methods(_panel(), METHODS, ["aviv", "active_price"])
    assert out.groupby("method_id")["circular_inputs"].first().str.len().gt(0).all()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_cointime.py -q`
Expected: FAIL, `KeyError: 'slrv'`.

- [ ] **Step 3: Implement**

`config/methods/slrv.yml`:
```yaml
method_id: slrv
family: E_cost
requires: [hodl_rc_0d, hodl_rc_6m_1y]
min_density: 0.5
min_rows: 30
returns_fair_value: false
params: {}
citation: "Dissertation 5.5, short-to-long-term realised value ratio"

failure_modes:
- 'Daily grain: the "1-day wave" is coins created on the day and unspent at its end, not a rolling
  24 h band (Plan 3b Ruling 2), so the level is not directly comparable with Glassnode''s SLRV.'
- Returns no fair value by declaration; the dissertation reads thresholds (below 0.04 in bear markets).
assumptions:
- Realized-cap weighting values each coin at the daily close of the day it was created.
```
`config/methods/rpv.yml`:
```yaml
method_id: rpv
family: E_cost
requires: [realized_profit_usd, market_cap_usd]
min_density: 0.5
min_rows: 30
returns_fair_value: false
params: {}
citation: "Dissertation 6.1, realised profits-to-value ratio"

failure_modes:
- 'Profit is measured between daily closes (creation day, spend day), so intraday round trips and
  same-day moves realise zero profit.'
- Exchange-internal and change-output spends count as realisations; the chain cannot tell a sale from a self-transfer.
assumptions:
- A spent output is a realisation at that day's close price.
```
`config/methods/sthpl.yml`:
```yaml
method_id: sthpl
family: E_cost
requires: [price_usd, sth_realized_cap, sth_supply]
min_density: 0.5
min_rows: 30
returns_fair_value: false
params: {sth_days: 155}
citation: "Dissertation 6.2, short-term-holder profit/loss (computed as STH-MVRV, Plan 3b Ruling 3)"

failure_modes:
- 'The dissertation''s literal STH P/L over LTH P/L is undefined when LTH P/L crosses zero; this is
  price over the realized price of coins younger than 155 days, which matches its own reading that
  unity is breakeven.'
assumptions:
- A coin's age is days since the output was created; 155 days separates short- from long-term holders.
```
`config/methods/aviv.yml`:
```yaml
method_id: aviv
family: E_cost
requires: [market_cap_usd, liveliness, investor_cap_usd]
min_density: 0.5
min_rows: 30
returns_fair_value: false
params: {}
citation: "Cointime Economics (Puell and Check): active value to investor value"

failure_modes:
- Inherits the MVRV identity's fragility (spec 6.2) through investor cap, which is realized cap minus thermocap.
- Liveliness here weights coinblocks by the protocol subsidy schedule, so lost coins never stop earning coinblocks.
assumptions:
- Active cap is market cap times Liveliness.
```
`config/methods/active_price.yml`:
```yaml
method_id: active_price
family: E_cost
requires: [realized_cap, active_supply]
min_density: 0.5
min_rows: 30
params: {}
citation: "Cointime Economics (Puell and Check): active price"

failure_modes:
- Inherits the MVRV identity's fragility (spec 6.2) through realized cap.
- A cost basis spread over the active supply only, so it rises as Liveliness falls with no change in price.
assumptions:
- Vaulted coins carry no marginal cost basis.
```
`config/methods/true_market_mean.yml`:
```yaml
method_id: true_market_mean
family: E_cost
requires: [investor_cap_usd, active_supply]
min_density: 0.5
min_rows: 30
params: {}
citation: "Cointime Economics (Puell and Check): true market mean price"

failure_modes:
- Inherits the MVRV identity's fragility (spec 6.2) through investor cap.
- Refuses on any day investor cap is not positive.
assumptions:
- Investor cap (realized cap minus thermocap) is the capital investors, not miners, put in.
```
`cryptoval/methods/cointime.py`:
```python
"""Family E(b) -- Cointime and coin-age signals (spec 10.6(b), Plan 3b).

Each is one ratio per day over columns derived from the BigQuery coin-age extraction. BTC is
the only asset with those columns, so every other asset is refused by the registry's density
check, which names the empty column. The four signals return no fair value; their value
lives in diagnostics under the method id. Active price and true market mean are prices.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from cryptoval.methods.base import Estimate, MethodSpec, _finite
from cryptoval.methods.monetary import _refuse
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def _ratio(num_name: str, num, den_name: str, den) -> tuple[float | None, dict, str]:
    num, den = _finite(num), _finite(den)
    if num is None or den is None or den <= 0:
        return None, {}, f"{den_name}={den!r}, {num_name}={num!r}: the ratio is undefined on this day"
    return num / den, {num_name: num, den_name: den}, ""


def _estimates(frame: pd.DataFrame, spec: MethodSpec, params: dict, compute: Callable) -> list[Estimate]:
    out, phash = [], param_set_hash(params)
    for row in frame.itertuples(index=False):
        value, inputs, reason = compute(row)
        if value is not None and spec.returns_fair_value and value <= 0:
            value, reason = None, f"{spec.method_id} is {value!r}; a price must be positive ({inputs})"
        if value is None:
            out.append(_refuse(spec, row, params, reason))
            continue
        out.append(Estimate(
            symbol=row.symbol, metric_date=row.metric_date, method_id=spec.method_id,
            param_set_hash=phash,
            fair_value=value if spec.returns_fair_value else None,
            band_low=None, band_high=None, inputs_used=inputs,
            diagnostics={} if spec.returns_fair_value else {spec.method_id: value},
            applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
        ))
    return out


@register
class Slrv:
    spec = _SPECS["slrv"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _estimates(frame, self.spec, params, lambda r: _ratio(
            "hodl_rc_0d", r.hodl_rc_0d, "hodl_rc_6m_1y", r.hodl_rc_6m_1y))


@register
class Rpv:
    spec = _SPECS["rpv"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _estimates(frame, self.spec, params, lambda r: _ratio(
            "realized_profit_usd", r.realized_profit_usd, "market_cap_usd", r.market_cap_usd))


@register
class Sthpl:
    spec = _SPECS["sthpl"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def compute(r):
            sth_price, _, reason = _ratio("sth_realized_cap", r.sth_realized_cap, "sth_supply", r.sth_supply)
            if sth_price is None:
                return None, {}, reason
            return _ratio("price_usd", r.price_usd, "sth_realized_price", sth_price)

        return _estimates(frame, self.spec, params, compute)


@register
class Aviv:
    spec = _SPECS["aviv"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def compute(r):
            cap, live = _finite(r.market_cap_usd), _finite(r.liveliness)
            active = cap * live if cap is not None and live is not None else None
            return _ratio("active_cap_usd", active, "investor_cap_usd", r.investor_cap_usd)

        return _estimates(frame, self.spec, params, compute)


@register
class ActivePrice:
    spec = _SPECS["active_price"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _estimates(frame, self.spec, params, lambda r: _ratio(
            "realized_cap", r.realized_cap, "active_supply", r.active_supply))


@register
class TrueMarketMean:
    spec = _SPECS["true_market_mean"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _estimates(frame, self.spec, params, lambda r: _ratio(
            "investor_cap_usd", r.investor_cap_usd, "active_supply", r.active_supply))
```
`cryptoval/methods/__init__.py`, after the cohort import:
```python
import cryptoval.methods.cointime  # noqa: F401
```

- [ ] **Step 4: Run the tests and regenerate cards and seeds**

Run: `.venv/bin/python -m pytest tests/test_methods_cointime.py -q && make cards seed && .venv/bin/python -m pytest -q`
Expected: PASS. Six new cards are in `docs/methods/`. The full suite passes.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/methods config/methods docs/methods dbt/seeds tests/test_methods_cointime.py
git commit -m "feat: Family E(b) methods -- SLRV, RPV, STHPL, AVIV, active price, true market mean"
```

---

### Task 7: Live extraction, checks and the chapter

**Files:**
- Create: `reports/what-do-coin-ages-say.md`
- Modify: `reports/modern-layer.md`, where the "What is absent" paragraph points to the new chapter

- [ ] **Step 1: Price the backfill**

Run: `.venv/bin/python -m cryptoval.cli utxo extract --dry-run`
Expected: `status: dry_run`, range `2009-01-03..<today-2>`, with `estimated_bytes` around 562e9.
Abort if it exceeds 900e9.

- [ ] **Step 2: Extract**

Run: `make utxo` (in the background, up to about an hour).
Expected: `status: extracted`. `run_manifest` shows three successes, with billed bytes ≤ the dry run.

- [ ] **Step 3: Derive, build and validate**

Run: `make utxo-derive dbt estimate dbt validate`
Expected:
- derive reports `status: derived`, with liveliness in (0, 1);
- dbt has 0 errors;
- validate shows no `btc_cointime_daily` failures.

If a vendor check fails, the cause is diagnosed (superpowers:systematic-debugging) before anything
else happens: the tolerance is never widened to pass.

- [ ] **Step 4: Write the chapter from the live numbers**

Query `fct_btc_cointime_daily` and `fct_valuation_estimate`. The chapter
`reports/what-do-coin-ages-say.md` covers:
- the extraction (bytes, range, rows);
- the vendor checks (max and median relative error, supply and realized cap);
- Liveliness today and its history;
- each signal against the dissertation's thresholds (SLRV < 0.04; RPV ≥ 2% and ≤ 0.001%; STHPL
  crossings of 1), with dates;
- active price and true market mean against price;
- what is absent (Cointime stock-to-flow, reserve risk, other chains);
- an explicit "descriptive only, no hypothesis family" statement.

In `reports/modern-layer.md`, add one sentence after the "What is absent" paragraph pointing to the
new chapter.

- [ ] **Step 5: Commit**

```bash
git add reports
git commit -m "docs: the coin-age chapter, from a live extraction"
```
