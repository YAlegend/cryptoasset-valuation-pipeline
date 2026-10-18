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
        bounds = pd.read_parquet(
            parts[RANGE_SOURCES[0]][run_id], columns=["range_start", "range_end"]
        )
        runs.append((Range(bounds["range_start"].iloc[0], bounds["range_end"].iloc[0]), run_id))
    runs.sort(key=lambda r: r[0].start)
    for (prev, a), (cur, b) in zip(runs, runs[1:], strict=False):
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
    for (prev, _), (cur, _) in zip(runs, runs[1:], strict=False):
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
    report = {"range": f"{rng.start}..{rng.end}", "estimated_bytes": total, "per_query": estimates}
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
