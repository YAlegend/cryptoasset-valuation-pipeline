"""Immutable bronze landing zone.

Writes are atomic: data lands in a sibling temporary directory and is only
renamed into place once the whole frame is written. A fetch that dies
half-way therefore leaves no partition at all, rather than a partial one
that later runs would mistake for complete (Review Focus 5).

Each run lands in its own ``run_id=`` partition and a collision is refused, never
overwritten: bronze is append-only.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import shutil

import pandas as pd

from cryptoval.hashing import stable_hash
from cryptoval.paths import PathResolver


def write_bronze(
    resolver: PathResolver,
    source: str,
    ingest_date: dt.date,
    df: pd.DataFrame,
    request_params: dict,
    run_id: str,
) -> str:
    if resolver.is_remote:
        raise NotImplementedError(
            "bronze writes to object storage are added in Plan 5; "
            "set a local CRYPTOVAL_DATA_ROOT for now"
        )
    if df is None or df.empty:
        raise ValueError(
            f"refusing to write an empty bronze partition for {source} {ingest_date}: "
            "an empty frame means the fetch failed"
        )

    enriched = df.copy()
    enriched["_fetched_at"] = pd.Timestamp.now(tz="UTC")
    enriched["_source"] = source
    enriched["_request_hash"] = stable_hash(request_params)
    enriched["_run_id"] = run_id

    final = pathlib.Path(resolver.bronze_partition(source, ingest_date, run_id))
    if final.exists():
        # A run_id is unique per run, so this is a bug rather than an intentional rewrite.
        # Bronze is append-only: nothing here deletes.
        raise FileExistsError(f"bronze partition already exists for run {run_id}: {final}")
    # The name must not begin with "run_id=": read_bronze globs that prefix, and a staging
    # directory already holds data.parquet before the rename, so it would look like a run.
    staging = final.with_name(".staging-" + final.name)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        enriched.to_parquet(staging / "data.parquet", index=False)
        staging.rename(final)
    finally:
        # On success the rename has already moved staging away; on failure this removes it.
        # Only staging is ever cleaned up: a finished partition is never deleted.
        shutil.rmtree(staging, ignore_errors=True)
    return str(final)


def read_bronze(
    resolver: PathResolver,
    source: str,
    ingest_date: dt.date,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Read one run's partition, defaulting to the newest run for that date."""
    if run_id is not None:
        path = pathlib.Path(resolver.bronze_partition(source, ingest_date, run_id)) / "data.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no bronze partition at {path}")
        return pd.read_parquet(path)

    prefix = pathlib.Path(resolver.bronze_date_prefix(source, ingest_date))
    runs = sorted(
        (p for p in prefix.glob("run_id=*") if (p / "data.parquet").exists()),
        key=lambda p: (p / "data.parquet").stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"no bronze runs under {prefix}")
    return pd.read_parquet(runs[-1] / "data.parquet")


def write_silver(resolver: PathResolver, table: str, df: pd.DataFrame) -> str:
    """Write a silver table as a single compacted Parquet file.

    Yearly or finer partitioning is unnecessary at this volume and would
    create the small-file problem the spec warns about (section 7.4); a
    single file per table keeps DuckDB scans fast and the layout simple.
    """
    if resolver.is_remote:
        raise NotImplementedError("silver writes to object storage are added in Plan 5")
    out_dir = pathlib.Path(resolver.table("silver", table))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part-0.parquet"
    # Silver is the system of record, so never delete before the replacement is safely
    # on disk: write beside it, then swap in a single atomic rename.
    staging = out_dir / "part-0.parquet.tmp"
    try:
        df.to_parquet(staging, index=False)
        for stale in out_dir.glob("*.parquet"):
            if stale != path:
                stale.unlink()
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)
    return str(path)
