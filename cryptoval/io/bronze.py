"""Immutable bronze landing zone.

Writes are atomic: data lands in a sibling temporary directory and is only
renamed into place once the whole frame is written. A fetch that dies
half-way therefore leaves no partition at all, rather than a partial one
that later runs would mistake for complete (Review Focus 5).
"""

from __future__ import annotations

import datetime as dt
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

    final = pathlib.Path(resolver.bronze_partition(source, ingest_date))
    staging = final.with_name(final.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        enriched.to_parquet(staging / "data.parquet", index=False)
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
    finally:
        # On success the rename has already moved staging away; on failure this removes it.
        shutil.rmtree(staging, ignore_errors=True)
    return str(final)


def read_bronze(resolver: PathResolver, source: str, ingest_date: dt.date) -> pd.DataFrame:
    path = pathlib.Path(resolver.bronze_partition(source, ingest_date)) / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no bronze partition at {path}")
    return pd.read_parquet(path)
