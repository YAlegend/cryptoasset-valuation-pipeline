"""The dashboard's published data (Phase 6, Tier 0).

The full .duckdb is ~300 MB; the marts the dashboard reads export to ~60 MB of zstd Parquet,
split under GitHub's 50 MB per-file warning, which the daily workflow commits to the `dashboard`
branch. The hosted app rebuilds a DuckDB file from that Parquet on start.
"""

from __future__ import annotations

import pathlib
import sys

import duckdb

TABLES = (
    "fct_asset_metrics_daily",
    "fct_btc_cointime_daily",
    "fct_valuation_estimate",
    "dim_method",
    "dim_asset",
)


def export(db: str, out: str) -> None:
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db, read_only=True)
    try:
        for t in TABLES:
            con.execute(
                f"copy (select * from {t}) to '{out}/{t}' "
                "(format parquet, compression zstd, file_size_bytes '40MB', overwrite true)"
            )
    finally:
        con.close()


def build(snap: str, db: str) -> None:
    con = duckdb.connect(db)
    try:
        for t in TABLES:
            con.execute(
                f"create or replace table {t} as select * from read_parquet('{snap}/{t}/*.parquet')"
            )
    finally:
        con.close()


def stamp(snap: str) -> int:
    """Changes whenever a git pull replaces the snapshot, so a running app rebuilds."""
    return max(int(p.stat().st_mtime) for p in pathlib.Path(snap).rglob("*.parquet"))


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
