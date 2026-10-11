"""DuckDB as a query engine over Parquet.

The .duckdb file is derived and disposable; deleting it loses nothing,
because the Parquet files under silver/ are the system of record
(spec section 7.1).
"""

from __future__ import annotations

import pathlib

import duckdb
import pandas as pd

from cryptoval.paths import PathResolver


def connect(resolver: PathResolver, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if resolver.is_remote:
        raise NotImplementedError("remote DuckDB roots are added in Plan 5")
    root = pathlib.Path(resolver.root)
    root.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(root / "cryptoval.duckdb"), read_only=read_only)


def register_silver(con: duckdb.DuckDBPyConnection, resolver: PathResolver) -> None:
    table_dir = pathlib.Path(resolver.table("silver", "asset_metrics"))
    if not table_dir.exists() or not any(table_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"no silver asset_metrics parquet under {table_dir}; run `make backfill` first"
        )
    glob = str(table_dir / "*.parquet")
    con.execute(
        f"create or replace view silver_asset_metrics as select * from read_parquet('{glob}')"
    )


def query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetch_df()
