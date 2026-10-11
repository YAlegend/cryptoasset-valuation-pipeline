import pathlib

import pandas as pd
import pytest

from cryptoval.io.duck import connect, query, register_silver
from cryptoval.paths import PathResolver
from cryptoval.scd2 import FAR_FUTURE


def _write_silver(root: pathlib.Path) -> None:
    out = pathlib.Path(root, "silver", "asset_metrics")
    out.mkdir(parents=True)
    pd.DataFrame(
        {
            "asset": ["btc", "btc"],
            "metric_date": pd.to_datetime(["2026-10-10", "2026-10-11"], utc=True),
            "CapMrktCurUSD": [1.0e12, 1.1e12],
            "realized_cap": [6.0e11, 6.1e11],
            "effective_start_ts": [pd.Timestamp("2026-10-11", tz="UTC")] * 2,
            "effective_end_ts": [FAR_FUTURE] * 2,
            "is_current": [True, True],
            "row_hash": ["a", "b"],
        }
    ).to_parquet(out / "part-0.parquet", index=False)


def test_connects_and_queries(tmp_path):
    con = connect(PathResolver(str(tmp_path)))
    assert query(con, "select 1 as x").loc[0, "x"] == 1


def test_registers_silver_view(tmp_path):
    _write_silver(tmp_path)
    r = PathResolver(str(tmp_path))
    con = connect(r)
    register_silver(con, r)
    df = query(con, "select count(*) as n from silver_asset_metrics")
    assert df.loc[0, "n"] == 2


def test_silver_view_preserves_bitemporal_columns(tmp_path):
    _write_silver(tmp_path)
    r = PathResolver(str(tmp_path))
    con = connect(r)
    register_silver(con, r)
    cols = set(query(con, "select * from silver_asset_metrics limit 1").columns)
    assert {"effective_start_ts", "effective_end_ts", "is_current", "row_hash"} <= cols


def test_as_of_filter_works_in_sql(tmp_path):
    _write_silver(tmp_path)
    r = PathResolver(str(tmp_path))
    con = connect(r)
    register_silver(con, r)
    sql = """
        select count(*) as n from silver_asset_metrics
        where effective_start_ts <= timestamptz '2026-10-12 00:00:00+00'
          and effective_end_ts  >  timestamptz '2026-10-12 00:00:00+00'
    """
    assert query(con, sql).loc[0, "n"] == 2
    # Before the vintage was recorded nothing was visible: the filter must discriminate.
    before = sql.replace("2026-10-12", "2026-10-10")
    assert query(con, before).loc[0, "n"] == 0


def test_register_silver_raises_a_clear_error_when_absent(tmp_path):
    r = PathResolver(str(tmp_path))
    con = connect(r)
    with pytest.raises(FileNotFoundError, match="no silver"):
        register_silver(con, r)


def test_duckdb_file_is_created_under_the_data_root(tmp_path):
    connect(PathResolver(str(tmp_path)))
    assert pathlib.Path(tmp_path, "cryptoval.duckdb").exists()
