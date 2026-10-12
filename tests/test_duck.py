import pathlib

import pandas as pd
import pytest

from cryptoval.io.duck import connect, query, register_silver
from cryptoval.paths import PathResolver
from cryptoval.scd2 import FAR_FUTURE


def _write_silver(root: pathlib.Path, with_superseded: bool = False) -> None:
    out = pathlib.Path(root, "silver", "asset_metrics")
    out.mkdir(parents=True)
    df = pd.DataFrame(
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
    )
    if with_superseded:
        # The first vintage of 2026-10-10, closed when the revision arrived on 10-11.
        old = df.iloc[[0]].copy()
        old["CapMrktCurUSD"] = 0.9e12
        old["effective_start_ts"] = pd.Timestamp("2026-10-10", tz="UTC")
        old["effective_end_ts"] = pd.Timestamp("2026-10-11", tz="UTC")
        old["is_current"] = False
        old["row_hash"] = "old"
        df = pd.concat([df, old], ignore_index=True)
    df.to_parquet(out / "part-0.parquet", index=False)


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
    _write_silver(tmp_path, with_superseded=True)
    r = PathResolver(str(tmp_path))
    con = connect(r)
    register_silver(con, r)

    def visible(ts: str) -> int:
        sql = f"""
            select count(*) as n from silver_asset_metrics
            where effective_start_ts <= timestamptz '{ts}+00'
              and effective_end_ts  >  timestamptz '{ts}+00'
        """
        return int(query(con, sql).loc[0, "n"])

    # After the revision: only the two current rows; the superseded one is excluded
    # by the effective_end_ts bound (3 rows exist in total).
    assert visible("2026-10-12 00:00:00") == 2
    # Between the first load and the revision: only the old vintage was known.
    assert visible("2026-10-10 12:00:00") == 1
    # Before anything was recorded.
    assert visible("2026-10-09 00:00:00") == 0
    # as-of and is_current genuinely differ.
    n_current = query(con, "select count(*) as n from silver_asset_metrics where is_current")
    assert n_current.loc[0, "n"] == 2
    assert query(con, "select count(*) as n from silver_asset_metrics").loc[0, "n"] == 3


def test_register_silver_raises_a_clear_error_when_absent(tmp_path):
    r = PathResolver(str(tmp_path))
    con = connect(r)
    with pytest.raises(FileNotFoundError, match="no silver"):
        register_silver(con, r)


def test_duckdb_file_is_created_under_the_data_root(tmp_path):
    connect(PathResolver(str(tmp_path)))
    assert pathlib.Path(tmp_path, "cryptoval.duckdb").exists()


@pytest.mark.parametrize("root", ["s3://bucket/data", "gs://bucket/data"])
def test_connect_rejects_remote_root_without_side_effects(tmp_path, monkeypatch, root):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(NotImplementedError):
        connect(PathResolver(root))
    assert list(tmp_path.iterdir()) == []


def test_staging_resolves_vendor_ids_to_symbols():
    """Run stg_asset_metrics.sql itself: AVAX's two vendor series become one row per day.

    The model's own SQL text is executed with its dbt references swapped for fixture
    relations, so this fails if the model changes shape -- a hand-written copy of the
    join would drift from it.
    """
    import datetime
    import re

    import duckdb

    model = pathlib.Path(__file__).parent.parent / "dbt/models/staging/stg_asset_metrics.sql"
    sql = model.read_text(encoding="utf-8")
    panel_read = """read_parquet('{{ var("data_root") }}/silver/asset_metrics/*.parquet')"""
    sql = sql.replace(panel_read, "silver_panel")
    sql = re.sub(r"\{\{\s*ref\('(\w+)'\)\s*\}\}", r"\1", sql)
    assert "{{" not in sql, "unsubstituted dbt reference in stg_asset_metrics.sql"

    con = duckdb.connect()
    con.execute(
        """
        create table silver_panel as select * from (values
            ('avaxc', timestamptz '2026-10-01 00:00:00+00', 1.0e9, 100.0, null, true),
            ('avax',  timestamptz '2026-10-01 00:00:00+00', null,  null,  5.0e7, true),
            -- avax serves a day that avaxc does not: a volume-only day.
            ('avax',  timestamptz '2026-10-02 00:00:00+00', null,  null,  6.0e7, true)
        ) as t(asset, metric_date, "CapMrktCurUSD", "AdrActCnt", "volume_reported_spot_usd_1d",
               is_current)
        """
    )
    # The model selects more columns than the fixture needs; give the rest as nulls.
    for col in ("CapMVRVCur", "realized_cap", "PriceUSD", "SplyCur", "HashRate", "IssTotUSD",
                "FeeTotNtv", "TxCnt", "row_hash"):
        con.execute(f'alter table silver_panel add column "{col}" double')
    con.execute("alter table silver_panel add column effective_start_ts timestamptz")
    con.execute("alter table silver_panel add column effective_end_ts timestamptz")
    con.execute(
        """
        create table asset_metric_sources as select * from (values
            ('AVAX','network','cm','avaxc'), ('AVAX','valuation','cm','avaxc'),
            ('AVAX','volume','cm','avax')
        ) as t(symbol, metric_group, source_kind, source_id)
        """
    )
    out = con.execute(
        f"select symbol, metric_date, active_addresses, market_cap_usd, "
        f"volume_reported_spot_usd from ({sql}) order by metric_date"
    ).fetchall()
    # One row per (symbol, date). Day 1 carries network+valuation from avaxc AND volume from
    # avax; day 2 exists only under the volume id and must survive with the rest null.
    assert out == [
        ("AVAX", datetime.date(2026, 10, 1), 100.0, 1.0e9, 5.0e7),
        ("AVAX", datetime.date(2026, 10, 2), None, None, 6.0e7),
    ]
