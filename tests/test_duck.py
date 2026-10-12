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


def test_staging_resolves_vendor_ids_to_symbols(tmp_path):
    """AVAX's two vendor series must recombine into one row per (symbol, date)."""
    import duckdb

    con = duckdb.connect()
    con.execute(
        """
        create table panel as select * from (values
            ('avaxc', date '2026-10-01', 1.0e9, 100.0, null),
            ('avax',  date '2026-10-01', null,  null,  5.0e7)
        ) as t(vendor_id, metric_date, market_cap_usd, active_addresses,
               volume_reported_spot_usd)
        """
    )
    con.execute(
        """
        create table sources as select * from (values
            ('AVAX','network','avaxc'), ('AVAX','valuation','avaxc'), ('AVAX','volume','avax')
        ) as t(symbol, metric_group, source_id)
        """
    )
    out = con.execute(
        """
        with network as (
            select s.symbol, p.metric_date, p.active_addresses from panel p
            join sources s on p.vendor_id = s.source_id and s.metric_group='network'),
        valuation as (
            select s.symbol, p.metric_date, p.market_cap_usd from panel p
            join sources s on p.vendor_id = s.source_id and s.metric_group='valuation'),
        volume as (
            select s.symbol, p.metric_date, p.volume_reported_spot_usd from panel p
            join sources s on p.vendor_id = s.source_id and s.metric_group='volume')
        select n.symbol, n.active_addresses, v.market_cap_usd, vol.volume_reported_spot_usd
        from network n
        join valuation v on n.symbol=v.symbol and n.metric_date=v.metric_date
        join volume vol on n.symbol=vol.symbol and n.metric_date=vol.metric_date
        """
    ).fetchall()
    # One row, carrying the network metrics from avaxc AND the volume from avax.
    assert out == [("AVAX", 100.0, 1.0e9, 5.0e7)]
