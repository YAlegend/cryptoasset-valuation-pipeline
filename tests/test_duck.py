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


_PANEL_METRIC_COLUMNS = (
    "CapMVRVCur", "realized_cap", "PriceUSD", "SplyCur", "HashRate", "IssTotUSD",
    "FeeTotNtv", "TxCnt",
)


def _model_sql(name: str, layer: str) -> str:
    """A dbt model's own SQL text, with its reads swapped for fixture relations.

    The model is executed, not paraphrased: a hand-written copy of a join would drift
    from the model it claims to test.
    """
    import re

    model = pathlib.Path(__file__).parent.parent / f"dbt/models/{layer}/{name}.sql"
    sql = model.read_text(encoding="utf-8")
    for source in ("asset_metrics", "market_chart"):
        sql = sql.replace(
            f"""read_parquet('{{{{ var("data_root") }}}}/silver/{source}/*.parquet')""",
            f"silver_{source}",
        )
    sql = re.sub(r"\{\{\s*ref\('(\w+)'\)\s*\}\}", r"\1", sql)
    assert "{{" not in sql, f"unsubstituted dbt reference in {name}.sql"
    return sql


def _silver_panel(con, rows: str) -> None:
    """Create the silver_asset_metrics fixture; columns the model needs but a test does not
    care about are added as nulls."""
    con.execute(
        f"""
        create table silver_asset_metrics as select * from (values {rows})
        as t(asset, metric_date, "CapMrktCurUSD", "AdrActCnt", "volume_reported_spot_usd_1d",
             effective_start_ts, row_hash, is_current)
        """
    )
    for col in _PANEL_METRIC_COLUMNS:
        con.execute(f'alter table silver_asset_metrics add column "{col}" double')
    con.execute("alter table silver_asset_metrics add column effective_end_ts timestamptz")


def test_staging_resolves_vendor_ids_to_symbols():
    """Run stg_asset_metrics.sql itself: AVAX's two vendor series become one row per day."""
    import datetime

    import duckdb

    sql = _model_sql("stg_asset_metrics", "staging")
    con = duckdb.connect()
    _silver_panel(
        con,
        """
        ('avaxc', timestamptz '2026-10-01 00:00:00+00', 1.0e9, 100.0, null,
         timestamptz '2026-10-02 00:00:00+00', 'h1', true),
        ('avax',  timestamptz '2026-10-01 00:00:00+00', null,  null,  5.0e7,
         timestamptz '2026-10-02 00:00:00+00', 'h2', true),
        -- avax serves a day that avaxc does not: a volume-only day.
        ('avax',  timestamptz '2026-10-02 00:00:00+00', null,  null,  6.0e7,
         timestamptz '2026-10-03 00:00:00+00', 'h3', true)
        """,
    )
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


def test_a_volume_only_symbol_day_still_carries_a_panel_vintage():
    """panel_vintage_ts is the latest vintage among the contributing metric groups.

    A day that exists solely through the volume group (cm:avax outside cm:avaxc's range) has
    no network row; deriving the vintage from `network` alone would leave it NULL, and the
    estimate built on it untraceable.
    """
    import datetime

    import duckdb

    sql = _model_sql("stg_asset_metrics", "staging")
    con = duckdb.connect()
    _silver_panel(
        con,
        """
        -- Day 1: network/valuation vintage Oct 2, volume vintage Oct 5 (a later restatement).
        ('avaxc', timestamptz '2026-10-01 00:00:00+00', 1.0e9, 100.0, null,
         timestamptz '2026-10-02 00:00:00+00', 'net1', true),
        ('avax',  timestamptz '2026-10-01 00:00:00+00', null,  null,  5.0e7,
         timestamptz '2026-10-05 00:00:00+00', 'vol1', true),
        -- Day 2: volume only.
        ('avax',  timestamptz '2026-10-02 00:00:00+00', null,  null,  6.0e7,
         timestamptz '2026-10-03 00:00:00+00', 'vol2', true),
        -- Day 3: network only; the volume group contributes nothing.
        ('avaxc', timestamptz '2026-10-03 00:00:00+00', 1.1e9, 110.0, null,
         timestamptz '2026-10-04 00:00:00+00', 'net3', true)
        """,
    )
    con.execute(
        """
        create table asset_metric_sources as select * from (values
            ('AVAX','network','cm','avaxc'), ('AVAX','valuation','cm','avaxc'),
            ('AVAX','volume','cm','avax')
        ) as t(symbol, metric_group, source_kind, source_id)
        """
    )
    out = con.execute(
        f"select metric_date, effective_start_ts at time zone 'UTC', row_hash "
        f"from ({sql}) order by metric_date"
    ).fetchall()
    d = datetime.datetime
    assert out == [
        # The latest contributing vintage wins, and the hash is that group's hash.
        (datetime.date(2026, 10, 1), d(2026, 10, 5), "vol1"),
        (datetime.date(2026, 10, 2), d(2026, 10, 3), "vol2"),
        (datetime.date(2026, 10, 3), d(2026, 10, 4), "net3"),
    ]


def test_the_two_volume_series_are_both_carried_and_never_coalesced():
    """Spec 10.1: they differ 5x on BTC, so a method must choose explicitly."""
    import duckdb

    con = duckdb.connect()
    con.execute(
        """
        create table fct as select * from (values
            ('BTC', date '2026-10-10', 1.666e12, 'coinmetrics', 2.97e9, 1.48e10),
            ('TRX', date '2026-10-10', 3.14e10,  'coingecko',   5.84e7, 2.54e8)
        ) as t(symbol, metric_date, market_cap_usd, market_cap_source,
               volume_reported_spot_usd, volume_total_usd)
        """
    )
    rows = con.execute(
        """
        select symbol, market_cap_source,
               volume_total_usd / nullif(volume_reported_spot_usd, 0) as ratio
        from fct order by symbol
        """
    ).fetchall()
    btc = rows[0]
    # If a model ever coalesced them the ratio would be exactly 1.0 and the 5x gap would
    # vanish without anyone noticing.
    assert btc[2] > 4.0
    # TRX's market cap can only come from CoinGecko; the row says so rather than implying
    # a Coin Metrics figure.
    assert rows[1][1] == "coingecko"


def test_fct_asset_metrics_daily_sql_keeps_volumes_apart_and_records_the_market_cap_source():
    """Run fct_asset_metrics_daily.sql itself against fixture staging relations."""
    import duckdb

    sql = _model_sql("fct_asset_metrics_daily", "marts")
    con = duckdb.connect()
    con.execute(
        """
        create table stg_asset_metrics as select * from (values
            -- BTC: Coin Metrics serves market cap and volume; CoinGecko serves its own volume.
            ('BTC', date '2026-10-10', 1.0e12, 2.0e11, 100.0, 1.0e3, 2.0e7, null, null,
             null, null, 5.0e9, timestamptz '2026-10-11 00:00:00+00'),
            -- BTC: a zero market cap must give a NULL velocity, never infinity.
            ('BTC', date '2026-10-11', 0.0, 2.0e11, 100.0, 1.0e3, 2.0e7, null, null,
             null, null, 5.0e9, timestamptz '2026-10-11 00:00:00+00'),
            -- BTC: Coin Metrics has no volume that day; CoinGecko's must not stand in for it.
            ('BTC', date '2026-10-12', 1.0e12, 2.0e11, 100.0, 1.0e3, 2.0e7, null, null,
             null, null, null, timestamptz '2026-10-11 00:00:00+00'),
            -- TRX: no Coin Metrics market cap at all.
            ('TRX', date '2026-10-10', null, null, null, null, null, null, null,
             null, null, 5.0e7, timestamptz '2026-10-11 00:00:00+00'),
            ('TRX', date '2026-10-11', null, null, null, null, null, null, null,
             null, null, 5.0e7, timestamptz '2026-10-11 00:00:00+00')
        ) as t(symbol, metric_date, market_cap_usd, realized_cap, price_usd, supply, hash_rate,
               issuance_usd, fees_native, active_addresses, tx_count,
               volume_reported_spot_usd, effective_start_ts)
        """
    )
    con.execute("alter table stg_asset_metrics add column mvrv double")
    con.execute("alter table stg_asset_metrics add column row_hash varchar")
    con.execute(
        """
        create table stg_market_chart as select * from (values
            ('BTC', date '2026-10-10', 100.5, 9.0e11, 1.5e10),
            ('BTC', date '2026-10-11', 100.5, 9.0e11, 1.5e10),
            ('BTC', date '2026-10-12', 100.5, 9.0e11, 1.5e10),
            ('TRX', date '2026-10-10', 0.3, 3.0e10, 2.5e8),
            -- TRX, second day: CoinGecko has no market cap either.
            ('TRX', date '2026-10-11', 0.3, null, 2.5e8)
        ) as t(symbol, metric_date, cg_price_usd, cg_market_cap_usd, cg_volume_total_usd)
        """
    )
    con.execute(
        """
        create table asset_metric_sources as select * from (values
            ('BTC','valuation','cm','btc'), ('TRX','valuation','coingecko','tron')
        ) as t(symbol, metric_group, source_kind, source_id)
        """
    )
    cols = [c[0] for c in con.execute(f"describe select * from ({sql})").fetchall()]
    assert "panel_vintage_ts" in cols and "effective_end_ts" not in cols

    rows = {
        (r[0], str(r[1])): r[2:]
        for r in con.execute(
            f"""
            select symbol, metric_date, market_cap_usd, market_cap_source,
                   volume_reported_spot_usd, volume_total_usd,
                   velocity_reported, velocity_total, panel_vintage_ts is not null
            from ({sql})
            """
        ).fetchall()
    }
    # Both volumes side by side, each with its own velocity over the same market cap.
    assert rows[("BTC", "2026-10-10")] == (
        1.0e12, "coinmetrics", 5.0e9, 1.5e10, 5.0e-3, 1.5e-2, True
    )
    # A zero market cap leaves both velocities NULL.
    assert rows[("BTC", "2026-10-11")][4:6] == (None, None)
    # Coin Metrics' missing volume stays missing; CoinGecko's is not substituted for it.
    assert rows[("BTC", "2026-10-12")][2:6] == (None, 1.5e10, None, 1.5e-2)
    # TRX's market cap is CoinGecko's, and the row says so.
    assert rows[("TRX", "2026-10-10")][:2] == (3.0e10, "coingecko")
    # Where CoinGecko has none either, the source is NULL too, and so are the velocities.
    assert rows[("TRX", "2026-10-11")][:2] == (None, None)
    assert rows[("TRX", "2026-10-11")][4:6] == (None, None)
