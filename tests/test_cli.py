import datetime as dt
import pathlib

import pandas as pd
import pytest

from cryptoval import cli
from cryptoval.cli import SOURCES, main, run_ingest, run_validate
from cryptoval.config import Asset, Settings
from cryptoval.io.bronze import write_silver
from cryptoval.paths import PathResolver
from cryptoval.scd2 import merge_scd2
from cryptoval.sources.base import FetchResult


def test_all_four_sources_are_registered():
    assert set(SOURCES) == {"coinmetrics", "defillama", "coingecko", "equities"}


def test_write_silver_round_trips(tmp_path):
    r = PathResolver(str(tmp_path))
    df = pd.DataFrame({"asset": ["btc"], "metric_date": pd.to_datetime(["2026-10-11"], utc=True)})
    path = write_silver(r, "asset_metrics", df)
    assert pathlib.Path(path).exists()
    assert len(pd.read_parquet(path)) == 1


def test_write_silver_overwrites_rather_than_appending(tmp_path):
    r = PathResolver(str(tmp_path))
    df = pd.DataFrame({"asset": ["btc"], "metric_date": pd.to_datetime(["2026-10-11"], utc=True)})
    write_silver(r, "asset_metrics", df)
    write_silver(r, "asset_metrics", pd.concat([df, df], ignore_index=True))
    files = list(pathlib.Path(r.table("silver", "asset_metrics")).glob("*.parquet"))
    assert len(files) == 1
    assert len(pd.read_parquet(files[0])) == 2  # the second write replaced the first


class _Boom:
    name = "coinmetrics"
    contract_path = "config/contracts/coinmetrics_asset_metrics.yml"

    def __init__(self, message):
        self.message = message

    def fetch(self, start, end, assets, client):
        raise RuntimeError(self.message)


def test_run_ingest_records_a_failure_without_raising(tmp_path, monkeypatch):
    monkeypatch.setitem(SOURCES, "coinmetrics", _Boom("HTTP 503"))
    settings = Settings(data_root=str(tmp_path))
    report = run_ingest(
        settings, [], ["coinmetrics"], dt.date(2026, 10, 1), dt.date(2026, 10, 11)
    )
    assert report["coinmetrics"]["status"] == "failed"
    assert "503" in report["coinmetrics"]["error"]
    manifest = (tmp_path / "gold" / "run_manifest.jsonl").read_text()
    assert '"status": "failed"' in manifest
    assert "HTTP 503" in manifest


def test_failed_fetch_leaves_no_bronze_partition(tmp_path, monkeypatch):
    monkeypatch.setitem(SOURCES, "coinmetrics", _Boom("boom"))
    settings = Settings(data_root=str(tmp_path))
    run_ingest(settings, [], ["coinmetrics"], dt.date(2026, 10, 1), dt.date(2026, 10, 11))
    bronze = pathlib.Path(tmp_path, "bronze")
    written = list(bronze.rglob("*.parquet")) if bronze.exists() else []
    assert written == []


class _Fixed:
    name = "coinmetrics"
    contract_path = "config/contracts/coinmetrics_asset_metrics.yml"

    def __init__(self, frame):
        self.frame = frame

    def fetch(self, start, end, assets, client):
        return FetchResult(self.frame, len(self.frame))


def _cm_frame(cap):
    return pd.DataFrame(
        {
            "asset": ["btc"],
            "metric_date": pd.to_datetime(["2026-10-09"], utc=True),
            "CapMrktCurUSD": [cap],
        }
    )


def test_successful_ingest_writes_bronze_and_silver_and_a_revision_is_bitemporal(
    tmp_path, monkeypatch
):
    settings = Settings(data_root=str(tmp_path))
    args = (settings, [], ["coinmetrics"], dt.date(2026, 10, 1), dt.date(2026, 10, 11))
    monkeypatch.setitem(SOURCES, "coinmetrics", _Fixed(_cm_frame(100.0)))
    assert run_ingest(*args)["coinmetrics"]["status"] == "success"
    monkeypatch.setitem(SOURCES, "coinmetrics", _Fixed(_cm_frame(120.0)))
    assert run_ingest(*args)["coinmetrics"]["status"] == "success"

    assert list(pathlib.Path(tmp_path, "bronze").rglob("data.parquet"))
    silver = pd.read_parquet(tmp_path / "silver" / "asset_metrics" / "part-0.parquet")
    assert len(silver) == 2
    assert silver.loc[silver["is_current"], "CapMrktCurUSD"].tolist() == [120.0]
    assert silver.loc[~silver["is_current"], "CapMrktCurUSD"].tolist() == [100.0]


def test_a_null_key_is_recorded_as_a_failed_run_not_filtered(tmp_path, monkeypatch):
    frame = _cm_frame(100.0)
    frame.loc[0, "asset"] = None
    monkeypatch.setitem(SOURCES, "coinmetrics", _Fixed(frame))
    settings = Settings(data_root=str(tmp_path))
    report = run_ingest(
        settings, [], ["coinmetrics"], dt.date(2026, 10, 1), dt.date(2026, 10, 11)
    )
    assert report["coinmetrics"]["status"] == "failed"
    assert "null" in report["coinmetrics"]["error"]
    assert not (tmp_path / "silver" / "asset_metrics").exists()


def test_cli_rejects_an_unknown_source():
    assert main(["ingest", "--sources", "nope"]) == 2


def test_cli_rejects_an_end_before_start():
    assert main(["backfill", "--start", "2026-10-11", "--end", "2026-10-01"]) == 2


def test_cli_help_returns_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


# Ruling 3: duplicate symbols in the asset universe


def test_duplicate_symbols_are_rejected():
    assets = [Asset(symbol="BTC", cm_id="btc"), Asset(symbol="ETH", cm_id="eth"),
              Asset(symbol="BTC", cm_id="btc2")]
    with pytest.raises(ValueError, match="BTC"):
        cli.assert_unique_symbols(assets)


def test_duplicate_symbols_are_case_insensitive():
    with pytest.raises(ValueError, match="duplicate"):
        cli.assert_unique_symbols([Asset(symbol="BTC"), Asset(symbol="btc")])


def test_the_shipped_asset_universe_has_no_duplicate_symbols():
    from cryptoval.config import load_config

    _, assets = load_config()
    cli.assert_unique_symbols(assets)


def test_main_exits_2_on_a_duplicate_symbol(monkeypatch, capsys):
    dup = [Asset(symbol="BTC", cm_id="a"), Asset(symbol="BTC", cm_id="b")]
    monkeypatch.setattr(cli, "load_config", lambda: (Settings(), dup))
    assert main(["ingest"]) == 2
    assert "BTC" in capsys.readouterr().err


# Rulings 1, 2, 4: validate


def _publish(tmp_path, table, frame, key, as_of_day=1):
    r = PathResolver(str(tmp_path))
    value_cols = [c for c in frame.columns if c not in key]
    merged = merge_scd2(
        pd.DataFrame(), frame, key, value_cols, dt.datetime(2026, 10, as_of_day, tzinfo=dt.UTC)
    )
    write_silver(r, table, merged)


def _llama_frame(rows):
    return pd.DataFrame(
        [
            {
                "protocol": p,
                "metric_date": pd.Timestamp(d, tz="UTC"),
                "total_fees_24h": v,
                "total_fees_30d": 1.0,
                "net_revenue_24h": 1.0,
                "methodology_hash": "h",
            }
            for p, d, v in rows
        ]
    )


def _validate_settings(tmp_path, monkeypatch, only):
    """Restrict validation to one table so other empty tables don't add noise."""
    monkeypatch.setattr(cli, "SILVER_TABLE", {only: cli.SILVER_TABLE[only]})
    return Settings(data_root=str(tmp_path))


def test_validate_flags_an_anomaly_in_one_protocol(tmp_path, monkeypatch):
    # Steady fees, then a 10x jump on the last day, for protocol A only.
    rows = [("A", f"2026-10-0{d}", 100.0) for d in range(1, 6)] + [("A", "2026-10-06", 1000.0)]
    rows += [("B", f"2026-10-0{d}", 100.0) for d in range(1, 7)]
    _publish(tmp_path, "protocol_revenue", _llama_frame(rows), ["protocol", "metric_date"])
    failures = run_validate(_validate_settings(tmp_path, monkeypatch, "defillama"))
    anomalies = [f for f in failures if "volume_anomaly" in f]
    assert len(anomalies) == 1
    assert anomalies[0].startswith("protocol_revenue/volume_anomaly[A:total_fees_24h]: ")
    assert "total_fees_24h=1000" in anomalies[0]


def test_volume_anomaly_is_computed_per_entity_not_across_the_frame(tmp_path, monkeypatch):
    # Each protocol is perfectly flat, but they sit at wildly different levels. Pooled
    # (ungrouped) the sorted series jumps between 1 and 1,000,000 and would be flagged.
    rows = []
    for d in range(1, 7):
        rows += [("small", f"2026-10-0{d}", 1.0), ("huge", f"2026-10-0{d}", 1_000_000.0)]
    _publish(tmp_path, "protocol_revenue", _llama_frame(rows), ["protocol", "metric_date"])
    failures = run_validate(_validate_settings(tmp_path, monkeypatch, "defillama"))
    assert [f for f in failures if "volume_anomaly" in f] == []


def test_validate_ignores_superseded_history_rows(tmp_path, monkeypatch):
    # A revision leaves a closed row behind; uniqueness must apply to the current view.
    r = PathResolver(str(tmp_path))
    key = ["asset", "metric_date"]
    first = merge_scd2(pd.DataFrame(), _cm_frame(100.0), key, ["CapMrktCurUSD"],
                       dt.datetime(2026, 10, 10, tzinfo=dt.UTC))
    second = merge_scd2(first, _cm_frame(120.0), key, ["CapMrktCurUSD"],
                        dt.datetime(2026, 10, 11, tzinfo=dt.UTC))
    write_silver(r, "asset_metrics", second)
    settings = _validate_settings(tmp_path, monkeypatch, "coinmetrics")
    assert [f for f in run_validate(settings) if "uniqueness" in f] == []


def test_validate_names_table_check_and_detail_and_main_exits_nonzero(
    tmp_path, monkeypatch, capsys
):
    bad = _cm_frame(-5.0)  # violates the minimum: 0 range
    _publish(tmp_path, "asset_metrics", bad, ["asset", "metric_date"])
    settings = _validate_settings(tmp_path, monkeypatch, "coinmetrics")
    failures = run_validate(settings)
    ranges = [f for f in failures if f.startswith("asset_metrics/ranges: ")]
    assert len(ranges) == 1 and "CapMrktCurUSD < 0" in ranges[0]

    monkeypatch.setattr(cli, "load_config", lambda: (settings, []))
    assert main(["validate"]) == 1
    err = capsys.readouterr().err
    assert "FAIL asset_metrics/ranges: " in err


def test_validate_reports_a_missing_table(tmp_path, monkeypatch):
    failures = run_validate(_validate_settings(tmp_path, monkeypatch, "coinmetrics"))
    assert failures == ["asset_metrics: no data ingested"]
