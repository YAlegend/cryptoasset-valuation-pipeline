import datetime as dt
import pathlib

import numpy as np
import pandas as pd
import pytest

from cryptoval import cli
from cryptoval.cli import SOURCES, main, run_ingest, run_validate
from cryptoval.config import Asset, Settings
from cryptoval.io.bronze import write_silver
from cryptoval.paths import PathResolver
from cryptoval.scd2 import merge_scd2
from cryptoval.sources.base import FetchResult


def test_every_source_is_registered():
    assert set(SOURCES) == {
        "coinmetrics",
        "defillama",
        "defillama_history",
        "defillama_tvl",
        "coingecko",
        "coingecko_supply",
        "equities",
    }


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
    report = run_ingest(settings, [], ["coinmetrics"], dt.date(2026, 10, 1), dt.date(2026, 10, 11))
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


class _Supply(_Fixed):
    name = "coingecko_supply"
    contract_path = "config/contracts/coingecko_supply.yml"


def test_a_daily_supply_ingest_with_unchanged_values_opens_no_new_version(
    tmp_path, monkeypatch, load_fixture
):
    """The real ingest path (_value_cols -> row_hash), two runs at different instants."""
    from cryptoval.sources.prices import parse_supply_metadata

    frame = parse_supply_metadata(load_fixture("coingecko_markets.json"))
    monkeypatch.setitem(SOURCES, "coingecko_supply", _Supply(frame))
    args = (
        Settings(data_root=str(tmp_path)),
        [],
        ["coingecko_supply"],
        dt.date(2026, 10, 13),
        dt.date(2026, 10, 13),
    )
    assert run_ingest(*args)["coingecko_supply"]["status"] == "success"
    path = tmp_path / "silver" / "supply_metadata" / "part-0.parquet"
    first = pd.read_parquet(path)
    assert run_ingest(*args)["coingecko_supply"]["status"] == "success"
    second = pd.read_parquet(path)

    assert len(first) == len(second) == 2
    assert second["is_current"].all()
    assert (second["effective_start_ts"] == first["effective_start_ts"]).all()
    # ETH's null max_supply survives parquet.
    assert second.set_index("gecko_id")["max_supply"].isna().to_dict() == {
        "aave": False,
        "ethereum": True,
    }


def test_a_null_key_is_recorded_as_a_failed_run_not_filtered(tmp_path, monkeypatch):
    frame = _cm_frame(100.0)
    frame.loc[0, "asset"] = None
    monkeypatch.setitem(SOURCES, "coinmetrics", _Fixed(frame))
    settings = Settings(data_root=str(tmp_path))
    report = run_ingest(settings, [], ["coinmetrics"], dt.date(2026, 10, 1), dt.date(2026, 10, 11))
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
    assets = [
        Asset(symbol="BTC", cm_id="btc"),
        Asset(symbol="ETH", cm_id="eth"),
        Asset(symbol="BTC", cm_id="btc2"),
    ]
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


# The protocols the validate tests publish are all "mapped", so the gate treats them as in
# scope; the scoping behaviour itself is pinned by the two tests that follow.
_UNIVERSE = [
    Asset(
        symbol="UNI",
        sector="defi",
        consensus="other",
        cm_id="uni",
        gecko_id="uniswap",
        defillama_slugs=("A", "B", "small", "huge"),
    ),
    Asset(symbol="BTC", sector="store_of_value", consensus="pow", cm_id="btc", gecko_id="bitcoin"),
    Asset(symbol="USDC", sector="stablecoin", consensus="other", cm_id="usdc", gecko_id="usd-coin"),
]


def _validate_settings(tmp_path, monkeypatch, only):
    """Restrict validation to one table so other empty tables don't add noise."""
    monkeypatch.setattr(cli, "SILVER_TABLE", {only: cli.SILVER_TABLE[only]})
    return Settings(data_root=str(tmp_path))


def test_validate_flags_an_anomaly_in_one_protocol(tmp_path, monkeypatch):
    # Steady fees, then a 10x jump on the last day, for protocol A only.
    rows = [("A", f"2026-10-0{d}", 100.0) for d in range(1, 6)] + [("A", "2026-10-06", 1000.0)]
    rows += [("B", f"2026-10-0{d}", 100.0) for d in range(1, 7)]
    _publish(tmp_path, "protocol_revenue", _llama_frame(rows), ["protocol", "metric_date"])
    report = run_validate(_validate_settings(tmp_path, monkeypatch, "defillama"), _UNIVERSE)
    anomalies = [w for w in report.warnings if "volume_anomaly" in w]
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
    report = run_validate(_validate_settings(tmp_path, monkeypatch, "defillama"), _UNIVERSE)
    assert report.warnings == []


def test_validate_ignores_superseded_history_rows(tmp_path, monkeypatch):
    # A revision leaves a closed row behind; uniqueness must apply to the current view.
    r = PathResolver(str(tmp_path))
    key = ["asset", "metric_date"]
    first = merge_scd2(
        pd.DataFrame(),
        _cm_frame(100.0),
        key,
        ["CapMrktCurUSD"],
        dt.datetime(2026, 10, 10, tzinfo=dt.UTC),
    )
    second = merge_scd2(
        first, _cm_frame(120.0), key, ["CapMrktCurUSD"], dt.datetime(2026, 10, 11, tzinfo=dt.UTC)
    )
    write_silver(r, "asset_metrics", second)
    settings = _validate_settings(tmp_path, monkeypatch, "coinmetrics")
    assert [f for f in run_validate(settings, _UNIVERSE).failures if "uniqueness" in f] == []


def test_validate_names_table_check_and_detail_and_main_exits_nonzero(
    tmp_path, monkeypatch, capsys
):
    bad = _cm_frame(-5.0)  # violates the minimum: 0 range
    _publish(tmp_path, "asset_metrics", bad, ["asset", "metric_date"])
    settings = _validate_settings(tmp_path, monkeypatch, "coinmetrics")
    failures = run_validate(settings, _UNIVERSE).failures
    ranges = [f for f in failures if f.startswith("asset_metrics/ranges: ")]
    assert len(ranges) == 1 and "CapMrktCurUSD < 0" in ranges[0]

    monkeypatch.setattr(cli, "load_config", lambda: (settings, _UNIVERSE))
    assert main(["validate"]) == 1
    err = capsys.readouterr().err
    assert "FAIL asset_metrics/ranges: " in err


def test_validate_reports_a_missing_table(tmp_path, monkeypatch):
    report = run_validate(_validate_settings(tmp_path, monkeypatch, "coinmetrics"), _UNIVERSE)
    assert report.failures == ["asset_metrics: no data ingested"]


# Fix round 1, Ruling D: a partial fetch is neither a success nor a total loss


class _Partial:
    name = "coingecko"
    contract_path = "config/contracts/coingecko_market_chart.yml"

    def fetch(self, start, end, assets, client):
        frame = pd.DataFrame(
            {
                "asset": ["bitcoin"],
                "metric_date": pd.to_datetime(["2026-10-09"], utc=True),
                "price_usd": [87000.0],
            }
        )
        return FetchResult(frame, 1, ("ethereum: 429 Too Many Requests",))


def test_run_ingest_keeps_partial_rows_and_records_the_status_and_error(tmp_path, monkeypatch):
    monkeypatch.setitem(SOURCES, "coingecko", _Partial())
    settings = Settings(data_root=str(tmp_path))
    report = run_ingest(settings, [], ["coingecko"], dt.date(2026, 10, 1), dt.date(2026, 10, 11))
    assert report["coingecko"]["status"] == "partial"
    assert report["coingecko"]["errors"] == ["ethereum: 429 Too Many Requests"]
    silver = pd.read_parquet(tmp_path / "silver" / "market_chart" / "part-0.parquet")
    assert silver["asset"].tolist() == ["bitcoin"]
    manifest = (tmp_path / "gold" / "run_manifest.jsonl").read_text()
    assert '"status": "partial"' in manifest
    assert "ethereum: 429" in manifest


def test_main_exits_nonzero_when_an_ingest_is_only_partial(tmp_path, monkeypatch):
    monkeypatch.setitem(SOURCES, "coingecko", _Partial())
    settings = Settings(data_root=str(tmp_path))
    monkeypatch.setattr(cli, "load_config", lambda: (settings, _UNIVERSE))
    assert main(["ingest", "--sources", "coingecko"]) == 1


# Fix round 1, Rulings B and C: the volume guard must actually bite


def _panel(table_rows, value_col):
    return pd.DataFrame(
        [
            {
                "asset": a,
                "metric_date": pd.Timestamp(d, tz="UTC"),
                "CapMrktCurUSD": 1.0e12,
                value_col: v,
            }
            for a, d, v in table_rows
        ]
    )


def _steady_then_spike(entity, value_col, spike=10_000.0):
    """Five flat days then a spike, ending today so freshness is not a confound."""
    today = dt.date.today()
    days = [today - dt.timedelta(days=n) for n in range(5, -1, -1)]
    rows = [(entity, d.isoformat(), 100.0) for d in days[:-1]]
    rows.append((entity, days[-1].isoformat(), spike))
    return _panel(rows, value_col)


def test_validate_flags_an_anomalous_coin_metrics_reported_volume(tmp_path, monkeypatch):
    frame = _steady_then_spike("btc", "volume_reported_spot_usd_1d")
    _publish(tmp_path, "asset_metrics", frame, ["asset", "metric_date"])
    report = run_validate(_validate_settings(tmp_path, monkeypatch, "coinmetrics"), _UNIVERSE)
    assert [f for f in report.failures if "volume_anomaly" in f] == []
    anomalies = [w for w in report.warnings if "volume_anomaly" in w]
    assert len(anomalies) == 1
    assert anomalies[0].startswith(
        "asset_metrics/volume_anomaly[btc:volume_reported_spot_usd_1d]: "
    )


def test_validate_flags_an_anomalous_coingecko_volume(tmp_path, monkeypatch):
    frame = _steady_then_spike("bitcoin", "volume_usd")
    _publish(tmp_path, "market_chart", frame, ["asset", "metric_date"])
    report = run_validate(_validate_settings(tmp_path, monkeypatch, "coingecko"), _UNIVERSE)
    anomalies = [w for w in report.warnings if "volume_anomaly" in w]
    assert len(anomalies) == 1
    assert anomalies[0].startswith("market_chart/volume_anomaly[bitcoin:volume_usd]: ")


def test_main_exits_zero_and_prints_warn_when_validation_only_warns(tmp_path, monkeypatch, capsys):
    """Ruling J: an anomaly is a signal to investigate, not proof the data is invalid.
    A real market event (BCH's fork day, +246x) is indistinguishable by ratio from the
    vendor's impossible values, so the ratio warns and the range maximum blocks."""
    _publish(
        tmp_path,
        "asset_metrics",
        _steady_then_spike("btc", "volume_reported_spot_usd_1d"),
        ["asset", "metric_date"],
    )
    settings = _validate_settings(tmp_path, monkeypatch, "coinmetrics")
    monkeypatch.setattr(cli, "load_config", lambda: (settings, _UNIVERSE))
    assert main(["validate"]) == 0
    captured = capsys.readouterr()
    assert "WARN asset_metrics/volume_anomaly[btc:" in captured.err
    assert "FAIL" not in captured.err


def test_an_impossible_volume_blocks_validation(tmp_path, monkeypatch):
    frame = _steady_then_spike("usdc", "volume_reported_spot_usd_1d", spike=6.99755e55)
    _publish(tmp_path, "asset_metrics", frame, ["asset", "metric_date"])
    report = run_validate(_validate_settings(tmp_path, monkeypatch, "coinmetrics"), _UNIVERSE)
    ranges = [f for f in report.failures if "ranges" in f]
    assert len(ranges) == 1
    assert "volume_reported_spot_usd_1d > 1" in ranges[0]


# Review fix pass: reconcile must not report "no drift" on a partial comparison


class _PartialReconcile:
    name = "coingecko"
    contract_path = "config/contracts/coingecko_market_chart.yml"

    def fetch(self, start, end, assets, client):
        frame = pd.DataFrame(
            {
                "asset": ["bitcoin"],
                "metric_date": pd.to_datetime(["2026-10-09"], utc=True),
                "price_usd": [87000.0],
            }
        )
        return FetchResult(frame, 1, ("ethereum: 429 Too Many Requests",))


def test_reconcile_reports_the_errors_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    """Spec 8.2 makes reconcile the detector for revisions outside the trailing window;
    'no drift' over 1 of 24 assets is a false all-clear."""
    monkeypatch.setitem(SOURCES, "coingecko", _PartialReconcile())
    settings = Settings(data_root=str(tmp_path))
    monkeypatch.setattr(cli, "load_config", lambda: (settings, _UNIVERSE))
    assert main(["reconcile", "--sources", "coingecko", "--start", "2026-10-01"]) == 1
    out = capsys.readouterr()
    assert "ethereum: 429" in out.err


def test_validate_fails_when_a_declared_column_is_wholly_null(tmp_path, monkeypatch):
    frame = _panel([("btc", "2026-10-09", 1.5)], "CapMVRVCur")
    frame["CapMVRVCur"] = None
    _publish(tmp_path, "asset_metrics", frame, ["asset", "metric_date"])
    report = run_validate(_validate_settings(tmp_path, monkeypatch, "coinmetrics"), _UNIVERSE)
    null_rate = [f for f in report.failures if "null_rate" in f]
    assert len(null_rate) == 1
    assert "CapMVRVCur" in null_rate[0]


def test_ingest_quarantines_an_impossible_value_before_it_reaches_silver(tmp_path, monkeypatch):
    """The stored vintage must never contain a value the contract calls impossible."""
    settings = Settings(data_root=str(tmp_path))

    class ImpossibleVolumeSource:
        name = "coinmetrics"
        contract_path = "config/contracts/coinmetrics_asset_metrics.yml"

        def fetch(self, start, end, assets, client):
            return FetchResult(
                pd.DataFrame(
                    {
                        "asset": ["usdc"],
                        "metric_date": pd.to_datetime(["2026-05-05"], utc=True),
                        "volume_reported_spot_usd_1d": [8.02e52],
                        "CapMrktCurUSD": [7.52e10],
                    }
                ),
                1,
            )

    monkeypatch.setitem(cli.SOURCES, "coinmetrics", ImpossibleVolumeSource())
    cli.run_ingest(settings, [], ["coinmetrics"], dt.date(2026, 5, 5), dt.date(2026, 5, 5))

    silver = pd.read_parquet(
        pathlib.Path(settings.resolver.table("silver", "asset_metrics")) / "part-0.parquet"
    )
    assert silver["volume_reported_spot_usd_1d"].isna().all()
    # The healthy market cap in the same row survived.
    assert silver["CapMrktCurUSD"].iloc[0] == 7.52e10

    quarantine = pd.read_parquet(
        pathlib.Path(settings.resolver.table("silver", "quarantine")) / "part-0.parquet"
    )
    assert quarantine["raw_value"].iloc[0] == 8.02e52
    assert quarantine["column"].iloc[0] == "volume_reported_spot_usd_1d"


def test_reconcile_does_not_report_a_quarantined_cell_as_a_revision(tmp_path, monkeypatch, capsys):
    """Silver stores the nulled cell; the fresh fetch still carries the impossible raw value.
    Reconcile must clean the fetch the same way, or that cell reads as a revision on every
    run and buries the real ones (spec 8.2). A genuine revision must still show."""
    settings = Settings(data_root=str(tmp_path))
    dates = pd.to_datetime(["2026-05-05", "2026-05-05"], utc=True)

    def frame(btc_cap):
        return pd.DataFrame(
            {
                "asset": ["usdc", "btc"],
                "metric_date": dates,
                "volume_reported_spot_usd_1d": [8.02e52, 2.9e9],
                "CapMrktCurUSD": [7.52e10, btc_cap],
            }
        )

    class Vendor:
        name = "coinmetrics"
        contract_path = "config/contracts/coinmetrics_asset_metrics.yml"
        btc_cap = 1.6e12

        def fetch(self, start, end, assets, client):
            return FetchResult(frame(self.btc_cap), 2)

    vendor = Vendor()
    monkeypatch.setitem(cli.SOURCES, "coinmetrics", vendor)
    monkeypatch.setattr(cli, "load_config", lambda: (settings, _UNIVERSE))
    cli.run_ingest(settings, [], ["coinmetrics"], dt.date(2026, 5, 5), dt.date(2026, 5, 5))

    def reconcile_output():
        argv = ["reconcile", "--sources", "coinmetrics", "--start", "2026-05-05"]
        assert main([*argv, "--end", "2026-05-05"]) == 0
        return capsys.readouterr().out

    # Nothing revised: the quarantined usdc cell must not count as drift.
    assert "'changed': 0" in reconcile_output()

    # btc's market cap is genuinely revised; usdc's quarantined cell is still not a diff.
    vendor.btc_cap = 1.7e12
    assert "'changed': 1" in reconcile_output()
    assert "'unchanged': 1" in reconcile_output()


def test_an_unmapped_protocols_failure_warns_while_a_mapped_one_blocks(tmp_path):
    """Spec 9.2: 2,991 protocols are ingested; a handful matter. The long tail must not
    hold the build hostage, and a real defect in a valuation input must still stop it."""
    settings = Settings(data_root=str(tmp_path))
    uni = Asset(
        symbol="UNI",
        sector="defi",
        consensus="other",
        cm_id="uni",
        gecko_id="uniswap",
        defillama_slugs=("Uniswap V3",),
    )

    def _write(rows: list[dict]) -> None:
        frame = pd.DataFrame(rows)
        frame["effective_start_ts"] = pd.Timestamp("2026-10-11", tz="UTC")
        frame["effective_end_ts"] = pd.NaT
        frame["is_current"] = True
        frame["row_hash"] = "h"
        write_silver(settings.resolver, "protocol_revenue", frame)

    # A negative GROSS fee is a vendor error the contract blocks -- but on a protocol we
    # never value, so it is only a warning.
    _write(
        [
            {
                "protocol": "Lombard LBTC",
                "metric_date": pd.Timestamp("2026-10-11", tz="UTC"),
                "total_fees_24h": -23915.0,
                "total_fees_30d": 1581269.0,
                "net_revenue_24h": -1913.0,
            },
        ]
    )
    report = cli.run_validate(settings, [uni])
    assert not [f for f in report.failures if "protocol_revenue/ranges" in f]
    assert [w for w in report.warnings if "protocol_revenue/ranges" in w]

    # The same defect on a MAPPED protocol blocks.
    _write(
        [
            {
                "protocol": "Uniswap V3",
                "metric_date": pd.Timestamp("2026-10-11", tz="UTC"),
                "total_fees_24h": -450744.0,
                "total_fees_30d": 1.2e7,
                "net_revenue_24h": 92267.0,
            },
        ]
    )
    report = cli.run_validate(settings, [uni])
    assert [f for f in report.failures if "protocol_revenue/ranges" in f]


def test_scoping_leaves_equities_blocking():
    """Equities are entirely in scope; the three vendors with entities outside it are scoped."""
    assert cli.SCOPED_SOURCES == {
        "defillama",
        "defillama_history",
        "defillama_tvl",
        "coinmetrics",
        "coingecko",
        "coingecko_supply",
    }
    assert "equities" not in cli.SCOPE_KEY


def test_an_empty_mapping_warns_rather_than_silently_passing_every_protocol(tmp_path, monkeypatch):
    """With no defillama_slugs anywhere every protocol is unmapped; that is correct but
    would hide a defect from an operator who forgot the mapping, so say so."""
    rows = [("Lombard LBTC", f"2026-10-0{d}", 100.0) for d in range(1, 4)]
    _publish(tmp_path, "protocol_revenue", _llama_frame(rows), ["protocol", "metric_date"])
    bare = [Asset(symbol="BTC", cm_id="btc")]
    report = run_validate(_validate_settings(tmp_path, monkeypatch, "defillama"), bare)
    assert [w for w in report.warnings if "no defillama_slugs configured" in w]
    assert report.failures == []


def test_an_impossible_volume_blocks_for_a_configured_id_and_only_warns_for_an_unmapped_one(
    tmp_path, monkeypatch
):
    """trx_eth was dropped from config but its pre-quarantine rows remain in silver. No mart
    reads it, so its impossible value warns (naming the id); the same value under a
    configured id still blocks, and the 1e12 bound has not moved."""
    settings = _validate_settings(tmp_path, monkeypatch, "coinmetrics")
    spike = 8.18e47

    def _publish_volume(entity):
        frame = _steady_then_spike(entity, "volume_reported_spot_usd_1d", spike=spike)
        _publish(tmp_path, "asset_metrics", frame, ["asset", "metric_date"])

    _publish_volume("trx_eth")
    report = run_validate(settings, _UNIVERSE)
    assert not [f for f in report.failures if "ranges" in f]
    named = [w for w in report.warnings if "asset_metrics/ranges [unmapped: trx_eth]" in w]
    assert len(named) == 1 and "volume_reported_spot_usd_1d > 1" in named[0]

    _publish_volume("usdc")  # configured in _UNIVERSE
    blocked = [f for f in run_validate(settings, _UNIVERSE).failures if "ranges" in f]
    assert len(blocked) == 1 and "volume_reported_spot_usd_1d > 1" in blocked[0]


def test_an_unmapped_coingecko_id_only_warns_too(tmp_path, monkeypatch):
    _publish(
        tmp_path,
        "market_chart",
        _panel([("old-token", "2026-10-09", -3.0)], "price_usd"),
        ["asset", "metric_date"],
    )
    report = run_validate(_validate_settings(tmp_path, monkeypatch, "coingecko"), _UNIVERSE)
    assert [w for w in report.warnings if "[unmapped: old-token]" in w]
    assert not [f for f in report.failures if "ranges" in f]


def _mvrv_violation(entity):
    """realized_cap deliberately disagrees with market_cap / mvrv (100 / 2 = 50, not 7)."""
    return pd.DataFrame(
        {
            "asset": [entity],
            "metric_date": pd.to_datetime(["2026-10-09"], utc=True),
            "CapMrktCurUSD": [100.0],
            "CapMVRVCur": [2.0],
            "realized_cap": [7.0],
        }
    )


def test_an_identity_violation_warns_for_an_unmapped_id_and_blocks_for_a_configured_one(
    tmp_path, monkeypatch
):
    """The identity checks are blocking tripwires (identities.py), so scoping must not make
    them vanish for an id config no longer maps: warn there, block here."""
    settings = _validate_settings(tmp_path, monkeypatch, "coinmetrics")

    _publish(tmp_path, "asset_metrics", _mvrv_violation("trx_eth"), ["asset", "metric_date"])
    report = run_validate(settings, _UNIVERSE)
    assert not [f for f in report.failures if "identity_mvrv" in f]
    warned = [w for w in report.warnings if "identity_mvrv [unmapped: trx_eth]" in w]
    assert len(warned) == 1

    _publish(tmp_path, "asset_metrics", _mvrv_violation("btc"), ["asset", "metric_date"])
    report = run_validate(settings, _UNIVERSE)
    assert [f for f in report.failures if "asset_metrics/identity_mvrv" in f]
    assert not [w for w in report.warnings if "identity_mvrv" in w]


def test_an_unmapped_protocols_volume_spike_warns_behind_the_marker(tmp_path, monkeypatch):
    """The volume guard goes live across the long tail once each protocol has five rows; it
    must be scoped and folded like every other check, not emitted per protocol unmarked."""
    rows = []
    for protocol in ("tail-1", "tail-2"):
        rows += [(protocol, f"2026-10-0{d}", 100.0) for d in range(1, 6)]
        rows.append((protocol, "2026-10-06", 100_000.0))
    rows += [("A", f"2026-10-0{d}", 100.0) for d in range(1, 6)] + [("A", "2026-10-06", 100_000.0)]
    _publish(tmp_path, "protocol_revenue", _llama_frame(rows), ["protocol", "metric_date"])
    report = run_validate(_validate_settings(tmp_path, monkeypatch, "defillama"), _UNIVERSE)

    folded = [w for w in report.warnings if "volume_anomaly[total_fees_24h] [unmapped: " in w]
    assert len(folded) == 1 and "tail-1, tail-2" in folded[0]
    mapped = [w for w in report.warnings if "volume_anomaly[A:total_fees_24h]" in w]
    assert len(mapped) == 1 and "[unmapped" not in mapped[0]
    assert report.failures == []


def _realized_cap_panel(symbol, rows, realized_cap):
    return pd.DataFrame(
        {
            "symbol": [symbol] * rows,
            "metric_date": pd.date_range("2026-01-01", periods=rows, freq="D").date,
            "realized_cap": realized_cap,
            "supply": [1.9e7] * rows,
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )


def test_the_estimate_command_writes_the_gold_table(tmp_path, monkeypatch):
    settings = Settings(data_root=str(tmp_path))
    panel = _realized_cap_panel("BTC", 60, [1.0e12 + i * 1.0e9 for i in range(60)])
    monkeypatch.setattr(cli, "_read_panel", lambda _settings: panel)
    report = cli.run_estimate(settings, [], method_ids=["realized_price"])

    assert report["realized_price"]["rows"] == 60
    out = pd.read_parquet(
        pathlib.Path(settings.resolver.table("gold", "valuation_estimate")) / "part-0.parquet"
    )
    assert set(out["method_id"]) == {"realized_price"}
    assert out["fair_value"].notna().all()


def test_the_estimate_command_reports_refusals_without_failing(tmp_path, monkeypatch):
    """A method that declines is a finding, so the command exits 0 and says how many."""
    settings = Settings(data_root=str(tmp_path))
    thin = _realized_cap_panel("BNB", 5, [1.0e10, None, None, None, None])
    monkeypatch.setattr(cli, "_read_panel", lambda _settings: thin)
    report = cli.run_estimate(settings, [], method_ids=["realized_price"])
    assert report["realized_price"]["refusals"] == 1
    assert report["realized_price"]["rows"] == 1


def test_main_exits_zero_when_a_method_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "load_config", lambda: (Settings(data_root=str(tmp_path)), [Asset(symbol="BTC")])
    )
    monkeypatch.setattr(
        cli, "_read_panel", lambda _settings: _realized_cap_panel("BNB", 5, [1.0e10] * 5)
    )
    assert main(["estimate", "--methods", "realized_price"]) == 0
    assert "refusals" in capsys.readouterr().out


def test_main_rejects_an_unknown_method_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "load_config", lambda: (Settings(data_root=str(tmp_path)), [Asset(symbol="BTC")])
    )
    assert main(["estimate", "--methods", "no_such_method"]) == 2


def _widen_fixtures():
    panel = pd.DataFrame(
        {
            "symbol": ["ETH", "ETH", "BTC"],
            "metric_date": pd.to_datetime(["2026-10-11", "2026-10-12", "2026-10-12"]).date,
            "price_usd": [2500.0, 2550.0, 80000.0],
        }
    )
    revenue = pd.DataFrame(
        {
            "symbol": ["ETH"],
            "metric_date": pd.to_datetime(["2026-10-12"]).date,
            "total_fees_usd": [1.0e6],
            "net_revenue_usd": [9.0e5],
            "revenue_regime": ["retains_revenue"],
            "protocol_count": [1],
        }
    )
    tvl = pd.DataFrame(
        {
            "symbol": ["ETH"],
            "metric_date": pd.to_datetime(["2026-10-12"]).date,
            "tvl_usd": [5.0e10],
        }
    )
    assets = pd.DataFrame(
        {
            "symbol": ["ETH", "BTC"],
            "sector": ["smart_contract_l1", "pow_monetary"],
            "total_supply": [1.2e8, 1.98e7],
            "max_supply": [None, 2.1e7],
            "supply_as_of": pd.to_datetime(["2026-10-13", "2026-10-13"]).date,
            "genesis_date": pd.to_datetime(["2015-07-30", "2009-01-03"]).date,
        }
    )
    return panel, revenue, tvl, assets


def test_widen_panel_left_joins_without_changing_the_row_count():
    panel, revenue, tvl, assets = _widen_fixtures()
    wide = cli._widen_panel(panel, revenue, tvl, assets)

    # The asset panel's grain is sacred: three rows in, three rows out.
    assert len(wide) == 3
    eth_12 = wide.set_index(["symbol", "metric_date"]).loc[("ETH", panel["metric_date"].iloc[1])]
    assert eth_12["net_revenue_usd"] == 9.0e5
    assert eth_12["tvl_usd"] == 5.0e10
    # BTC has no protocol revenue, and ETH's own 10-11 row predates it: null, not zero.
    assert pd.isna(wide.loc[wide["symbol"] == "BTC", "net_revenue_usd"]).all()
    assert pd.isna(wide.loc[wide["metric_date"] == panel["metric_date"].iloc[0], "tvl_usd"]).all()
    # BTC's max supply survives the join; ETH's null stays null.
    assert wide.loc[wide["symbol"] == "BTC", "max_supply"].iloc[0] == 2.1e7
    assert pd.isna(wide.loc[wide["symbol"] == "ETH", "max_supply"]).all()


def test_widen_panel_refuses_a_duplicated_join_key_rather_than_fanning_out():
    panel, revenue, tvl, assets = _widen_fixtures()
    doubled = pd.concat([revenue, revenue], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        cli._widen_panel(panel, doubled, tvl, assets)


def test_widen_panel_refuses_a_column_that_would_be_suffixed_away():
    """Both protocol marts carry protocol_count and composition_complete. A merge would
    land them as _x/_y and a method naming the column in `requires` would find neither."""
    panel, revenue, tvl, assets = _widen_fixtures()
    tvl = tvl.assign(protocol_count=1)
    with pytest.raises(ValueError, match="collide"):
        cli._widen_panel(panel, revenue, tvl, assets)


def test_widen_panel_survives_a_mart_that_does_not_exist_yet():
    panel, revenue, tvl, assets = _widen_fixtures()
    empty = pd.DataFrame(columns=["symbol", "metric_date", "tvl_usd"])
    wide = cli._widen_panel(panel, revenue, empty, assets)
    assert len(wide) == 3
    assert "tvl_usd" not in wide.columns


def test_read_panel_widens_against_the_real_marts(tmp_path, monkeypatch):
    """The column list is a contract Tasks 8-10 name in `requires`: pin it against the
    marts themselves, because a renamed mart column is otherwise a silent refusal."""
    import duckdb

    db = tmp_path / "cryptoval.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "create table fct_asset_metrics_daily as select 'ETH' as symbol, "
        "date '2026-10-12' as metric_date, 2550.0 as price_usd"
    )
    con.execute(
        "create table fct_protocol_revenue_daily as select 'ETH' as symbol, "
        "date '2026-10-12' as metric_date, 1.0e6 as total_fees_usd, 9.0e5 as net_revenue_usd, "
        "'retains_revenue' as revenue_regime, 1 as protocol_count, 1 as protocol_count_prior_day"
    )
    con.execute(
        "create table fct_protocol_tvl_daily as select 'ETH' as symbol, "
        "date '2026-10-12' as metric_date, 5.0e10 as tvl_usd, 1 as protocol_count"
    )
    con.execute(
        "create table dim_asset as select 'ETH' as symbol, 'smart_contract_l1' as sector, "
        "1.2e8 as total_supply, cast(null as double) as max_supply, "
        "date '2026-10-13' as supply_as_of, date '2015-07-30' as genesis_date, 'eth' as cm_id"
    )
    con.close()

    wide = cli._read_panel(Settings(data_root=str(tmp_path)))
    assert len(wide) == 1
    for column in (
        "total_fees_usd",
        "net_revenue_usd",
        "revenue_regime",
        "protocol_count",
        "tvl_usd",
        "total_supply",
        "max_supply",
        "supply_as_of",
        "sector",
        "genesis_date",
    ):
        assert column in wide.columns, column
    # protocol_count came from the revenue mart alone; the TVL mart's copy is not selected.
    assert "protocol_count_x" not in wide.columns
    assert "cm_id" not in wide.columns


def test_a_subset_estimate_run_does_not_delete_the_methods_it_did_not_run(tmp_path):
    """write_gold replaces the table wholesale, so `estimate --methods <subset>` would
    otherwise leave the mart holding a fraction of the methods with every test still green."""
    stored = pd.DataFrame(
        {
            "method_id": ["stock_to_flow", "stock_to_flow", "price_to_fees"],
            "symbol": ["BTC", "ETH", "AAVE"],
            "fair_value": [1.0, 2.0, 3.0],
        }
    )
    (tmp_path / "gold" / "valuation_estimate").mkdir(parents=True)
    stored.to_parquet(tmp_path / "gold" / "valuation_estimate" / "part-0.parquet")

    fresh = pd.DataFrame({"method_id": ["price_to_fees"], "symbol": ["AAVE"], "fair_value": [9.0]})
    merged = cli._keep_unrun_methods(Settings(data_root=str(tmp_path)), fresh)

    assert set(merged["method_id"]) == {"stock_to_flow", "price_to_fees"}
    # The re-run method is REPLACED, not appended to.
    assert merged.loc[merged["method_id"] == "price_to_fees", "fair_value"].tolist() == [9.0]
    assert len(merged) == 3


def test_a_full_estimate_run_still_replaces_everything(tmp_path):
    """A method deleted from config must disappear from gold, not be carried forward."""
    stored = pd.DataFrame({"method_id": ["retired"], "symbol": ["BTC"], "fair_value": [1.0]})
    (tmp_path / "gold" / "valuation_estimate").mkdir(parents=True)
    stored.to_parquet(tmp_path / "gold" / "valuation_estimate" / "part-0.parquet")
    fresh = pd.DataFrame({"method_id": ["retired"], "symbol": ["BTC"], "fair_value": [2.0]})
    merged = cli._keep_unrun_methods(Settings(data_root=str(tmp_path)), fresh)
    assert merged["fair_value"].tolist() == [2.0]


def _score_inputs(days=2000):
    dates = pd.date_range("2019-01-01", periods=days, freq="D")
    price = 100.0 * np.exp(np.cumsum(np.full(days, 0.0005)))
    vintage = pd.Timestamp("2026-10-12 08:30", tz="UTC")
    panel = pd.DataFrame(
        {"symbol": "BTC", "metric_date": dates, "price_usd": price, "panel_vintage_ts": vintage}
    )
    estimates = pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": dates,
            "method_id": "metcalfe_estimated",
            "fair_value": price * np.linspace(0.5, 2.0, days),
            "applicable": True,
        }
    )
    return panel, estimates


def test_score_writes_the_documented_columns_to_gold(tmp_path, monkeypatch):
    from cryptoval.scoring.runner import SCORE_COLUMNS

    settings = Settings(data_root=str(tmp_path))
    inputs = _score_inputs()
    monkeypatch.setattr(cli, "_read_score_inputs", lambda _settings: inputs)
    report = cli.run_score(settings, [], horizons=[30])

    written = pd.read_parquet(
        pathlib.Path(settings.resolver.table("gold", "model_score")) / "part-0.parquet"
    )
    assert list(written.columns) == SCORE_COLUMNS
    assert (written["metric"] == "hypotheses_tested").any()
    assert report["naive_persistence"]["rows"] > 0
    # The vintage ties the scores to the panel state, as fct_valuation_estimate does.
    stamped = written[written["model_id"] != "*"]["panel_vintage_ts"]
    assert stamped.notna().all()
    assert (stamped == inputs[0]["panel_vintage_ts"].max()).all()


def test_the_score_reader_selects_the_panel_vintage_from_the_warehouse(tmp_path):
    """Patching the reader would hide a SELECT that drops the column, and with it the tie
    between every score and the panel state it was computed from."""
    import duckdb

    con = duckdb.connect(str(tmp_path / "cryptoval.duckdb"))
    con.execute(
        "create table fct_asset_metrics_daily as select 'BTC' as symbol, "
        "date '2026-10-12' as metric_date, 80000.0 as price_usd, "
        "timestamp '2026-10-12 08:30:00' as panel_vintage_ts, 1 as unrelated"
    )
    con.execute(
        "create table fct_valuation_estimate as select 'BTC' as symbol, "
        "date '2026-10-12' as metric_date, 'metcalfe_estimated' as method_id, "
        "1.0 as fair_value, true as applicable"
    )
    con.close()

    panel, estimates = cli._read_score_inputs(Settings(data_root=str(tmp_path)))
    assert list(panel.columns) == ["symbol", "metric_date", "price_usd", "panel_vintage_ts"]
    assert panel["panel_vintage_ts"].notna().all()
    assert "fair_value" in estimates.columns


def test_main_exits_zero_when_a_model_loses_to_the_baseline(tmp_path, monkeypatch, capsys):
    """Failing to beat persistence is the expected result (spec 11.3), not a failed run."""
    monkeypatch.setattr(
        cli, "load_config", lambda: (Settings(data_root=str(tmp_path)), [Asset(symbol="BTC")])
    )
    inputs = _score_inputs()
    monkeypatch.setattr(cli, "_read_score_inputs", lambda _settings: inputs)
    assert main(["score", "--models", "naive_persistence,historical_mean", "--horizons", "30"]) == 0
    assert "historical_mean" in capsys.readouterr().out


def test_main_rejects_an_unknown_score_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "load_config", lambda: (Settings(data_root=str(tmp_path)), [Asset(symbol="BTC")])
    )
    assert main(["score", "--models", "no_such_model"]) == 2


@pytest.mark.parametrize("bad", ["0", "-5", "-1", "7.5", "abc", "", "30,0", "30,"])
def test_a_bad_horizon_is_refused_at_the_boundary_before_any_work(
    bad, tmp_path, monkeypatch, capsys
):
    """-5 scores a BACKWARD return the lagged features can partly see and writes the result
    to gold looking like a finding; 0 divides by zero; -1 collides with the sentinel."""
    monkeypatch.setattr(
        cli, "load_config", lambda: (Settings(data_root=str(tmp_path)), [Asset(symbol="BTC")])
    )

    def no_work(_settings):
        raise AssertionError("a bad horizon must be refused before the warehouse is read")

    monkeypatch.setattr(cli, "_read_score_inputs", no_work)
    assert main(["score", "--horizons", bad]) == 2
    err = capsys.readouterr().err
    assert "positive integer" in err
    assert not (tmp_path / "gold").exists()


def test_a_repeated_horizon_is_collapsed_in_order():
    assert cli._parse_horizons("30,7,30") == [30, 7]


def test_a_repeated_horizon_does_not_duplicate_the_gold_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "load_config", lambda: (Settings(data_root=str(tmp_path)), [Asset(symbol="BTC")])
    )
    inputs = _score_inputs()
    monkeypatch.setattr(cli, "_read_score_inputs", lambda _settings: inputs)
    assert main(["score", "--models", "naive_persistence", "--horizons", "30,30"]) == 0
    written = pd.read_parquet(tmp_path / "gold" / "model_score" / "part-0.parquet")
    keys = ["model_id", "symbol", "horizon", "fold", "metric"]
    assert not written.duplicated(subset=keys).any()


def test_a_subset_score_run_does_not_delete_the_models_it_did_not_run(tmp_path, monkeypatch):
    """write_gold replaces the table wholesale, so `score --models <subset>` would otherwise
    leave the mart holding a fraction of the models with every dbt test still green."""
    settings = Settings(data_root=str(tmp_path))
    inputs = _score_inputs()
    monkeypatch.setattr(cli, "_read_score_inputs", lambda _settings: inputs)
    gold = pathlib.Path(settings.resolver.table("gold", "model_score")) / "part-0.parquet"

    cli.run_score(settings, [], ["naive_persistence", "historical_mean"], [30])
    cli.run_score(settings, [], ["naive_persistence", "random_walk_drift"], [30])
    out = pd.read_parquet(gold)

    assert {"naive_persistence", "historical_mean", "random_walk_drift"} <= set(out["model_id"])
    keys = ["model_id", "symbol", "horizon", "fold", "metric"]
    assert not out.duplicated(subset=keys).any()
    # The correction is recomputed over the MERGED family, not left as the first run's.
    dm = out[(out["metric"] == "dm_p_value") & out["value"].notna()]
    count = out[out["metric"] == "hypotheses_tested"]
    assert len(count) == 1
    assert count["value"].iloc[0] == len(dm) == 2
    assert set(out[out["metric"] == "dm_p_value_holm"]["model_id"]) == {
        "historical_mean",
        "random_walk_drift",
    }


def test_a_subset_horizon_run_keeps_the_other_horizons(tmp_path, monkeypatch):
    settings = Settings(data_root=str(tmp_path))
    inputs = _score_inputs()
    monkeypatch.setattr(cli, "_read_score_inputs", lambda _settings: inputs)
    cli.run_score(settings, [], ["naive_persistence"], [30])
    cli.run_score(settings, [], ["naive_persistence"], [7])
    out = pd.read_parquet(
        pathlib.Path(settings.resolver.table("gold", "model_score")) / "part-0.parquet"
    )
    assert {7, 30} <= set(out["horizon"])


def test_a_models_subset_without_the_baseline_is_refused(tmp_path, monkeypatch, capsys):
    """Without naive_persistence in the run no DM test can be formed, so the subset's stored
    DM and Holm rows would be dropped and the Holm family silently shrink (a weaker correction)."""
    monkeypatch.setattr(
        cli, "load_config", lambda: (Settings(data_root=str(tmp_path)), [Asset(symbol="BTC")])
    )

    def no_work(_settings):
        raise AssertionError("refused before the warehouse is read")

    monkeypatch.setattr(cli, "_read_score_inputs", no_work)
    assert main(["score", "--models", "historical_mean,random_walk_drift"]) == 2
    err = capsys.readouterr().err
    assert "must include naive_persistence" in err
    assert "historical_mean" in err
    assert not (tmp_path / "gold").exists()


def test_a_rerun_replaces_the_short_history_rows_rather_than_duplicating_them(
    tmp_path, monkeypatch
):
    """A symbol too short to fold gets a model_id '*' refusal per horizon. They belong to no
    model, so a rerun of any models at that horizon must replace them."""
    settings = Settings(data_root=str(tmp_path))
    panel, estimates = _score_inputs()
    dates = pd.date_range("2019-01-01", periods=50, freq="D")
    short = pd.DataFrame({"symbol": "ETH", "metric_date": dates, "price_usd": 10.0})
    short["panel_vintage_ts"] = panel["panel_vintage_ts"].iloc[0]
    inputs = (pd.concat([panel, short], ignore_index=True), estimates)
    monkeypatch.setattr(cli, "_read_score_inputs", lambda _settings: inputs)
    gold = pathlib.Path(settings.resolver.table("gold", "model_score")) / "part-0.parquet"

    cli.run_score(settings, [], ["naive_persistence", "historical_mean"], [30])
    first = pd.read_parquet(gold)
    assert (first["model_id"] == "*").sum() >= 1  # the fixture really produces the rows
    cli.run_score(settings, [], ["naive_persistence", "random_walk_drift"], [30])

    out = pd.read_parquet(gold)
    keys = ["model_id", "symbol", "horizon", "fold", "metric"]
    assert not out.duplicated(subset=keys).any()
    assert (out["model_id"] == "*").sum() == (first["model_id"] == "*").sum()


def test_the_score_report_separates_beating_persistence_from_losing_to_it(tmp_path, monkeypatch):
    """A significant DM statistic is not a win: positive means persistence had the lower loss."""
    settings = Settings(data_root=str(tmp_path))
    monkeypatch.setattr(cli, "_read_score_inputs", lambda _settings: _score_inputs())
    report = cli.run_score(settings, [], ["naive_persistence", "historical_mean", "ols"], [30])

    written = pd.read_parquet(
        pathlib.Path(settings.resolver.table("gold", "model_score")) / "part-0.parquet"
    )
    for model_id in ("historical_mean", "ols"):
        rows = written[written["model_id"] == model_id]
        holm = rows[rows["metric"] == "dm_p_value_holm"]["value"].to_numpy()
        stat = rows[rows["metric"] == "dm_statistic"]["value"].to_numpy()
        outcome = report[model_id]
        assert outcome["beat_baseline_after_holm"] == int(((holm < 0.05) & (stat < 0)).sum())
        assert outcome["lost_to_baseline_after_holm"] == int(((holm < 0.05) & (stat > 0)).sum())
    assert report["naive_persistence"]["beat_baseline_after_holm"] == 0
    # Not vacuous: the drifting fixture gives at least one significant verdict to classify.
    assert sum(r["lost_to_baseline_after_holm"] for r in report.values()) + sum(
        r["beat_baseline_after_holm"] for r in report.values()
    )


def test_a_significant_loss_to_persistence_is_not_counted_as_a_win(tmp_path, monkeypatch):
    """The live run's shape: noise targets, a model that loses to the zero forecast. Counting
    significance without sign reported 39-65 'beats' per model when every one was a loss."""
    days = 2000
    rng = np.random.default_rng(7)
    dates = pd.date_range("2019-01-01", periods=days, freq="D")
    price = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, days)))
    vintage = pd.Timestamp("2026-10-12 08:30", tz="UTC")
    panel = pd.DataFrame(
        {"symbol": "BTC", "metric_date": dates, "price_usd": price, "panel_vintage_ts": vintage}
    )
    estimates = pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": dates,
            "method_id": "metcalfe_estimated",
            "fair_value": price * np.exp(rng.normal(0.0, 1.0, days)),
            "applicable": True,
        }
    )
    settings = Settings(data_root=str(tmp_path))
    monkeypatch.setattr(cli, "_read_score_inputs", lambda _settings: (panel, estimates))
    report = cli.run_score(settings, [], ["naive_persistence", "feedforward_nn"], [1])

    written = pd.read_parquet(
        pathlib.Path(settings.resolver.table("gold", "model_score")) / "part-0.parquet"
    )
    rows = written[written["model_id"] == "feedforward_nn"]
    holm = rows[rows["metric"] == "dm_p_value_holm"]["value"].to_numpy()
    stat = rows[rows["metric"] == "dm_statistic"]["value"].to_numpy()
    assert ((holm < 0.05) & (stat > 0)).sum() >= 1  # not vacuous: a real significant loss
    assert report["feedforward_nn"]["beat_baseline_after_holm"] == 0
    assert report["feedforward_nn"]["lost_to_baseline_after_holm"] == int(
        ((holm < 0.05) & (stat > 0)).sum()
    )


def test_each_significant_holm_row_takes_its_own_dm_statistic(tmp_path, monkeypatch):
    """The sign must come from a KEYED join. The significant cells are not always a prefix
    of the tested ones, so lining the two up positionally gives one cell's verdict another
    cell's sign -- the mis-attribution that printed "beats" for losses in the live run.
    Crafted rather than fitted: in every end-to-end fixture every cell is significant, and
    there positional and keyed alignment cannot be told apart.
    """
    from cryptoval.scoring.runner import SCORE_COLUMNS

    def _crafted_row(horizon, metric, value):
        row = dict.fromkeys(SCORE_COLUMNS)
        row.update(
            model_id="ols",
            symbol="BTC",
            horizon=horizon,
            fold=-1,
            metric=metric,
            value=value,
            panel_vintage_ts=pd.Timestamp("2026-10-12 08:30", tz="UTC"),
        )
        return row

    # h=30 is NOT significant and its statistic is positive; h=90 is significant and beat
    # persistence. Only the h=90 row is a verdict, and its sign is -3.0, not 5.0.
    crafted = pd.DataFrame(
        [
            _crafted_row(30, "dm_p_value_holm", 0.9),
            _crafted_row(90, "dm_p_value_holm", 0.01),
            _crafted_row(30, "dm_statistic", 5.0),
            _crafted_row(90, "dm_statistic", -3.0),
        ]
    )[SCORE_COLUMNS]

    settings = Settings(data_root=str(tmp_path))
    monkeypatch.setattr(cli, "_read_score_inputs", lambda _settings: _score_inputs())
    monkeypatch.setattr(cli, "correct_family_wise", lambda scores, **kwargs: crafted)
    report = cli.run_score(settings, [], ["naive_persistence"], [30])

    assert report["ols"]["beat_baseline_after_holm"] == 1
    assert report["ols"]["lost_to_baseline_after_holm"] == 0


def _econ_inputs(days=1200, symbols=("BTC",), seed=4):
    rng = np.random.default_rng(seed)
    out = []
    for symbol in symbols:
        trend = np.cumsum(rng.normal(0.002, 0.01, days)) + 20.0
        out.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "metric_date": pd.date_range("2019-01-01", periods=days, freq="D"),
                    "market_cap_usd": np.exp(trend + rng.normal(0.0, 0.05, days)),
                    "realized_cap": np.exp(trend),
                    "volume_total_usd": np.exp(trend - 3.0),
                    "active_addresses": np.exp(trend / 2.0),
                    "panel_vintage_ts": pd.Timestamp("2026-10-12 08:30", tz="UTC"),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def test_econ_writes_the_documented_columns_to_gold(tmp_path, monkeypatch):
    from cryptoval.econometrics.runner import TEST_COLUMNS

    settings = Settings(data_root=str(tmp_path))
    levels_panel = _econ_inputs()
    monkeypatch.setattr(cli, "_read_econ_inputs", lambda _settings: levels_panel)
    report = cli.run_econ(settings, [])

    written = pd.read_parquet(
        pathlib.Path(settings.resolver.table("gold", "econometric_test")) / "part-0.parquet"
    )
    assert list(written.columns) == TEST_COLUMNS
    assert (written["metric"] == "hypotheses_tested").any()
    assert report["mvrv_anchor"]["rows"] > 0


def test_econ_refuses_an_unknown_pair_at_the_cli(capsys):
    assert cli.main(["econ", "--pairs", "not_a_pair"]) == 2
    assert "unknown pair" in capsys.readouterr().err
