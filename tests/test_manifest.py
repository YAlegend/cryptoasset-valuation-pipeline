import datetime as dt

from cryptoval.hashing import stable_hash
from cryptoval.manifest import RunManifest
from cryptoval.paths import PathResolver


def test_stable_hash_is_order_independent_for_dicts():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_stable_hash_changes_with_content():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_records_a_successful_run(tmp_path):
    m = RunManifest(PathResolver(str(tmp_path)))
    rec = m.start("coinmetrics", {"start": "2026-10-01"})
    m.finish(rec, rows_in=10, rows_out=10, status="success")
    df = m.read()
    assert len(df) == 1
    assert df.loc[0, "status"] == "success"
    assert df.loc[0, "source"] == "coinmetrics"
    assert df.loc[0, "rows_out"] == 10


def test_records_a_failed_run_with_the_error(tmp_path):
    m = RunManifest(PathResolver(str(tmp_path)))
    rec = m.start("defillama", {})
    m.finish(rec, rows_in=0, rows_out=0, status="failed", error="HTTP 503")
    df = m.read()
    assert df.loc[0, "status"] == "failed"
    assert "503" in df.loc[0, "error"]


def test_appends_across_runs(tmp_path):
    m = RunManifest(PathResolver(str(tmp_path)))
    for _ in range(3):
        rec = m.start("coingecko", {})
        m.finish(rec, rows_in=1, rows_out=1, status="success")
    assert len(m.read()) == 3


def test_monthly_byte_ledger_sums_only_the_requested_month(tmp_path):
    m = RunManifest(PathResolver(str(tmp_path)))
    r1 = m.start("bigquery", {}, started_at=dt.datetime(2026, 10, 5, tzinfo=dt.UTC))
    m.finish(r1, rows_in=0, rows_out=1, status="success", bytes_scanned=100)
    r2 = m.start("bigquery", {}, started_at=dt.datetime(2026, 10, 9, tzinfo=dt.UTC))
    m.finish(r2, rows_in=0, rows_out=1, status="success", bytes_scanned=250)
    r3 = m.start("bigquery", {}, started_at=dt.datetime(2026, 11, 1, tzinfo=dt.UTC))
    m.finish(r3, rows_in=0, rows_out=1, status="success", bytes_scanned=999)
    assert m.bytes_scanned_in_month(2026, 10) == 350
    assert m.bytes_scanned_in_month(2026, 11) == 999


def test_read_on_empty_manifest_returns_empty_frame(tmp_path):
    m = RunManifest(PathResolver(str(tmp_path)))
    df = m.read()
    assert df.empty
    assert "status" in df.columns
