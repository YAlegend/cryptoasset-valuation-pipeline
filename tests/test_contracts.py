import datetime as dt

import pandas as pd
import pytest

from cryptoval.contracts.checks import (
    check_freshness,
    check_not_null,
    check_ranges,
    check_required_columns,
    check_unique,
    check_volume_anomaly,
    run_checks,
)
from cryptoval.contracts.loader import load_contract

CONTRACT = "config/contracts/coinmetrics_asset_metrics.yml"


def _frame(**overrides):
    base = pd.DataFrame(
        {
            "asset": ["btc", "btc"],
            "metric_date": pd.to_datetime(["2026-10-10", "2026-10-11"], utc=True),
            "CapMrktCurUSD": [1.0e12, 1.1e12],
        }
    )
    for k, v in overrides.items():
        base[k] = v
    return base


def test_loads_contract_metadata():
    c = load_contract(CONTRACT)
    assert c.name == "coinmetrics_asset_metrics"
    assert c.version == "1.0.0"
    assert c.unique_on == ["asset", "metric_date"]
    assert c.freshness_max_delay_hours == 48
    assert c.volume_anomaly_threshold == 0.5
    assert "CapMrktCurUSD" in c.required_columns


def test_required_columns_pass_and_fail():
    c = load_contract(CONTRACT)
    assert check_required_columns(_frame(), c).passed
    bad = _frame().drop(columns=["CapMrktCurUSD"])
    res = check_required_columns(bad, c)
    assert not res.passed and "CapMrktCurUSD" in res.detail


def test_uniqueness_detects_duplicate_keys():
    c = load_contract(CONTRACT)
    assert check_unique(_frame(), c).passed
    dup = pd.concat([_frame(), _frame()], ignore_index=True)
    res = check_unique(dup, c)
    assert not res.passed and "2" in res.detail


def test_not_null_on_required_columns():
    c = load_contract(CONTRACT)
    bad = _frame()
    bad.loc[0, "CapMrktCurUSD"] = None
    assert not check_not_null(bad, c).passed


def test_ranges_reject_negative_market_cap():
    c = load_contract(CONTRACT)
    bad = _frame(CapMrktCurUSD=[-1.0, 1.0e12])
    res = check_ranges(bad, c)
    assert not res.passed and "CapMrktCurUSD" in res.detail


def test_freshness_passes_when_recent_and_fails_when_stale():
    c = load_contract(CONTRACT)
    now = dt.datetime(2026, 10, 11, 12, tzinfo=dt.UTC)
    assert check_freshness(_frame(), c, "metric_date", now).passed
    stale = _frame()
    stale["metric_date"] = pd.to_datetime(["2026-09-01", "2026-09-02"], utc=True)
    assert not check_freshness(stale, c, "metric_date", now).passed


def test_volume_anomaly_flags_a_tenfold_jump():
    c = load_contract(CONTRACT)
    steady = pd.DataFrame(
        {
            "metric_date": pd.to_datetime(
                [f"2026-09-{d:02d}" for d in range(1, 21)], utc=True
            ),
            "value": [100.0] * 19 + [1000.0],
        }
    )
    res = check_volume_anomaly(steady, c, "value", "metric_date")
    assert not res.passed
    assert "1000" in res.detail or "anomal" in res.detail.lower()


def test_volume_anomaly_allows_normal_variation():
    c = load_contract(CONTRACT)
    steady = pd.DataFrame(
        {
            "metric_date": pd.to_datetime(
                [f"2026-09-{d:02d}" for d in range(1, 21)], utc=True
            ),
            "value": [100.0, 105.0, 98.0] * 6 + [102.0, 99.0],
        }
    )
    assert check_volume_anomaly(steady, c, "value", "metric_date").passed


def test_run_checks_aggregates_all_results():
    c = load_contract(CONTRACT)
    results = run_checks(_frame(), c, now=dt.datetime(2026, 10, 11, 12, tzinfo=dt.UTC))
    assert len(results) >= 4
    assert all(r.passed for r in results)


def test_unknown_contract_path_raises():
    with pytest.raises(FileNotFoundError):
        load_contract("config/contracts/does_not_exist.yml")
