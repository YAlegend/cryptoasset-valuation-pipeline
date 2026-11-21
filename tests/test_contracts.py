import datetime as dt

import pandas as pd
import pytest

from cryptoval.contracts.checks import (
    check_freshness,
    check_not_null,
    check_null_rate,
    check_ranges,
    check_required_columns,
    check_unique,
    check_volume_anomaly,
    run_checks,
)
from cryptoval.contracts.loader import load_contract

CONTRACT = "config/contracts/coinmetrics_asset_metrics.yml"
NOW = dt.datetime(2026, 10, 11, 12, tzinfo=dt.UTC)


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
    assert c.volume_anomaly_threshold == 5.0
    assert c.required_columns == ["asset", "metric_date"]


def test_required_columns_pass_and_fail():
    c = load_contract(CONTRACT)
    assert check_required_columns(_frame(), c).passed
    bad = _frame().drop(columns=["metric_date"])
    res = check_required_columns(bad, c)
    assert not res.passed and "metric_date" in res.detail


def test_uniqueness_detects_duplicate_keys():
    c = load_contract(CONTRACT)
    assert check_unique(_frame(), c).passed
    dup = pd.concat([_frame(), _frame()], ignore_index=True)
    res = check_unique(dup, c)
    assert not res.passed and "2" in res.detail


def test_not_null_on_required_columns():
    c = load_contract(CONTRACT)
    bad = _frame()
    bad.loc[0, "asset"] = None
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
            "metric_date": pd.to_datetime([f"2026-09-{d:02d}" for d in range(1, 21)], utc=True),
            "value": [100.0] * 19 + [1000.0],
        }
    )
    res = check_volume_anomaly(steady, c, "value", "metric_date")
    assert not res.passed
    assert "value=1000" in res.detail


def test_volume_anomaly_allows_normal_variation():
    c = load_contract(CONTRACT)
    steady = pd.DataFrame(
        {
            "metric_date": pd.to_datetime([f"2026-09-{d:02d}" for d in range(1, 21)], utc=True),
            "value": [100.0, 105.0, 98.0] * 6 + [102.0, 99.0],
        }
    )
    assert check_volume_anomaly(steady, c, "value", "metric_date").passed


def _long_series(n_days, latest, history=100.0):
    return pd.DataFrame(
        {
            "metric_date": pd.date_range("2026-01-01", periods=n_days, tz="UTC"),
            "value": [history] * (n_days - 1) + [latest],
        }
    )


def test_volume_anomaly_judges_only_the_newest_observation():
    """A years-old event must not be re-reported on every validate: a +100x day early in a
    long history is not a signal about today's data."""
    c = load_contract(CONTRACT)
    df = _long_series(400, latest=100.0)
    df.loc[50, "value"] = 10_000.0  # a fork-day-scale spike, long ago
    assert check_volume_anomaly(df, c, "value", "metric_date").passed


def test_volume_anomaly_ignores_a_routine_move_but_flags_a_fork_day_spike():
    """The documented events are +246x (BCH), +102x (DOGE), +89x (MANA); a 62% day-on-day
    move (BTC's measured latest) is ordinary. Both halves, so the fix is not mere silence."""
    c = load_contract(CONTRACT)
    routine = _long_series(60, latest=38.0)  # a 62% fall, like BTC on 2026-10-11
    assert check_volume_anomaly(routine, c, "value", "metric_date").passed
    assert check_volume_anomaly(_long_series(60, latest=160.0), c, "value", "metric_date").passed

    spike = _long_series(60, latest=100.0 * 101)  # +100x
    res = check_volume_anomaly(spike, c, "value", "metric_date")
    assert not res.passed
    assert "value=10100" in res.detail and "2026-03-01" in res.detail


def test_volume_anomaly_is_independent_of_stored_row_order():
    c = load_contract(CONTRACT)
    spike = _long_series(60, latest=100.0 * 101)
    reversed_spike = spike.iloc[::-1]  # stored newest-first, as silver need not be sorted
    assert not check_volume_anomaly(reversed_spike, c, "value", "metric_date").passed
    reversed_calm = _long_series(60, latest=38.0).iloc[::-1]
    assert check_volume_anomaly(reversed_calm, c, "value", "metric_date").passed


def test_run_checks_aggregates_all_results():
    c = load_contract(CONTRACT)
    results = run_checks(_frame(), c, now=dt.datetime(2026, 10, 11, 12, tzinfo=dt.UTC))
    assert len(results) >= 4
    assert all(r.passed for r in results)


def test_unknown_contract_path_raises():
    with pytest.raises(FileNotFoundError):
        load_contract("config/contracts/does_not_exist.yml")


def test_volume_anomaly_zero_trailing_mean_passes_with_reason():
    c = load_contract(CONTRACT)
    zeros = pd.DataFrame(
        {
            "metric_date": pd.to_datetime([f"2026-09-{d:02d}" for d in range(1, 21)], utc=True),
            "value": [0.0] * 20,
        }
    )
    res = check_volume_anomaly(zeros, c, "value", "metric_date")
    assert res.passed and "baseline" in res.detail


def test_ranges_fail_on_unparseable_values():
    c = load_contract(CONTRACT)
    bad = _frame(CapMrktCurUSD=["N/A", 1.0e12])
    res = check_ranges(bad, c)
    assert not res.passed and "CapMrktCurUSD" in res.detail and "N/A" in res.detail


def _write(tmp_path, text):
    p = tmp_path / "c.yml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_typo_in_quality_section_raises(tmp_path):
    with pytest.raises(ValueError, match="uniquness"):
        load_contract(_write(tmp_path, "quality:\n  uniquness: {combination: [a]}\n"))


def test_typo_in_quality_subkey_raises(tmp_path):
    with pytest.raises(ValueError, match="max_delay_hour"):
        load_contract(_write(tmp_path, "quality:\n  freshness: {max_delay_hour: 3}\n"))


def test_typo_in_bound_key_raises(tmp_path):
    text = "schema:\n  columns:\n    x: {type: float, minimumm: 0}\n"
    with pytest.raises(ValueError, match="minimumm"):
        load_contract(_write(tmp_path, text))


def test_non_numeric_bound_raises(tmp_path):
    text = "schema:\n  columns:\n    x: {type: float, minimum: abc}\n"
    with pytest.raises(ValueError, match="numeric"):
        load_contract(_write(tmp_path, text))


def test_quality_notes_allowed(tmp_path):
    c = load_contract(_write(tmp_path, "quality:\n  notes: free text\n"))
    assert c.unique_on == []


def test_unknown_top_level_key_raises(tmp_path):
    with pytest.raises(ValueError, match="qualty"):
        load_contract(_write(tmp_path, "qualty:\n  freshness: {max_delay_hours: 3}\n"))


def test_unknown_schema_key_raises(tmp_path):
    with pytest.raises(ValueError, match="requried"):
        load_contract(_write(tmp_path, "schema:\n  requried: [a]\n"))


def test_string_required_raises(tmp_path):
    with pytest.raises(ValueError, match="schema.required"):
        load_contract(_write(tmp_path, "schema:\n  required: asset\n"))


def test_string_combination_raises(tmp_path):
    with pytest.raises(ValueError, match="combination"):
        load_contract(_write(tmp_path, "quality:\n  uniqueness: {combination: id}\n"))


def test_non_mapping_column_spec_raises(tmp_path):
    text = "schema:\n  columns:\n    mcap: float\n"
    with pytest.raises(ValueError, match="schema.columns.mcap"):
        load_contract(_write(tmp_path, text))


# ---- Fix round 1: contract calibration against live vendor behaviour -----

COINGECKO_CONTRACT = "config/contracts/coingecko_market_chart.yml"
DEFILLAMA_CONTRACT = "config/contracts/defillama_fees.yml"


def test_coingecko_contract_carries_a_volume_anomaly_threshold():
    """Ruling C: the wired guard checks nothing without a threshold."""
    assert load_contract(COINGECKO_CONTRACT).volume_anomaly_threshold == 5.0


def test_market_cap_is_not_required_because_coin_metrics_omits_it():
    """Ruling E: 6,623 live rows have no CapMrktCurUSD; the vendor simply omits it."""
    c = load_contract(CONTRACT)
    assert "CapMrktCurUSD" not in c.required_columns
    assert c.required_columns == ["asset", "metric_date"]
    missing_cap = _frame()
    missing_cap.loc[0, "CapMrktCurUSD"] = None
    assert check_not_null(missing_cap, c).passed


def test_a_negative_market_cap_still_fails_after_the_not_null_relaxation():
    c = load_contract(CONTRACT)
    assert not check_ranges(_frame(CapMrktCurUSD=[-1.0, 1.0e12]), c).passed


def test_coin_metrics_contract_declares_reported_spot_volume():
    """Ruling B: the free full-history source of the equation-of-exchange T term."""
    assert "volume_reported_spot_usd_1d" in load_contract(CONTRACT).ranges


def test_defillama_fees_are_not_required_but_negative_fees_still_fail():
    """Rulings F and H: 383 protocols report no fees; 12 report negative fees."""
    c = load_contract(DEFILLAMA_CONTRACT)
    assert "total_fees_24h" not in c.required_columns
    frame = pd.DataFrame({"protocol": ["A", "B"], "total_fees_24h": [None, 10.0]})
    assert check_not_null(frame, c).passed
    negative = pd.DataFrame({"protocol": ["A"], "total_fees_24h": [-2803.0]})
    assert not check_ranges(negative, c).passed


# ---- Fix round 1, Rulings J and K: a gate that fires on healthy data is no gate


def _growing(days: int, rate: float) -> pd.DataFrame:
    """A series growing steadily — the normal shape of a decade of crypto volume."""
    return pd.DataFrame(
        {
            "metric_date": pd.date_range("2015-01-01", periods=days, freq="D", tz="UTC"),
            "volume_reported_spot_usd_1d": [1_000_000.0 * (1 + rate) ** i for i in range(days)],
        }
    )


def test_the_volume_baseline_is_a_trailing_window_not_all_of_history():
    """An all-time mean turns secular growth into a permanent anomaly: 68.8% of live
    Coin Metrics asset-days were flagged by the expanding baseline."""
    c = load_contract(CONTRACT)
    steady_growth = _growing(200, 0.01)
    result = check_volume_anomaly(steady_growth, c, "volume_reported_spot_usd_1d", "metric_date")
    assert result.passed, result.detail


def test_the_volume_check_still_flags_a_step_change_against_the_trailing_window():
    c = load_contract(CONTRACT)
    frame = _growing(60, 0.0)
    frame.loc[59, "volume_reported_spot_usd_1d"] = 8.0e52
    result = check_volume_anomaly(frame, c, "volume_reported_spot_usd_1d", "metric_date")
    assert not result.passed
    assert "volume_reported_spot_usd_1d" in result.detail


def test_an_impossible_volume_fails_the_blocking_range_bound():
    """Ruling J: 14 live rows carried volumes up to 7e55 USD. Impossibility blocks;
    merely anomalous values warn."""
    c = load_contract(CONTRACT)
    frame = pd.DataFrame(
        {
            "asset": ["usdc"],
            "metric_date": pd.to_datetime(["2026-07-31"], utc=True),
            "volume_reported_spot_usd_1d": [6.99755e55],
        }
    )
    res = check_ranges(frame, c)
    assert not res.passed
    assert "volume_reported_spot_usd_1d" in res.detail


def test_equities_have_their_own_contract_with_weekend_aware_freshness():
    """Ruling K: equity markets close, so a 48h SLA fails every Monday."""
    c = load_contract("config/contracts/equity_prices.yml")
    assert c.name == "equity_prices"
    assert c.required_columns == ["asset", "metric_date", "price_usd"]
    assert c.freshness_max_delay_hours == 144
    assert c.volume_anomaly_threshold is None
    friday_close = pd.DataFrame(
        {
            "asset": ["AAPL"],
            "metric_date": pd.to_datetime(["2026-10-09"], utc=True),
            "price_usd": [250.0],
            "market_cap_usd": [None],
            "volume_usd": [None],
        }
    )
    monday = dt.datetime(2026, 10, 12, 13, tzinfo=dt.UTC)
    assert check_freshness(friday_close, c, "metric_date", monday).passed
    assert check_not_null(friday_close, c).passed


# ---- Review fix pass: a column that disappears must fail loudly (spec 9.2) ----


def _panel_with_nulls(null_fraction: float, column: str = "CapMVRVCur") -> pd.DataFrame:
    n = 1000
    nulls = int(n * null_fraction)
    return pd.DataFrame(
        {
            "asset": ["btc"] * n,
            "metric_date": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            column: [None] * nulls + [1.5] * (n - nulls),
        }
    )


def test_a_metric_that_vanishes_from_the_feed_fails_the_null_rate_bound():
    """Spec 9.2: 'if it moves to Pro, the contract fails loudly rather than the panel
    going half-null'. Dropping CapMrktCurUSD from `required` (Ruling E) removed the only
    detector, because required_columns drives both presence and nullity."""
    c = load_contract(CONTRACT)
    res = check_null_rate(_panel_with_nulls(1.0), c)
    assert not res.passed
    assert "CapMVRVCur" in res.detail
    assert "100" in res.detail


def test_normal_vendor_sparsity_passes_the_null_rate_bound():
    """Live CapMVRVCur nullity on 2026-10-11 was 11.7%: a gate that fires on that is noise."""
    c = load_contract(CONTRACT)
    assert check_null_rate(_panel_with_nulls(0.117), c).passed


def test_the_null_rate_bound_is_part_of_the_standard_check_run():
    c = load_contract(CONTRACT)
    names = [r.name for r in run_checks(_panel_with_nulls(1.0), c, now=NOW)]
    assert "null_rate" in names
    assert not [r for r in run_checks(_panel_with_nulls(1.0), c, now=NOW) if r.name == "null_rate"][
        0
    ].passed


def test_the_loader_rejects_a_non_numeric_null_fraction(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text(
        "contract: {name: x, version: '1.0.0'}\n"
        "schema:\n  required: [a]\n  columns:\n    a: {type: float, max_null_fraction: most}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_null_fraction"):
        load_contract(str(bad))


def test_defillama_uniqueness_is_keyed_by_protocol_and_date():
    """The snapshot is per-day, so a protocol-only key fails on the second day of
    operation while the SCD2 key is [protocol, metric_date]."""
    c = load_contract(DEFILLAMA_CONTRACT)
    assert c.unique_on == ["protocol", "metric_date"]
    two_days = pd.DataFrame(
        {
            "protocol": ["Curve DEX", "Curve DEX"],
            "metric_date": pd.to_datetime(["2026-10-11", "2026-10-12"], utc=True),
            "total_fees_24h": [50642.0, 51000.0],
        }
    )
    assert check_unique(two_days, c).passed


def test_negative_net_revenue_is_legitimate_but_negative_gross_fees_are_not():
    """Spec 9.2: a protocol can pay out more than it takes in; it cannot collect negative fees.

    Measured on the live feed 2026-10-12: Ensuro (insurance, claims above premiums),
    Hegic (options, writers lost) and Fragment ($1.05M fees, -$852K net revenue from
    rebates) all report negative net revenue as a true description of their day.
    """
    contract = load_contract("config/contracts/defillama_fees.yml")

    legitimate = pd.DataFrame(
        {
            "protocol": ["Ensuro", "Fragment"],
            "metric_date": pd.to_datetime(["2026-10-11", "2026-10-11"], utc=True),
            "total_fees_24h": [0.0, 1053729.0],
            "total_fees_30d": [113662.0, 32315485.0],
            "net_revenue_24h": [-1281.0, -852147.0],
        }
    )
    assert check_ranges(legitimate, contract).passed

    vendor_error = legitimate.copy()
    vendor_error.loc[0, "total_fees_24h"] = -23915.0
    result = check_ranges(vendor_error, contract)
    assert not result.passed
    assert "total_fees_24h" in result.detail


def test_net_revenue_has_no_lower_bound_in_the_contract():
    contract = load_contract("config/contracts/defillama_fees.yml")
    assert "minimum" not in contract.ranges.get("net_revenue_24h", {})
    # Gross columns keep theirs.
    assert contract.ranges["total_fees_24h"]["minimum"] == 0
    assert contract.ranges["total_fees_30d"]["minimum"] == 0
