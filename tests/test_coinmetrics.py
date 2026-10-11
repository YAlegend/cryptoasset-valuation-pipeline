import numpy as np
import pandas as pd
import pytest

from cryptoval.sources.coinmetrics import (
    FREE_METRICS,
    CoinMetricsSource,
    derive_realized_cap,
    parse_timeseries,
)


def test_free_metrics_include_the_load_bearing_series():
    for m in ("CapMrktCurUSD", "CapMVRVCur", "HashRate", "IssTotUSD", "FeeTotNtv", "AdrActCnt"):
        assert m in FREE_METRICS


def test_free_metrics_exclude_paid_realized_cap():
    assert "CapRealUSD" not in FREE_METRICS


def test_parses_payload_into_a_typed_panel(load_fixture):
    df = parse_timeseries(load_fixture("coinmetrics_btc_eth.json"))
    assert len(df) == 3
    assert set(df["asset"]) == {"btc", "eth"}
    assert pd.api.types.is_numeric_dtype(df["CapMrktCurUSD"])
    assert df["CapMrktCurUSD"].iloc[0] == pytest.approx(1.725e12)


def test_metric_date_is_timezone_aware_utc(load_fixture):
    df = parse_timeseries(load_fixture("coinmetrics_btc_eth.json"))
    assert str(df["metric_date"].dt.tz) == "UTC"
    assert df["metric_date"].iloc[0] == pd.Timestamp("2026-10-05", tz="UTC")


def test_derives_realized_cap_from_the_mvrv_identity(load_fixture):
    df = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_btc_eth.json")))
    btc = df[(df["asset"] == "btc") & (df["metric_date"] == pd.Timestamp("2026-10-05", tz="UTC"))]
    assert btc["realized_cap"].iloc[0] == pytest.approx(1.725e12 / 1.5967)


def test_zero_mvrv_yields_nan_not_inf(load_fixture):
    df = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_null_metrics.json")))
    row = df[df["asset"] == "xyz"]
    assert np.isnan(row["realized_cap"].iloc[0])
    assert not np.isinf(row["realized_cap"].iloc[0])


def test_negative_mvrv_yields_nan(load_fixture):
    df = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_null_metrics.json")))
    row = df[df["asset"] == "neg"]
    assert np.isnan(row["realized_cap"].iloc[0])


def test_null_mvrv_yields_nan(load_fixture):
    df = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_null_metrics.json")))
    row = df[df["asset"] == "eth"]
    assert np.isnan(row["realized_cap"].iloc[0])


def test_null_metrics_are_not_coerced_to_zero(load_fixture):
    df = parse_timeseries(load_fixture("coinmetrics_null_metrics.json"))
    eth = df[df["asset"] == "eth"]
    assert eth["HashRate"].isna().all()
    assert (eth["HashRate"] == 0).sum() == 0


def test_missing_metric_column_is_absent_rather_than_fabricated(load_fixture):
    df = parse_timeseries(load_fixture("coinmetrics_null_metrics.json"))
    assert "AdrActCnt" not in df.columns or df["AdrActCnt"].isna().all()


def test_source_exposes_name_and_contract_path():
    s = CoinMetricsSource()
    assert s.name == "coinmetrics"
    assert s.contract_path.endswith("coinmetrics_asset_metrics.yml")


def test_derive_realized_cap_is_idempotent(load_fixture):
    once = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_btc_eth.json")))
    twice = derive_realized_cap(once)
    pd.testing.assert_series_equal(once["realized_cap"], twice["realized_cap"])
