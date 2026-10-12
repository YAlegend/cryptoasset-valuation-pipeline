import datetime as dt
import time

import httpx
import numpy as np
import pandas as pd
import pytest

from cryptoval.config import Asset
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
    # The fixture never carries AdrActCnt, so the column must genuinely be absent.
    assert "AdrActCnt" not in df.columns


def test_source_exposes_name_and_contract_path():
    s = CoinMetricsSource()
    assert s.name == "coinmetrics"
    assert s.contract_path.endswith("coinmetrics_asset_metrics.yml")


def test_derive_realized_cap_is_idempotent(load_fixture):
    once = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_btc_eth.json")))
    twice = derive_realized_cap(once)
    pd.testing.assert_series_equal(once["realized_cap"], twice["realized_cap"])


# --- fetch(): driven through httpx.MockTransport, no network -------------------------------

START, END = dt.date(2026, 10, 5), dt.date(2026, 10, 6)


def _asset(symbol: str, cm_id: str | None) -> Asset:
    return Asset(
        symbol=symbol,
        sector=None,
        consensus="pow",
        cm_id=cm_id,
        gecko_id=None,
        yahoo_id=None,
        is_stablecoin=False,
        is_equity=False,
    )


def _row(asset: str, day: str) -> dict:
    return {
        "asset": asset,
        "time": f"{day}T00:00:00.000000000Z",
        "CapMrktCurUSD": "1000",
        "CapMVRVCur": "2",
    }


def _client(handler, requests: list[httpx.Request]) -> httpx.Client:
    def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(recording))


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(time, "sleep", calls.append)
    return calls


def test_fetch_skips_assets_without_cm_id_and_makes_no_request():
    requests: list[httpx.Request] = []
    client = _client(lambda r: httpx.Response(200, json={"data": []}), requests)
    result = CoinMetricsSource().fetch(START, END, [_asset("SOL", None)], client)
    assert len(requests) == 0
    assert result.rows_in == 0
    assert result.frame.empty
    assert list(result.frame.columns) == ["asset", "metric_date"]


def test_fetch_requests_only_resolvable_ids():
    requests: list[httpx.Request] = []
    client = _client(lambda r: httpx.Response(200, json={"data": []}), requests)
    CoinMetricsSource().fetch(START, END, [_asset("SOL", None), _asset("BTC", "btc")], client)
    assert len(requests) == 1
    assert requests[0].url.params["assets"] == "btc"


def test_fetch_follows_next_page_url_and_stops():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json={"data": [_row("btc", "2026-10-06")]})
        return httpx.Response(
            200,
            json={
                "data": [_row("btc", "2026-10-05")],
                "next_page_url": "https://community-api.coinmetrics.io/v4/next?page=2",
            },
        )

    result = CoinMetricsSource().fetch(
        START, END, [_asset("BTC", "btc")], _client(handler, requests)
    )
    assert len(requests) == 2
    assert "assets" not in requests[1].url.params  # params reset after the first page
    assert result.rows_in == 2
    assert list(result.frame["metric_date"]) == [
        pd.Timestamp("2026-10-05", tz="UTC"),
        pd.Timestamp("2026-10-06", tz="UTC"),
    ]
    assert "realized_cap" in result.frame.columns


def test_fetch_retries_429_then_succeeds(sleeps):
    requests: list[httpx.Request] = []
    responses = iter(
        [httpx.Response(429), httpx.Response(200, json={"data": [_row("btc", "2026-10-05")]})]
    )
    client = _client(lambda r: next(responses), requests)
    result = CoinMetricsSource().fetch(START, END, [_asset("BTC", "btc")], client)
    assert len(requests) == 2
    assert result.rows_in == 1
    assert len(result.frame) == 1
    assert sleeps == [1.0]


def test_fetch_raises_after_exhausting_retries_on_429(sleeps):
    requests: list[httpx.Request] = []
    client = _client(lambda r: httpx.Response(429), requests)
    with pytest.raises(httpx.HTTPStatusError):
        CoinMetricsSource(max_retries=3).fetch(START, END, [_asset("BTC", "btc")], client)
    assert len(requests) == 3
    assert sleeps == [1.0, 2.0]


def test_fetch_raises_on_self_referential_next_page_url():
    requests: list[httpx.Request] = []
    loop_url = "https://community-api.coinmetrics.io/v4/loop?page=1"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [], "next_page_url": loop_url})

    with pytest.raises(RuntimeError, match=r"1000 pages.*loop\?page=1"):
        CoinMetricsSource().fetch(START, END, [_asset("BTC", "btc")], _client(handler, requests))
    assert len(requests) == 1000


def test_fetch_fails_fast_on_403_without_retry(sleeps):
    requests: list[httpx.Request] = []
    client = _client(lambda r: httpx.Response(403), requests)
    with pytest.raises(httpx.HTTPStatusError):
        CoinMetricsSource().fetch(START, END, [_asset("BTC", "btc")], client)
    assert len(requests) == 1
    assert sleeps == []


def test_fetch_requests_reported_spot_volume(load_fixture):
    """Ruling B: the free, full-history equation-of-exchange T term."""
    requests: list[httpx.Request] = []
    client = _client(lambda r: httpx.Response(200, json={"data": []}), requests)
    CoinMetricsSource().fetch(START, END, [_asset("BTC", "btc")], client)
    assert "volume_reported_spot_usd_1d" in requests[0].url.params["metrics"].split(",")


def test_reported_spot_volume_parses_as_a_number_and_keeps_its_nulls(load_fixture):
    """The fixture lagged the API after Ruling B added the metric, so nothing offline
    ever saw it traverse the parser — the same fixture-honesty gap that hid Ruling G."""
    df = parse_timeseries(load_fixture("coinmetrics_btc_eth.json"))
    assert pd.api.types.is_numeric_dtype(df["volume_reported_spot_usd_1d"])
    btc = df[df["asset"] == "btc"].sort_values("metric_date")
    assert btc["volume_reported_spot_usd_1d"].iloc[0] == pytest.approx(2.2819828752e10)
    eth = df[df["asset"] == "eth"]
    assert eth["volume_reported_spot_usd_1d"].isna().all()


def test_fetch_requests_every_configured_vendor_id():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"data": [
            {"asset": "avaxc", "time": "2026-10-01T00:00:00.000000000Z", "PriceUSD": "30.0"},
        ]})

    assets = [
        Asset(symbol="AVAX", sector="smart_contract_l1", consensus="pos", cm_id="avaxc",
              gecko_id="avalanche-2", metric_sources={"volume": "cm:avax"}),
        Asset(symbol="BTC", sector="pow_monetary", consensus="pow", cm_id="btc"),
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        CoinMetricsSource().fetch(dt.date(2026, 10, 1), dt.date(2026, 10, 2), assets, client)

    # Sorted union, so avax rides along with avaxc rather than replacing it.
    assert captured["assets"] == "avax,avaxc,btc"


def test_status_sidecar_columns_are_dropped():
    payload = {"data": [{
        "asset": "btc",
        "time": "2026-10-01T00:00:00.000000000Z",
        "FlowInExUSD": "1000.0",
        "FlowInExUSD-status": "unavailable",
        "FlowInExUSD-status-time": "2026-10-01T00:00:00.000000000Z",
        "FlowOutExUSD-status": "unavailable",
    }]}
    df = parse_timeseries(payload)
    assert [c for c in df.columns if "status" in c] == []
    # The metric itself survives; only the sidecars go.
    assert df["FlowInExUSD"].iloc[0] == 1000.0
