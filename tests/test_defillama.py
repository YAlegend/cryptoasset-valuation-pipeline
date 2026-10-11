import datetime as dt
import time

import httpx
import numpy as np
import pandas as pd
import pytest

from cryptoval.sources.defillama import (
    BASE_URL,
    OVERVIEW_PATH,
    REVENUE_PATH,
    DefiLlamaSource,
    methodology_hash,
    parse_fees_overview,
)


def test_parses_protocols(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    assert len(df) == 3
    assert set(df["protocol"]) == {"Curve DEX", "Aave V2", "WBTC"}


def test_the_fees_payload_carries_no_revenue_column(load_fixture):
    """It never did: net revenue lives behind dataType=dailyRevenue (Ruling G)."""
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    assert "net_revenue_24h" not in df.columns
    assert df[df["protocol"] == "Curve DEX"].iloc[0]["total_fees_24h"] == 7415


def test_methodology_hash_is_stable_across_key_order():
    a = {"methodology": {"Fees": "x", "Revenue": "y"}}
    b = {"methodology": {"Revenue": "y", "Fees": "x"}}
    assert methodology_hash(a) == methodology_hash(b)


def test_methodology_hash_changes_when_the_text_changes():
    a = {"methodology": {"Fees": "Trading fees paid by users"}}
    b = {"methodology": {"Fees": "Trading fees plus incentives"}}
    assert methodology_hash(a) != methodology_hash(b)


def test_empty_methodology_still_hashes(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    wbtc = df[df["protocol"] == "WBTC"].iloc[0]
    assert len(wbtc["methodology_hash"]) == 64


def test_source_metadata():
    s = DefiLlamaSource()
    assert s.name == "defillama"
    assert s.contract_path.endswith("defillama_fees.yml")


START = dt.date(2026, 10, 10)
END = dt.date(2026, 10, 11)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _respond(request, status, json_body=None):
    if json_body is None:
        return httpx.Response(status, request=request)
    return httpx.Response(status, json=json_body, request=request)


@pytest.fixture
def sleeps(monkeypatch):
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


def test_fetch_success_returns_frame_rows_in_and_utc_metric_date(load_fixture, sleeps):
    requested: list[str] = []

    def handler(request):
        requested.append(str(request.url))
        return _two_call_handler(load_fixture)(request)

    with _client(handler) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)

    assert requested == [f"{BASE_URL}{OVERVIEW_PATH}", f"{BASE_URL}{REVENUE_PATH}"]
    assert sleeps == []
    assert result.rows_in == 3
    assert len(result.frame) == 3
    assert set(result.frame["protocol"]) == {"Curve DEX", "Aave V2", "WBTC"}
    stamp = result.frame["metric_date"].iloc[0]
    assert stamp == pd.Timestamp("2026-10-11", tz="UTC")
    assert stamp.tz is not None and str(stamp.tz) == "UTC"
    assert (result.frame["metric_date"] == stamp).all()


def test_fetch_retries_once_on_503_then_succeeds(load_fixture, sleeps):
    requests_seen: list[httpx.Request] = []
    fail_once = {"fees": True}

    def handler(request):
        requests_seen.append(request)
        if "dataType=dailyRevenue" not in str(request.url) and fail_once.pop("fees", False):
            return _respond(request, 503)
        return _two_call_handler(load_fixture)(request)

    with _client(handler) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)

    assert len(requests_seen) == 3  # fees 503, fees 200, revenue 200
    assert sleeps == [1.0]
    assert result.rows_in == 3
    assert len(result.frame) == 3
    assert result.errors == ()


def test_fetch_raises_after_retries_exhausted_with_doubling_backoff(sleeps):
    requests_seen: list[httpx.Request] = []

    def handler(request):
        requests_seen.append(request)
        return _respond(request, 503)

    with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            DefiLlamaSource(max_retries=3).fetch(START, END, [], client)

    assert len(requests_seen) == 3
    assert sleeps == [1.0, 2.0]


# ---- Fix round 1, Ruling G: revenue comes from its own dataType ----------


def _two_call_handler(load_fixture, revenue_status=200):
    fees = load_fixture("defillama_fees_overview.json")
    revenue = load_fixture("defillama_revenue_overview.json")

    def handler(request):
        if "dataType=dailyRevenue" in str(request.url):
            if revenue_status != 200:
                return _respond(request, revenue_status)
            return _respond(request, 200, revenue)
        return _respond(request, 200, fees)

    return handler


def test_net_revenue_comes_from_the_revenue_endpoint_not_the_fees_payload(
    load_fixture, sleeps
):
    """The fees overview carries no revenue24h field at all, so the column was 100% null."""
    requested: list[str] = []

    def handler(request):
        requested.append(str(request.url))
        return _two_call_handler(load_fixture)(request)

    with _client(handler) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)

    assert requested == [f"{BASE_URL}{OVERVIEW_PATH}", f"{BASE_URL}{REVENUE_PATH}"]
    curve = result.frame[result.frame["protocol"] == "Curve DEX"].iloc[0]
    assert curve["total_fees_24h"] == 7415
    assert curve["net_revenue_24h"] == 1483
    assert result.errors == ()


def test_a_protocol_absent_from_the_revenue_feed_keeps_a_null_revenue(load_fixture, sleeps):
    with _client(_two_call_handler(load_fixture)) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)
    aave = result.frame[result.frame["protocol"] == "Aave V2"].iloc[0]
    assert np.isnan(aave["net_revenue_24h"])
    assert aave["total_fees_24h"] == 5266


def test_a_genuine_zero_revenue_stays_zero(load_fixture, sleeps):
    with _client(_two_call_handler(load_fixture)) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)
    wbtc = result.frame[result.frame["protocol"] == "WBTC"].iloc[0]
    assert wbtc["net_revenue_24h"] == 0.0
    assert not np.isnan(wbtc["net_revenue_24h"])


def test_fees_are_kept_when_the_revenue_call_fails_and_the_error_is_reported(
    load_fixture, sleeps
):
    with _client(_two_call_handler(load_fixture, revenue_status=503)) as client:
        result = DefiLlamaSource(max_retries=2).fetch(START, END, [], client)
    assert len(result.frame) == 3
    assert result.frame["total_fees_24h"].tolist() == [7415.0, 5266.0, 0.0]
    assert result.frame["net_revenue_24h"].isna().all()
    assert len(result.errors) == 1
    assert "revenue" in result.errors[0]
