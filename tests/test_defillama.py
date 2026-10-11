import datetime as dt

import httpx
import numpy as np
import pandas as pd
import pytest

from cryptoval.sources import defillama
from cryptoval.sources.defillama import (
    BASE_URL,
    OVERVIEW_PATH,
    DefiLlamaSource,
    methodology_hash,
    parse_fees_overview,
)


def test_parses_protocols(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    assert len(df) == 3
    assert set(df["protocol"]) == {"Curve DEX", "Aave V2", "WBTC"}


def test_keeps_fees_and_net_revenue_as_distinct_columns(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    curve = df[df["protocol"] == "Curve DEX"].iloc[0]
    assert curve["total_fees_24h"] == 7415
    assert curve["net_revenue_24h"] == 1483


def test_missing_net_revenue_is_nan_not_a_fallback_to_gross_fees(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    aave = df[df["protocol"] == "Aave V2"].iloc[0]
    assert np.isnan(aave["net_revenue_24h"])
    assert aave["total_fees_24h"] == 5266


def test_genuine_zero_revenue_stays_zero_not_nan(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    wbtc = df[df["protocol"] == "WBTC"].iloc[0]
    assert wbtc["net_revenue_24h"] == 0.0
    assert not np.isnan(wbtc["net_revenue_24h"])


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
    monkeypatch.setattr(defillama.time, "sleep", recorded.append)
    return recorded


def test_fetch_success_returns_frame_rows_in_and_utc_metric_date(load_fixture, sleeps):
    payload = load_fixture("defillama_fees_overview.json")
    requested: list[str] = []

    def handler(request):
        requested.append(str(request.url))
        return _respond(request, 200, payload)

    with _client(handler) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)

    assert requested == [f"{BASE_URL}{OVERVIEW_PATH}"]
    assert sleeps == []
    assert result.rows_in == 3
    assert len(result.frame) == 3
    assert set(result.frame["protocol"]) == {"Curve DEX", "Aave V2", "WBTC"}
    stamp = result.frame["metric_date"].iloc[0]
    assert stamp == pd.Timestamp("2026-10-11", tz="UTC")
    assert stamp.tz is not None and str(stamp.tz) == "UTC"
    assert (result.frame["metric_date"] == stamp).all()


def test_fetch_retries_once_on_503_then_succeeds(load_fixture, sleeps):
    payload = load_fixture("defillama_fees_overview.json")
    responses = [(503, None), (200, payload)]
    requests_seen: list[httpx.Request] = []

    def handler(request):
        requests_seen.append(request)
        status, body = responses.pop(0)
        return _respond(request, status, body)

    with _client(handler) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)

    assert len(requests_seen) == 2
    assert sleeps == [1.0]
    assert result.rows_in == 3
    assert len(result.frame) == 3


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
