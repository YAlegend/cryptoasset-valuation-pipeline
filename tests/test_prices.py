import datetime as dt
import time

import httpx
import numpy as np
import pandas as pd
import pytest
import yfinance

from cryptoval.config import Asset
from cryptoval.sources.prices import (
    CoinGeckoSource,
    EquitySource,
    compute_velocity,
    parse_market_chart,
)

PANEL = ["asset", "metric_date", "price_usd", "market_cap_usd", "volume_usd"]


def test_parses_three_series_into_one_panel(load_fixture):
    df = parse_market_chart(load_fixture("coingecko_btc_market_chart.json"), "bitcoin")
    assert len(df) == 3
    assert list(df.columns) == PANEL


def test_asset_is_the_gecko_id(load_fixture):
    df = parse_market_chart(load_fixture("coingecko_btc_market_chart.json"), "bitcoin")
    assert set(df["asset"]) == {"bitcoin"}


def test_epoch_millis_become_utc_dates(load_fixture):
    df = parse_market_chart(load_fixture("coingecko_btc_market_chart.json"), "bitcoin")
    assert str(df["metric_date"].dt.tz) == "UTC"
    assert (df["metric_date"] == df["metric_date"].dt.normalize()).all()


def test_values_align_across_the_three_series(load_fixture):
    df = parse_market_chart(load_fixture("coingecko_btc_market_chart.json"), "bitcoin")
    first = df.sort_values("metric_date").iloc[0]
    assert first["price_usd"] == pytest.approx(85540.08)
    assert first["market_cap_usd"] == pytest.approx(1.71888314248e12)
    assert first["volume_usd"] == pytest.approx(4.5e10)


def test_velocity_is_volume_over_market_cap():
    df = pd.DataFrame({"market_cap_usd": [1000.0], "volume_usd": [50.0]})
    out = compute_velocity(df)
    assert out["velocity"].iloc[0] == pytest.approx(0.05)
    assert out["holding_time_days"].iloc[0] == pytest.approx(20.0)


def test_zero_market_cap_gives_nan_velocity_not_inf():
    df = pd.DataFrame({"market_cap_usd": [0.0], "volume_usd": [50.0]})
    out = compute_velocity(df)
    assert np.isnan(out["velocity"].iloc[0])


def test_zero_volume_gives_nan_holding_time_not_inf():
    df = pd.DataFrame({"market_cap_usd": [1000.0], "volume_usd": [0.0]})
    out = compute_velocity(df)
    assert np.isnan(out["holding_time_days"].iloc[0])


def test_source_metadata():
    assert CoinGeckoSource().name == "coingecko"
    assert CoinGeckoSource().contract_path.endswith("coingecko_market_chart.yml")
    assert EquitySource().name == "equities"


# ---- CoinGeckoSource.fetch ------------------------------------------------

START = dt.date(2026, 10, 1)
END = dt.date(2026, 10, 3)


def _asset(symbol: str, gecko_id: str | None) -> Asset:
    return Asset(symbol=symbol, gecko_id=gecko_id)


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


def test_fetch_skips_assets_without_gecko_id_and_makes_no_request(load_fixture, sleeps):
    requests: list[httpx.Request] = []
    client = _client(
        lambda r: httpx.Response(200, json=load_fixture("coingecko_btc_market_chart.json")),
        requests,
    )
    result = CoinGeckoSource().fetch(START, END, [_asset("XXX", None)], client)
    assert len(requests) == 0
    assert sleeps == []
    assert result.rows_in == 0
    assert result.frame.empty
    assert list(result.frame.columns) == PANEL


def test_fetch_two_assets_makes_two_requests_and_returns_both(load_fixture, sleeps):
    requests: list[httpx.Request] = []
    client = _client(
        lambda r: httpx.Response(200, json=load_fixture("coingecko_btc_market_chart.json")),
        requests,
    )
    assets = [_asset("BTC", "bitcoin"), _asset("SOL", None), _asset("ETH", "ethereum")]
    result = CoinGeckoSource(sleep_between_assets_s=7.0).fetch(START, END, assets, client)
    assert len(requests) == 2
    assert [r.url.path for r in requests] == [
        "/api/v3/coins/bitcoin/market_chart/range",
        "/api/v3/coins/ethereum/market_chart/range",
    ]
    assert set(result.frame["asset"]) == {"bitcoin", "ethereum"}
    assert len(result.frame) == 6
    assert result.rows_in == 6
    assert sleeps == [7.0, 7.0]
    assert "velocity" in result.frame.columns
    assert "holding_time_days" in result.frame.columns


def test_fetch_sends_utc_epoch_second_bounds(load_fixture, sleeps):
    requests: list[httpx.Request] = []
    client = _client(
        lambda r: httpx.Response(200, json=load_fixture("coingecko_btc_market_chart.json")),
        requests,
    )
    CoinGeckoSource().fetch(START, END, [_asset("BTC", "bitcoin")], client)
    params = requests[0].url.params
    assert params["vs_currency"] == "usd"
    assert params["from"] == str(int(dt.datetime(2026, 10, 1, tzinfo=dt.UTC).timestamp()))
    assert params["from"] == "1790812800"
    # end of the end day, inclusive
    assert params["to"] == str(int(dt.datetime(2026, 10, 3, 23, 59, 59, tzinfo=dt.UTC).timestamp()))
    assert params["to"] == "1791071999"


def test_fetch_retries_429_then_succeeds_with_pinned_sleeps(load_fixture, sleeps):
    requests: list[httpx.Request] = []
    responses = iter(
        [
            httpx.Response(429),
            httpx.Response(200, json=load_fixture("coingecko_btc_market_chart.json")),
        ]
    )
    client = _client(lambda r: next(responses), requests)
    result = CoinGeckoSource(sleep_between_assets_s=7.0).fetch(
        START, END, [_asset("BTC", "bitcoin")], client
    )
    assert len(requests) == 2
    # 2.0s retry backoff, then the 7.0s inter-asset pause
    assert sleeps == [2.0, 7.0]
    assert len(result.frame) == 3


def test_fetch_raises_after_exhausting_retries_on_429(sleeps):
    requests: list[httpx.Request] = []
    client = _client(lambda r: httpx.Response(429), requests)
    with pytest.raises(httpx.HTTPStatusError):
        CoinGeckoSource(max_retries=3).fetch(START, END, [_asset("BTC", "bitcoin")], client)
    assert len(requests) == 3
    assert sleeps == [2.0, 4.0]


# ---- EquitySource.fetch ---------------------------------------------------


def _equity(symbol: str, yahoo_id: str | None, is_equity: bool = True) -> Asset:
    return Asset(symbol=symbol, yahoo_id=yahoo_id, is_equity=is_equity)


def _yf_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2026-10-01", "2026-10-02", "2026-10-05"], name="Date")
    columns = pd.MultiIndex.from_product(
        [["Close", "Volume"], ["AAPL", "SPY"]], names=["Price", "Ticker"]
    )
    data = [
        [250.0, 600.0, 1e6, 2e6],
        [np.nan, 605.5, 1e6, 2e6],
        [252.5, 607.0, 1e6, 2e6],
    ]
    return pd.DataFrame(data, index=index, columns=columns)


def test_equity_fetch_tidies_closes_and_drops_null_closes(monkeypatch):
    calls: list[dict] = []

    def fake_download(tickers, **kwargs):
        calls.append({"tickers": tickers, **kwargs})
        return _yf_frame()

    monkeypatch.setattr(yfinance, "download", fake_download)
    assets = [
        _equity("AAPL", "AAPL"),
        _equity("SPY", "SPY"),
        _equity("BTC", None, is_equity=False),
        _equity("NOYAHOO", None),
    ]
    result = EquitySource().fetch(START, END, assets, client=None)

    assert len(calls) == 1
    assert calls[0]["tickers"] == ["AAPL", "SPY"]
    frame = result.frame
    assert list(frame.columns) == PANEL
    assert str(frame["metric_date"].dt.tz) == "UTC"
    assert (frame["metric_date"] == frame["metric_date"].dt.normalize()).all()
    # 3 dates x 2 tickers = 6, minus the single null AAPL close
    assert len(frame) == 5
    assert result.rows_in == 5
    assert frame["price_usd"].notna().all()
    assert set(frame["asset"]) == {"AAPL", "SPY"}
    aapl = frame[frame["asset"] == "AAPL"].sort_values("metric_date")
    assert aapl["price_usd"].tolist() == [250.0, 252.5]
    assert pd.Timestamp("2026-10-02", tz="UTC") not in set(aapl["metric_date"])
    spy = frame[frame["asset"] == "SPY"].sort_values("metric_date")
    assert spy["price_usd"].tolist() == [600.0, 605.5, 607.0]
    assert frame["market_cap_usd"].isna().all()
    assert frame["volume_usd"].isna().all()


def test_equity_fetch_with_no_tickers_returns_empty_frame_without_download(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("yfinance.download must not be called")

    monkeypatch.setattr(yfinance, "download", boom)
    result = EquitySource().fetch(START, END, [_equity("BTC", None, is_equity=False)], client=None)
    assert result.rows_in == 0
    assert result.frame.empty
    assert list(result.frame.columns) == PANEL
