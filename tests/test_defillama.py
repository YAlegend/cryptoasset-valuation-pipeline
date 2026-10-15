import datetime as dt
import time

import httpx
import numpy as np
import pandas as pd
import pytest

from cryptoval.scd2 import merge_scd2
from cryptoval.sources.defillama import (
    BASE_URL,
    OVERVIEW_PATH,
    REVENUE_PATH,
    DefiLlamaHistorySource,
    DefiLlamaSource,
    DefiLlamaTvlSource,
    methodology_hash,
    parse_fee_history,
    parse_fees_overview,
    parse_tvl_history,
    slug_map,
    trim_leading_zeros,
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


def test_net_revenue_comes_from_the_revenue_endpoint_not_the_fees_payload(load_fixture, sleeps):
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


def test_a_failed_revenue_call_fails_the_whole_fetch(load_fixture, sleeps):
    """Review finding 1: returning fees with an all-NULL net_revenue_24h let merge_scd2
    close every good revenue row and publish nulls as the current vintage, with every
    validate check green. Losing one run is strictly better than that."""
    with _client(_two_call_handler(load_fixture, revenue_status=503)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            DefiLlamaSource(max_retries=2).fetch(START, END, [], client)


def test_a_duplicate_protocol_in_the_revenue_feed_is_dropped_rather_than_guessed(
    load_fixture, sleeps
):
    fees = load_fixture("defillama_fees_overview.json")
    revenue = {
        "protocols": [
            {"name": "Curve DEX", "total24h": 1483},
            {"name": "Curve DEX", "total24h": 9999},
            {"name": "WBTC", "total24h": 0},
        ]
    }

    def handler(request):
        if "dataType=dailyRevenue" in str(request.url):
            return _respond(request, 200, revenue)
        return _respond(request, 200, fees)

    with _client(handler) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)

    assert len(result.frame) == 3  # no row multiplication
    curve = result.frame[result.frame["protocol"] == "Curve DEX"].iloc[0]
    assert np.isnan(curve["net_revenue_24h"])  # conflicting values become a null, not a guess
    assert result.frame[result.frame["protocol"] == "WBTC"].iloc[0]["net_revenue_24h"] == 0.0
    assert len(result.errors) == 1
    assert "Curve DEX" in result.errors[0]


def test_revenue_for_a_protocol_with_no_fees_row_is_reported_not_silently_dropped(
    load_fixture, sleeps
):
    """Fees is the row spine, so a revenue-only protocol loses its revenue. Live on
    2026-10-11 the revenue feed was a strict subset of fees (0 orphans), so this is a
    latent gap — but a silent one until it is counted."""
    fees = load_fixture("defillama_fees_overview.json")
    revenue = {
        "protocols": [
            {"name": "Curve DEX", "total24h": 1483},
            {"name": "Ghost Protocol", "total24h": 777},
        ]
    }

    def handler(request):
        if "dataType=dailyRevenue" in str(request.url):
            return _respond(request, 200, revenue)
        return _respond(request, 200, fees)

    with _client(handler) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)

    assert set(result.frame["protocol"]) == {"Curve DEX", "Aave V2", "WBTC"}
    assert len(result.errors) == 1
    assert "Ghost Protocol" in result.errors[0]


def test_slug_map_reads_display_name_to_slug_from_the_overview(load_fixture):
    mapping = slug_map(load_fixture("defillama_overview_slugs.json"))
    assert mapping["Aave V3"] == "aave-v3"
    assert mapping["Sky Lending"] == "sky-lending"
    assert mapping["Ethereum"] == "ethereum"


def test_parse_fee_history_reads_unix_seconds_as_utc_dates(load_fixture):
    frame = parse_fee_history(load_fixture("defillama_summary_fees.json"), "total_fees_usd")
    assert list(frame.columns) == ["protocol", "metric_date", "total_fees_usd"]
    assert frame["metric_date"].dt.tz is not None, "timestamps must be tz-aware UTC"
    assert frame["metric_date"].iloc[0] == pd.Timestamp("2015-08-07", tz="UTC")
    assert frame["protocol"].unique().tolist() == ["Ethereum"]
    # float64 always: an int64 column hashes differently and re-vintages every stored row.
    assert frame["total_fees_usd"].dtype == "float64"
    # Nothing from the unfinished current day survives parsing (Ruling 37).
    assert (frame["metric_date"] < pd.Timestamp.now(tz="UTC").normalize()).all()


def test_parse_fee_history_drops_the_unfinished_current_day():
    """A daily aggregate of a day that has not ended is partial, and the vendor's own
    total24h equals it: ETH read 354,748 mid-day against 404,507 for the day before."""
    today = pd.Timestamp.now(tz="UTC").normalize()
    payload = {
        "name": "X",
        "totalDataChart": [
            [int((today - pd.Timedelta(days=2)).timestamp()), 10],
            [int((today - pd.Timedelta(days=1)).timestamp()), 20],
            [int(today.timestamp()), 7],
        ],
    }
    frame = parse_fee_history(payload, "total_fees_usd")
    assert frame["total_fees_usd"].tolist() == [10.0, 20.0]
    assert frame["metric_date"].max() == today - pd.Timedelta(days=1)


def test_trim_leading_zeros_drops_pre_existence_but_keeps_interior_zeros():
    frame = pd.DataFrame(
        {
            "protocol": "X",
            "metric_date": pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC"),
            "net_revenue_usd": [0.0, 0.0, 12.0, 0.0, 5.0, 0.0],
        }
    )
    trimmed, dropped = trim_leading_zeros(frame, "net_revenue_usd")
    assert dropped == 2
    assert trimmed["metric_date"].iloc[0] == pd.Timestamp("2024-01-03", tz="UTC")
    # The interior zeros are real zero-fee days and must survive.
    assert trimmed["net_revenue_usd"].tolist() == [12.0, 0.0, 5.0, 0.0]


def test_trim_leading_zeros_on_an_all_zero_series_returns_nothing_rather_than_everything():
    frame = pd.DataFrame(
        {
            "protocol": "X",
            "metric_date": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "net_revenue_usd": [0.0, 0.0, 0.0],
        }
    )
    trimmed, dropped = trim_leading_zeros(frame, "net_revenue_usd")
    assert trimmed.empty and dropped == 3


def _eth_asset():
    from cryptoval.config import Asset

    return Asset(symbol="ETH", cm_id="eth", gecko_id="ethereum", defillama_slugs=["Ethereum"])


def test_history_source_fetches_only_configured_protocols(monkeypatch, load_fixture):
    overview = load_fixture("defillama_overview_slugs.json")
    summary = load_fixture("defillama_summary_fees.json")
    called: list[str] = []

    def fake_get(client, url, params=None, **kwargs):
        called.append(url)
        return overview if "/overview/" in url else summary

    monkeypatch.setattr("cryptoval.sources.defillama.get_with_retry", fake_get)
    result = DefiLlamaHistorySource().fetch(
        dt.date(2015, 1, 1), dt.date(2026, 10, 13), [_eth_asset()], client=None
    )

    # One overview call for the slug map, then two per protocol (fees and revenue dataTypes).
    assert sum("/overview/" in u for u in called) == 1
    assert sum("/summary/fees/ethereum" in u for u in called) == 2
    assert not result.frame.empty
    assert list(result.frame.columns) == [
        "protocol",
        "metric_date",
        "total_fees_usd",
        "net_revenue_usd",
        "methodology_hash",
    ]
    # The display name, not the payload's, is what the rollup joins on.
    assert result.frame["protocol"].unique().tolist() == ["Ethereum"]
    assert result.frame["methodology_hash"].notna().all()
    # Aave V3 is in the overview but not configured on any asset: it must not be fetched.
    assert not any("aave-v3" in u for u in called)


def test_history_source_names_a_protocol_whose_slug_is_unknown(monkeypatch, load_fixture):
    from cryptoval.config import Asset

    overview = load_fixture("defillama_overview_slugs.json")
    monkeypatch.setattr(
        "cryptoval.sources.defillama.get_with_retry",
        lambda client, url, params=None, **kw: overview,
    )
    assets = [Asset(symbol="MKR", cm_id="mkr", gecko_id="maker", defillama_slugs=["Maker"])]
    result = DefiLlamaHistorySource().fetch(
        dt.date(2015, 1, 1), dt.date(2026, 10, 13), assets, client=None
    )
    assert any("Maker" in e for e in result.errors), "an unmatched protocol name must be reported"


def test_history_source_metadata():
    s = DefiLlamaHistorySource()
    assert s.name == "defillama_history"
    assert s.contract_path.endswith("defillama_history.yml")


def test_history_source_does_not_call_routine_trims_or_zero_revenue_an_error(
    monkeypatch, load_fixture
):
    """Leading-zero trims and a zero revenue series are routine; an `errors` entry would
    mark every ingest "partial" forever. An untracked FEES series is still named."""
    overview = load_fixture("defillama_overview_slugs.json")
    summary = load_fixture("defillama_summary_fees.json")
    zeros = {**summary, "totalDataChart": [[t, 0] for t, _ in summary["totalDataChart"]]}

    def serve(fees, revenue):
        def fake_get(client, url, params=None, **kwargs):
            if "/overview/" in url:
                return overview
            return revenue if params else fees

        return fake_get

    run = DefiLlamaHistorySource().fetch
    args = (dt.date(2015, 1, 1), dt.date(2026, 10, 13), [_eth_asset()], None)

    monkeypatch.setattr("cryptoval.sources.defillama.get_with_retry", serve(summary, zeros))
    result = run(*args)
    assert result.errors == ()
    assert result.frame["total_fees_usd"].notna().all()
    # Ruling 36: inside the fee span a zero revenue is a MEASUREMENT -- the protocol earned
    # fees and distributed all of them -- so it stays 0.0 rather than becoming a hole.
    assert (result.frame["net_revenue_usd"] == 0.0).all()

    monkeypatch.setattr("cryptoval.sources.defillama.get_with_retry", serve(zeros, summary))
    result = run(*args)
    assert any("Ethereum/total_fees_usd" in e for e in result.errors)
    assert result.frame["total_fees_usd"].isna().all()


def test_revenue_is_trimmed_by_the_fee_series_start_not_its_own():
    """Ruling 36. ETH earned fees from 2015 and burned nothing until EIP-1559, so 2,190 of
    its rows were NULL "unknown" when the truth is "holders received none of it"."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    revenue = pd.DataFrame({"protocol": "X", "metric_date": dates, "net_revenue_usd": 0.0})
    revenue.loc[4:, "net_revenue_usd"] = 7.0
    # Its own first non-zero value would drop four rows; the fee series began on day 2.
    own, dropped_own = trim_leading_zeros(revenue, "net_revenue_usd")
    assert dropped_own == 4
    by_fees, dropped = trim_leading_zeros(revenue, "net_revenue_usd", start_after=dates[1])
    assert dropped == 1
    assert by_fees["net_revenue_usd"].tolist() == [0.0, 0.0, 0.0, 7.0, 7.0]


def test_the_history_fetch_asks_for_revenue_with_the_right_data_type(monkeypatch, load_fixture):
    """The fees fixture used to BE the revenue payload, so one shape served both calls and
    a swapped dataType was undetectable."""
    overview = load_fixture("defillama_overview_slugs.json")
    fees = load_fixture("defillama_summary_fees.json")
    revenue = load_fixture("defillama_summary_revenue.json")
    seen: list[dict | None] = []

    def fake_get(client, url, params=None, **kwargs):
        if "/overview/" in url:
            return overview
        seen.append(params)
        if params and params.get("dataType") == "dailyRevenue":
            return revenue
        return fees

    monkeypatch.setattr("cryptoval.sources.defillama.get_with_retry", fake_get)
    result = DefiLlamaHistorySource().fetch(
        dt.date(2015, 1, 1), dt.date(2026, 10, 13), [_eth_asset()], None
    )
    assert seen == [None, {"dataType": "dailyRevenue"}]
    # A named day, not iloc[-1]: the parser drops the current UTC date, so the last row of a
    # frozen fixture is whichever day the clock has not yet reached -- the assertion used to
    # go red by itself the morning after it was written.
    frame = result.frame.set_index("metric_date")
    day = frame.loc[pd.Timestamp("2026-10-13", tz="UTC")]
    # The two series are different quantities and must not arrive equal.
    assert day["total_fees_usd"] != day["net_revenue_usd"]
    assert day["total_fees_usd"] == pytest.approx(404507.0)
    assert day["net_revenue_usd"] == pytest.approx(94003.0)


def test_two_identical_history_fetches_change_no_stored_row(monkeypatch, load_fixture):
    """The revision signal reconcile and as_of_view exist for: an int64/float64 flip in one
    value column would re-vintage all 27,728 rows and call a dtype a restatement."""
    overview = load_fixture("defillama_overview_slugs.json")
    fees = load_fixture("defillama_summary_fees.json")
    revenue = load_fixture("defillama_summary_revenue.json")

    def fake_get(client, url, params=None, **kwargs):
        if "/overview/" in url:
            return overview
        return revenue if params else fees

    monkeypatch.setattr("cryptoval.sources.defillama.get_with_retry", fake_get)
    args = (dt.date(2015, 1, 1), dt.date(2026, 10, 13), [_eth_asset()], None)
    first = DefiLlamaHistorySource().fetch(*args).frame
    second = DefiLlamaHistorySource().fetch(*args).frame
    values = ["total_fees_usd", "net_revenue_usd", "methodology_hash"]
    keys = ["protocol", "metric_date"]
    stored = merge_scd2(
        pd.DataFrame(columns=[*keys, *values]),
        first,
        keys,
        values,
        pd.Timestamp("2026-01-01", tz="UTC"),
    )
    again = merge_scd2(stored, second, keys, values, pd.Timestamp("2026-01-02", tz="UTC"))
    assert len(again) == len(stored)
    assert int(again["is_current"].sum()) == int(stored["is_current"].sum())


def test_one_display_name_mapping_to_two_slugs_is_refused(load_fixture):
    """Last-wins would address the summary endpoint with another protocol's slug."""
    overview = load_fixture("defillama_overview_slugs.json")
    doubled = {
        "protocols": [
            *overview["protocols"],
            {"name": overview["protocols"][0]["name"], "slug": "someone-else"},
        ]
    }
    with pytest.raises(ValueError, match="several slugs"):
        slug_map(doubled)


def _aave_asset():
    from cryptoval.config import Asset

    return Asset(symbol="AAVE", cm_id="aave", gecko_id="aave", defillama_slugs=["Aave V3"])


def test_parse_tvl_history_reads_the_tvl_array(load_fixture):
    frame = parse_tvl_history(load_fixture("defillama_protocol_tvl.json"))
    assert list(frame.columns) == ["protocol", "metric_date", "tvl_usd"]
    assert frame["metric_date"].dt.tz is not None
    assert (frame["tvl_usd"] > 0).all()
    assert frame["protocol"].unique().tolist() == ["Aave V3"]
    assert frame["tvl_usd"].dtype == "float64"


def test_parse_tvl_history_drops_the_partial_current_day(load_fixture):
    """The array carries today's 00:00 point AND a live one; both normalise to today, so
    keeping them would break the (protocol, metric_date) uniqueness contract."""
    payload = load_fixture("defillama_protocol_tvl.json")
    now = pd.Timestamp.now(tz="UTC")
    payload["tvl"] = [
        {"date": int((now.normalize() - pd.Timedelta(days=1)).timestamp()), "totalLiquidityUSD": 1},
        {"date": int(now.normalize().timestamp()), "totalLiquidityUSD": 2},
        {"date": int(now.timestamp()), "totalLiquidityUSD": 3},
    ]
    frame = parse_tvl_history(payload)
    assert len(frame) == 1
    assert not frame["metric_date"].duplicated().any()


def test_tvl_source_refuses_a_payload_above_its_size_guard(monkeypatch, load_fixture):
    overview = load_fixture("defillama_overview_slugs.json")
    big = {
        "name": "Aave V3",
        "tvl": [{"date": 1_700_000_000, "totalLiquidityUSD": 1.0}],
        "padding": "x" * 2_000_000,
    }
    monkeypatch.setattr(
        "cryptoval.sources.defillama.get_with_retry",
        lambda client, url, params=None, **kw: overview if "/overview/" in url else big,
    )
    result = DefiLlamaTvlSource(max_payload_mb=1.0).fetch(
        dt.date(2020, 1, 1), dt.date(2026, 10, 13), [_aave_asset()], client=None
    )
    assert result.frame.empty
    assert any("exceeds the" in e and "MB" in e for e in result.errors)


def test_two_identical_tvl_fetches_change_no_stored_row(monkeypatch, load_fixture):
    """Same int64/float64 trap as the fee history: the vendor sends an int for a small TVL."""
    overview = load_fixture("defillama_overview_slugs.json")
    payload = load_fixture("defillama_protocol_tvl.json")
    monkeypatch.setattr(
        "cryptoval.sources.defillama.get_with_retry",
        lambda client, url, params=None, **kw: overview if "/overview/" in url else payload,
    )
    args = (dt.date(2015, 1, 1), dt.date(2026, 10, 13), [_aave_asset()], None)
    first = DefiLlamaTvlSource().fetch(*args).frame
    second = DefiLlamaTvlSource().fetch(*args).frame
    keys, values = ["protocol", "metric_date"], ["tvl_usd"]
    stored = merge_scd2(
        pd.DataFrame(columns=[*keys, *values]),
        first,
        keys,
        values,
        pd.Timestamp("2026-01-01", tz="UTC"),
    )
    again = merge_scd2(stored, second, keys, values, pd.Timestamp("2026-01-02", tz="UTC"))
    assert len(again) == len(stored)
    assert int(again["is_current"].sum()) == int(stored["is_current"].sum())
