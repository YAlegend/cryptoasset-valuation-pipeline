import httpx
import pandas as pd
import pytest

from cryptoval.factors import universe
from cryptoval.factors.settings import FactorConfig


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _entry(asset, metric, frequency):
    return {
        "asset": asset,
        "metrics": [{"metric": metric, "frequencies": [{"frequency": frequency}]}],
    }


def test_catalog_ids_keeps_daily_market_cap_listings_only():
    payload = {
        "data": [
            _entry("btc", "CapMrktCurUSD", "1d"),
            _entry("xyz", "CapMrktCurUSD", "1h"),
            _entry("abc", "PriceUSD", "1d"),
        ]
    }
    with _client(lambda request: httpx.Response(200, json=payload)) as client:
        assert universe.catalog_ids(client) == ["btc"]


def test_an_exclusion_not_in_the_catalog_is_refused():
    config = FactorConfig(exclude={"wbtc": "wrapper", "wbtcc": "wrapper"})
    with pytest.raises(ValueError, match="wbtcc"):
        universe.check_exclusions(config, ["btc", "wbtc"])


def test_fetch_follows_pages_and_asks_to_ignore_unsupported_metrics():
    seen = []
    day = "2020-01-05T00:00:00.000000000Z"

    def handler(request):
        seen.append(request.url)
        if "page=2" in str(request.url):
            return httpx.Response(
                200, json={"data": [{"asset": "eth", "time": day, "PriceUSD": "130"}]}
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {"asset": "btc", "time": day, "PriceUSD": "7400", "CapMrktCurUSD": "1.3e11"}
                ],
                "next_page_url": "https://community-api.coinmetrics.io/v4/x?page=2",
            },
        )

    with _client(handler) as client:
        out = universe.fetch_universe(client, ["btc", "eth"])
    assert seen[0].params["ignore_unsupported_errors"] == "true"
    assert len(seen) == 2
    assert list(out.columns) == universe.UNIVERSE_COLUMNS
    assert list(out["asset"]) == ["btc", "eth"]
    assert out["metric_date"].dt.tz is None


def test_to_universe_nulls_non_positive_values_and_refuses_duplicates():
    raw = pd.DataFrame(
        {
            "asset": ["a", "b"],
            "metric_date": pd.to_datetime(["2020-01-05", "2020-01-05"], utc=True),
            "PriceUSD": [0.0, 2.0],
            "CapMrktCurUSD": [5.0, -1.0],
            "CapMVRVCur": [0.0, 1.5],
        }
    )
    out = universe.to_universe(raw)
    assert out["price_usd"].isna().tolist() == [True, False]
    assert out["market_cap_usd"].isna().tolist() == [False, True]
    assert out["mvrv"].isna().tolist() == [True, False]
    assert out["active_addresses"].isna().all()  # a metric nobody served is a null column
    with pytest.raises(ValueError, match="duplicate"):
        universe.to_universe(pd.concat([raw, raw]))
