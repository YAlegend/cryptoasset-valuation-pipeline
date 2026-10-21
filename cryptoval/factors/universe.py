"""The Coin Metrics community universe (design 1-3): every listing serving a daily market cap.

Membership is point-in-time INSIDE the catalog (no asset enters before its data does), but
the catalog itself is chosen ex post by Coin Metrics. That selection bias is declared in the
chapter, not corrected here.
"""

from __future__ import annotations

import httpx
import pandas as pd

from cryptoval.factors.settings import FactorConfig
from cryptoval.sources.base import get_with_retry
from cryptoval.sources.coinmetrics import BASE_URL, MAX_PAGES, parse_timeseries

METRICS = {
    "PriceUSD": "price_usd",
    "CapMrktCurUSD": "market_cap_usd",
    "AdrActCnt": "active_addresses",
    "CapMVRVCur": "mvrv",
}
UNIVERSE_COLUMNS = ["asset", "metric_date", *METRICS.values()]
_POSITIVE = ("price_usd", "market_cap_usd", "mvrv")


def catalog_ids(client: httpx.Client, max_retries: int = 5) -> list[str]:
    payload = get_with_retry(
        client,
        f"{BASE_URL}/catalog-v2/asset-metrics",
        {"metrics": "CapMrktCurUSD", "page_size": 10_000},
        max_retries=max_retries,
    )
    if payload.get("next_page_url"):
        raise RuntimeError("the catalog no longer fits one page; follow next_page_url")
    return sorted(
        entry["asset"]
        for entry in payload["data"]
        if any(
            m["metric"] == "CapMrktCurUSD" and any(f["frequency"] == "1d" for f in m["frequencies"])
            for m in entry["metrics"]
        )
    )


def check_exclusions(config: FactorConfig, ids: list[str]) -> None:
    unknown = sorted(set(config.exclude) - set(ids))
    if unknown:
        # A typo would silently keep the asset it meant to remove.
        raise ValueError(f"excluded ids not in the Coin Metrics catalog: {unknown}")


def to_universe(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.rename(columns=METRICS).reindex(columns=UNIVERSE_COLUMNS)
    out["metric_date"] = pd.to_datetime(out["metric_date"], utc=True).dt.tz_localize(None)
    for col in METRICS.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in _POSITIVE:
        out[col] = out[col].where(out[col] > 0)
    duplicated = out.duplicated(["asset", "metric_date"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicate (asset, metric_date) rows")
    return out.sort_values(["asset", "metric_date"], ignore_index=True)


def fetch_universe(client: httpx.Client, ids: list[str], max_retries: int = 5) -> pd.DataFrame:
    params: dict | None = {
        "assets": ",".join(ids),
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "page_size": 10_000,
        "start_time": "2009-01-01",
        # most listings lack AdrActCnt or CapMVRVCur; without this one gap fails the request
        "ignore_unsupported_errors": "true",
    }
    next_url: str | None = f"{BASE_URL}/timeseries/asset-metrics"
    frames: list[pd.DataFrame] = []
    while next_url:
        if len(frames) >= MAX_PAGES:
            raise RuntimeError(f"pagination exceeded {MAX_PAGES} pages; last url: {next_url}")
        payload = get_with_retry(client, next_url, params, max_retries=max_retries)
        frames.append(parse_timeseries(payload))
        next_url = payload.get("next_page_url")
        params = None  # next_page_url carries the query string
    return to_universe(pd.concat(frames, ignore_index=True))
