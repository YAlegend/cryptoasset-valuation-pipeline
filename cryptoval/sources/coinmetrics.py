"""Coin Metrics Community API adapter.

Realized capitalisation is NOT free (CapRealUSD returns 403 on the
community tier), but CapMVRVCur is, and MVRV is market cap over realized
cap by definition. So realized_cap is derived from that identity, and
cryptoval.contracts.identities guards the derivation (spec section 6.2).
"""

from __future__ import annotations

import datetime as dt

import httpx
import numpy as np
import pandas as pd

from cryptoval.config import Asset
from cryptoval.sources.base import FetchResult, get_with_retry

MAX_PAGES = 1000

BASE_URL = "https://community-api.coinmetrics.io/v4"

FREE_METRICS: tuple[str, ...] = (
    "CapMrktCurUSD",
    "CapMVRVCur",
    "PriceUSD",
    "SplyCur",
    "HashRate",
    "IssTotUSD",
    "FeeTotNtv",
    "AdrActCnt",
    "AdrBalCnt",
    "TxCnt",
    "TxTfrCnt",
    "FlowInExUSD",
    "FlowOutExUSD",
    # Reported spot volume: free on the community tier back to 2015 and therefore the
    # full-history source of the equation-of-exchange T term, which CoinGecko's keyless
    # API cannot serve beyond 365 days (fix round 1, Ruling B).
    "volume_reported_spot_usd_1d",
)

_NON_METRIC_COLUMNS = {"asset", "time", "metric_date"}


def parse_timeseries(payload: dict) -> pd.DataFrame:
    rows = payload.get("data", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["asset", "metric_date"])
    df["metric_date"] = pd.to_datetime(df["time"], utc=True).dt.normalize()
    df = df.drop(columns=["time"])
    # Coin Metrics returns "<metric>-status" and "<metric>-status-time" beside some metrics.
    # On the live panel all three are 100% empty, and they are strings in a numeric frame.
    sidecars = [c for c in df.columns if c.endswith(("-status", "-status-time"))]
    if sidecars:
        df = df.drop(columns=sidecars)
    for col in df.columns:
        if col not in _NON_METRIC_COLUMNS:
            # errors="coerce" turns genuine nulls into NaN; it must never
            # turn them into 0.0 (Review Focus 3).
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def derive_realized_cap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not {"CapMrktCurUSD", "CapMVRVCur"}.issubset(out.columns):
        out["realized_cap"] = np.nan
        return out
    mvrv = pd.to_numeric(out["CapMVRVCur"], errors="coerce")
    # Zero, negative and null MVRV are all meaningless: yield NaN, never inf
    # and never a silently-wrong value (Review Focus 1).
    safe_mvrv = mvrv.where(mvrv > 0)
    out["realized_cap"] = pd.to_numeric(out["CapMrktCurUSD"], errors="coerce") / safe_mvrv
    return out


class CoinMetricsSource:
    name = "coinmetrics"
    contract_path = "config/contracts/coinmetrics_asset_metrics.yml"

    def __init__(self, max_retries: int = 5, page_size: int = 10_000) -> None:
        self.max_retries = max_retries
        self.page_size = page_size

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult:
        # The union, not one id per asset: three assets draw different metric groups from
        # different Coin Metrics series (spec 5.1), so both ids must be requested.
        ids = sorted({cm_id for a in assets for cm_id in a.cm_ids})
        if not ids:
            return FetchResult(pd.DataFrame(columns=["asset", "metric_date"]), 0)

        frames: list[pd.DataFrame] = []
        rows_in = 0
        url = f"{BASE_URL}/timeseries/asset-metrics"
        params: dict | None = {
            "assets": ",".join(ids),
            "metrics": ",".join(FREE_METRICS),
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "frequency": "1d",
            "page_size": self.page_size,
        }
        next_url: str | None = url
        pages = 0
        while next_url:
            if pages >= MAX_PAGES:
                # Raise rather than return partial data: a silently truncated
                # backfill is worse than a loud failure.
                raise RuntimeError(
                    f"Coin Metrics pagination exceeded {MAX_PAGES} pages; last url: {next_url}"
                )
            pages += 1
            payload = get_with_retry(client, next_url, params, max_retries=self.max_retries)
            rows_in += len(payload.get("data", []) or [])
            frames.append(parse_timeseries(payload))
            next_url = payload.get("next_page_url")
            params = None  # next_page_url already carries the query string

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if combined.empty:
            return FetchResult(pd.DataFrame(columns=["asset", "metric_date"]), rows_in)
        return FetchResult(derive_realized_cap(combined), rows_in)
