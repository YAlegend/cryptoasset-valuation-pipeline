"""Price-history adapters: CoinGecko for crypto, yfinance for equities.

CoinGecko supplies the equation-of-exchange inputs (price, market cap,
volume) for every asset including those absent from Coin Metrics' free
tier, which is how the documented SOL gap still gets price coverage
(spec section 5).
"""

from __future__ import annotations

import datetime as dt
import time

import httpx
import numpy as np
import pandas as pd

from cryptoval.config import Asset
from cryptoval.sources.base import FetchResult, get_with_retry

BASE_URL = "https://api.coingecko.com/api/v3"
PANEL_COLUMNS = ["asset", "metric_date", "price_usd", "market_cap_usd", "volume_usd"]


def _series(rows: list[list[float]], name: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["metric_date", name])
    df = pd.DataFrame(rows, columns=["ts_ms", name])
    df["metric_date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.normalize()
    return df.drop(columns=["ts_ms"]).groupby("metric_date", as_index=False).last()


def parse_market_chart(payload: dict, gecko_id: str) -> pd.DataFrame:
    price = _series(payload.get("prices", []) or [], "price_usd")
    mcap = _series(payload.get("market_caps", []) or [], "market_cap_usd")
    vol = _series(payload.get("total_volumes", []) or [], "volume_usd")
    if price.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    df = price.merge(mcap, on="metric_date", how="left").merge(vol, on="metric_date", how="left")
    df["asset"] = gecko_id
    return df.reindex(columns=PANEL_COLUMNS)


def compute_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Zero or negative denominators give NaN, never inf.
    mcap = pd.to_numeric(out["market_cap_usd"], errors="coerce").where(lambda s: s > 0)
    vol = pd.to_numeric(out["volume_usd"], errors="coerce").where(lambda s: s > 0)
    out["velocity"] = vol / mcap
    out["holding_time_days"] = mcap / vol
    return out


class CoinGeckoSource:
    name = "coingecko"
    contract_path = "config/contracts/coingecko_market_chart.yml"

    def __init__(self, max_retries: int = 5, sleep_between_assets_s: float = 2.5) -> None:
        self.max_retries = max_retries
        self.sleep_between_assets_s = sleep_between_assets_s

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult:
        frames: list[pd.DataFrame] = []
        errors: list[str] = []
        rows_in = 0
        for asset in assets:
            if not asset.gecko_id:
                continue
            try:
                payload = get_with_retry(
                    client,
                    f"{BASE_URL}/coins/{asset.gecko_id}/market_chart/range",
                    {
                        "vs_currency": "usd",
                        "from": int(
                            dt.datetime.combine(start, dt.time.min, tzinfo=dt.UTC).timestamp()
                        ),
                        "to": int(
                            dt.datetime.combine(end, dt.time.max, tzinfo=dt.UTC).timestamp()
                        ),
                    },
                    max_retries=self.max_retries,
                    initial_delay=2.0,
                )
            except httpx.HTTPError as exc:
                # Nothing fetched yet: fail loudly. Otherwise keep what succeeded and stop —
                # a rate limit does not clear by moving on to the next asset, and a full pass
                # over the free tier takes minutes to redo (fix round 1, Ruling D).
                if not frames:
                    raise
                errors.append(f"{asset.gecko_id}: {exc}")
                break
            rows_in += len(payload.get("prices", []) or [])
            frames.append(parse_market_chart(payload, asset.gecko_id))
            time.sleep(self.sleep_between_assets_s)  # Demo tier is rate limited

        if not frames:
            return FetchResult(pd.DataFrame(columns=PANEL_COLUMNS), 0)
        return FetchResult(
            compute_velocity(pd.concat(frames, ignore_index=True)), rows_in, tuple(errors)
        )


class EquitySource:
    name = "equities"
    contract_path = "config/contracts/coingecko_market_chart.yml"

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult:
        import yfinance

        tickers = [a.yahoo_id for a in assets if a.is_equity and a.yahoo_id]
        if not tickers:
            return FetchResult(pd.DataFrame(columns=PANEL_COLUMNS), 0)

        raw = yfinance.download(
            tickers, start=start.isoformat(), end=end.isoformat(), auto_adjust=True, progress=False
        )
        closes = raw["Close"] if "Close" in raw else raw
        tidy = (
            closes.reset_index()
            .melt(id_vars=closes.index.name or "Date", var_name="asset", value_name="price_usd")
            .rename(columns={closes.index.name or "Date": "metric_date"})
        )
        tidy["metric_date"] = pd.to_datetime(tidy["metric_date"], utc=True).dt.normalize()
        # np.nan, not pd.NA: these are float columns, and stable_hash (and therefore
        # SCD2 change detection) cannot encode NAType (fix round 1, Ruling A).
        tidy["market_cap_usd"] = np.nan
        tidy["volume_usd"] = np.nan
        frame = tidy.reindex(columns=PANEL_COLUMNS).dropna(subset=["price_usd"])
        return FetchResult(frame, len(frame))
