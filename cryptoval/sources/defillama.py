"""DefiLlama adapter for protocol fees and revenue.

Gross fees, net protocol revenue and value reaching token holders are three
different quantities (spec section 10.4). This adapter keeps them in
separate columns and never substitutes one for another, so Plan 2's
multiples cannot quietly use the wrong denominator.
"""

from __future__ import annotations

import datetime as dt

import httpx
import pandas as pd

from cryptoval.config import Asset
from cryptoval.hashing import stable_hash
from cryptoval.sources.base import FetchResult, get_with_retry

BASE_URL = "https://api.llama.fi"
OVERVIEW_PATH = "/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"


def methodology_hash(record: dict) -> str:
    return stable_hash(record.get("methodology") or {})


def parse_fees_overview(payload: dict) -> pd.DataFrame:
    rows = []
    for record in payload.get("protocols", []) or []:
        rows.append(
            {
                "protocol": record.get("name"),
                "total_fees_24h": record.get("total24h"),
                "total_fees_30d": record.get("total30d"),
                "net_revenue_24h": record.get("revenue24h"),
                "methodology_hash": methodology_hash(record),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "protocol",
                "total_fees_24h",
                "total_fees_30d",
                "net_revenue_24h",
                "methodology_hash",
            ]
        )
    for col in ("total_fees_24h", "total_fees_30d", "net_revenue_24h"):
        # A null revenue stays null. It is not the same as zero revenue, and
        # it is certainly not gross fees.
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class DefiLlamaSource:
    name = "defillama"
    contract_path = "config/contracts/defillama_fees.yml"

    def __init__(self, max_retries: int = 5) -> None:
        self.max_retries = max_retries

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult:
        payload = get_with_retry(client, f"{BASE_URL}{OVERVIEW_PATH}", max_retries=self.max_retries)
        rows_in = len(payload.get("protocols", []) or [])
        frame = parse_fees_overview(payload)
        frame["metric_date"] = pd.Timestamp(end, tz="UTC").normalize()
        return FetchResult(frame, rows_in)
