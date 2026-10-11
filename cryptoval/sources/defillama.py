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
# The fees overview carries no revenue field: net protocol revenue is the SAME endpoint
# under a different dataType, and reading revenue24h off the fees payload left the column
# 100% null on live data (fix round 1, Ruling G).
REVENUE_PATH = OVERVIEW_PATH + "&dataType=dailyRevenue"

FEE_COLUMNS = ["protocol", "total_fees_24h", "total_fees_30d", "methodology_hash"]
PANEL_COLUMNS = [
    "protocol",
    "total_fees_24h",
    "total_fees_30d",
    "net_revenue_24h",
    "methodology_hash",
]


def methodology_hash(record: dict) -> str:
    return stable_hash(record.get("methodology") or {})


def parse_fees_overview(payload: dict) -> pd.DataFrame:
    rows = [
        {
            "protocol": record.get("name"),
            "total_fees_24h": record.get("total24h"),
            "total_fees_30d": record.get("total30d"),
            "methodology_hash": methodology_hash(record),
        }
        for record in payload.get("protocols", []) or []
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=FEE_COLUMNS)
    for col in ("total_fees_24h", "total_fees_30d"):
        # A null stays null. It is not the same as zero.
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_revenue_overview(payload: dict) -> pd.DataFrame:
    """Net protocol revenue per protocol, from the dailyRevenue dataType."""
    rows = [
        {"protocol": record.get("name"), "net_revenue_24h": record.get("total24h")}
        for record in payload.get("protocols", []) or []
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["protocol", "net_revenue_24h"])
    # A null revenue stays null. It is not the same as zero revenue, and it is
    # certainly not gross fees.
    df["net_revenue_24h"] = pd.to_numeric(df["net_revenue_24h"], errors="coerce")
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
        # rows_in counts the fees payload only: that is the row spine, and revenue is an
        # enrichment joined onto it, so rows_in stays comparable with rows_out.
        rows_in = len(payload.get("protocols", []) or [])
        fees = parse_fees_overview(payload)

        # No try/except: a revenue failure must fail the whole fetch. Returning fees with
        # an all-NULL net_revenue_24h would make merge_scd2 close every good revenue row
        # and publish nulls as the current vintage, with every validate check still green.
        # Losing one run is strictly better than losing the stored series.
        revenue = parse_revenue_overview(
            get_with_retry(client, f"{BASE_URL}{REVENUE_PATH}", max_retries=self.max_retries)
        )

        errors: list[str] = []
        conflicting = sorted(
            revenue.loc[revenue.duplicated(subset="protocol", keep=False), "protocol"].unique()
        )
        if conflicting:
            # Two revenue figures for one name: dropping BOTH turns the conflict into an
            # explicit null. keep="first" would pick one by payload order, which is a guess.
            revenue = revenue.drop_duplicates(subset="protocol", keep=False)
            errors.append(
                f"duplicate protocol(s) in the revenue feed, revenue dropped: {conflicting}"
            )

        # Fees is the row spine, so a protocol DefiLlama lists for revenue but not for
        # fees loses its revenue entirely. Count it rather than dropping it in silence.
        orphans = sorted(set(revenue["protocol"]) - set(fees["protocol"]))
        if orphans:
            errors.append(f"{len(orphans)} protocol(s) with revenue but no fees row: {orphans}")

        frame = fees.merge(revenue, on="protocol", how="left").reindex(columns=PANEL_COLUMNS)
        frame["metric_date"] = pd.Timestamp(end, tz="UTC").normalize()
        return FetchResult(frame, rows_in, tuple(errors))
