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


SUMMARY_PATH = "/summary/fees"
HISTORY_COLUMNS = [
    "protocol",
    "metric_date",
    "total_fees_usd",
    "net_revenue_usd",
    "methodology_hash",
]


def slug_map(payload: dict) -> dict[str, str]:
    """Display name -> URL slug, from the overview payload.

    The configured identity in assets.yml is the display name, because that is what the
    overview matches on; the summary endpoint is addressed by slug. Both live in the same
    payload, so the mapping is read rather than hand-maintained -- a hand-written slug is
    one rebrand away from a silent 404.
    """
    return {
        record["name"]: record["slug"]
        for record in payload.get("protocols", []) or []
        if record.get("name") and record.get("slug")
    }


def parse_fee_history(payload: dict, column: str) -> pd.DataFrame:
    """One protocol's daily series from a /summary/fees response.

    DefiLlama returns [unix_seconds, usd] pairs. Parsed with unit="s", utc=True: a naive
    timestamp here would shift every cash flow by up to a day against the price panel.
    """
    chart = payload.get("totalDataChart") or []
    if not chart:
        return pd.DataFrame(columns=["protocol", "metric_date", column])
    frame = pd.DataFrame(chart, columns=["ts", column])
    frame["metric_date"] = pd.to_datetime(frame["ts"], unit="s", utc=True).dt.normalize()
    frame["protocol"] = payload.get("name")
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["protocol", "metric_date", column]]


def trim_leading_zeros(frame: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Drop the pre-existence zeros at the head of a series, keeping interior zeros.

    DefiLlama's chart starts at the DATASET's start, not the protocol's: Ethereum's
    revenue series opens in August 2015 with zeros. Those are "this protocol did not
    exist", and a DCF that averages them understates the protocol's economics for as long
    as the window reaches back. An INTERIOR zero is different -- a genuine zero-fee day,
    or an adapter outage -- and is kept, because dropping it would silently shorten the
    series that later rows average over.
    """
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    nonzero = values.to_numpy().nonzero()[0]
    if len(nonzero) == 0:
        return frame.iloc[0:0], len(frame)
    first = int(nonzero[0])
    return frame.iloc[first:].reset_index(drop=True), first


class DefiLlamaHistorySource:
    """Per-protocol daily fee and revenue HISTORY (spec 10.4's denominator discipline).

    Separate from DefiLlamaSource rather than folded into it: that source's row grain is
    one snapshot row per protocol per run, this one's is one row per protocol per DAY of
    history, and merging two grains into one silver table would make the SCD2 merge
    compare a 4,000-row backfill against a 1-row snapshot.

    The endpoint serves the whole series regardless of start/end, so those are unused:
    every run re-reads the full history and SCD2 keeps only what changed.
    """

    name = "defillama_history"
    contract_path = "config/contracts/defillama_history.yml"

    def __init__(self, max_retries: int = 5) -> None:
        self.max_retries = max_retries

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult:
        overview = get_with_retry(
            client, f"{BASE_URL}{OVERVIEW_PATH}", max_retries=self.max_retries
        )
        slugs = slug_map(overview)
        wanted = {slug for asset in assets for slug in asset.defillama_slugs or []}

        frames: list[pd.DataFrame] = []
        errors: list[str] = []
        rows_in = 0
        # No try/except around the summary calls, for the reason DefiLlamaSource gives: a
        # failed call that returned partial data would let SCD2 close the stored series.
        for protocol in sorted(wanted):
            slug = slugs.get(protocol)
            if slug is None:
                # A configured protocol the feed no longer lists: name it. Plan 2 Task 5
                # already proved this happens (Maker -> Sky Lending).
                errors.append(f"no DefiLlama slug for configured protocol {protocol!r}")
                continue
            series: dict[str, pd.DataFrame] = {}
            fees_hash = ""
            for column, data_type in (
                ("total_fees_usd", None),
                ("net_revenue_usd", "dailyRevenue"),
            ):
                payload = get_with_retry(
                    client,
                    f"{BASE_URL}{SUMMARY_PATH}/{slug}",
                    {"dataType": data_type} if data_type else None,
                    max_retries=self.max_retries,
                )
                if column == "total_fees_usd":
                    fees_hash = methodology_hash(payload)
                parsed = parse_fee_history(payload, column)
                rows_in += len(parsed)
                trimmed, _ = trim_leading_zeros(parsed, column)
                if trimmed.empty:
                    # Fees nobody tracked is a gap worth naming. Revenue that is zero or
                    # absent is routine (Uniswap V4's fee switch is off), and an `errors`
                    # entry would make every ingest "partial" forever.
                    if column == "total_fees_usd":
                        errors.append(f"{protocol}/{column}: series is entirely zero or empty")
                    continue
                # The configured display name, not the payload's: the rollup joins on it.
                series[column] = trimmed.assign(protocol=protocol)

            if not series:
                continue
            merged = None
            for frame in series.values():
                merged = (
                    frame
                    if merged is None
                    else merged.merge(frame, on=["protocol", "metric_date"], how="outer")
                )
            merged["methodology_hash"] = fees_hash
            frames.append(merged)

        if not frames:
            return FetchResult(pd.DataFrame(columns=HISTORY_COLUMNS), rows_in, tuple(errors))
        combined = pd.concat(frames, ignore_index=True).reindex(columns=HISTORY_COLUMNS)
        return FetchResult(combined, rows_in, tuple(errors))
