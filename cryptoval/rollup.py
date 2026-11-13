"""Roll DefiLlama's per-protocol rows up to one row per valuation symbol.

Spec 5.1. DefiLlama splits a protocol across versions and renames it on rebrand, so
an asset's revenue is a LIST of slugs: UNI is Uniswap V4 plus V3 plus V2, and MKR is
Sky Lending plus Sky Money because no protocol named Maker exists in the feed any
more. Summing the wrong set is not a loud failure -- it is a confidently wrong DCF
input -- so the one configuration that silently inflates a sum is refused here.
"""

from __future__ import annotations

import pandas as pd

from cryptoval.config import Asset

REVENUE_COLUMNS = ["total_fees_24h", "total_fees_30d", "net_revenue_24h"]
OUTPUT_COLUMNS = ["symbol", "metric_date", *REVENUE_COLUMNS, "protocol_count"]


def assert_no_aggregate_collision(slugs: tuple[str, ...], symbol: str) -> None:
    """Refuse a slug set holding both an aggregate and its own components.

    Summing "Uniswap" together with "Uniswap V3" double-counts V3. A shared prefix only
    means a family when it ends at a word boundary: "Aave" and "Aavegotchi" are
    unrelated protocols that happen to share four letters.
    """
    for outer in slugs:
        for inner in slugs:
            if outer == inner:
                continue
            if inner.startswith(outer + " "):
                raise ValueError(
                    f"{symbol}: defillama_slugs holds both {outer!r} and {inner!r}, which "
                    f"would double-count {inner!r}; list either the aggregate or its "
                    "components, not both"
                )


def rollup_protocols(df: pd.DataFrame, assets: list[Asset]) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for asset in assets:
        if not asset.defillama_slugs:
            continue
        assert_no_aggregate_collision(tuple(asset.defillama_slugs), asset.symbol)
        for slug in asset.defillama_slugs:
            mapping[slug] = asset.symbol

    if df.empty or not mapping:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    scoped = df[df["protocol"].isin(mapping)].copy()
    if scoped.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    scoped["symbol"] = scoped["protocol"].map(mapping)

    present = [c for c in REVENUE_COLUMNS if c in scoped.columns]
    grouped = scoped.groupby(["symbol", "metric_date"], as_index=False, dropna=False)
    # min_count=1 keeps an all-null group null: pandas would otherwise sum it to 0.0,
    # which asserts "this protocol retained nothing" rather than "we do not know".
    summed = grouped[present].sum(min_count=1)
    counts = grouped.size().rename(columns={"size": "protocol_count"})
    out = summed.merge(counts, on=["symbol", "metric_date"], how="left")
    return out.reindex(columns=OUTPUT_COLUMNS)
