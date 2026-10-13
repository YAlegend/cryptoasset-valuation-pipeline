"""Generate dbt seeds from config/assets.yml.

Spec 8.1: dim_asset is SEEDED, not derived. Deriving it from the panel made its grain
the Coin Metrics vendor id, so `trx_eth` and `avaxc` were the dimension's keys and
nothing joined the four sources. The config is the authority on the asset universe;
these CSVs are a committed projection of it that dbt can read.
"""

from __future__ import annotations

import csv
import pathlib

from cryptoval.config import METRIC_GROUPS, Asset
from cryptoval.methods.base import MethodSpec

ASSET_COLUMNS = [
    "symbol",
    "sector",
    "consensus",
    "is_stablecoin",
    "is_equity",
    "cm_id",
    "gecko_id",
    "yahoo_id",
    "defillama_slugs",
    "coverage_note",
    "genesis_date",
]
SOURCE_COLUMNS = ["symbol", "metric_group", "source_kind", "source_id"]
METHOD_COLUMNS = [
    "method_id",
    "family",
    "requires",
    "min_density",
    "min_rows",
    "citation",
    "returns_fair_value",
]


def _write(path: pathlib.Path, columns: list[str], rows: list[dict]) -> str:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def write_asset_seeds(assets: list[Asset], seed_dir: str) -> list[str]:
    out = pathlib.Path(seed_dir)
    out.mkdir(parents=True, exist_ok=True)

    asset_rows = [
        {
            "symbol": a.symbol,
            "sector": a.sector or "",
            "consensus": a.consensus,
            "is_stablecoin": str(a.is_stablecoin).lower(),
            "is_equity": str(a.is_equity).lower(),
            "cm_id": a.cm_id or "",
            "gecko_id": a.gecko_id or "",
            "yahoo_id": a.yahoo_id or "",
            # Pipe-delimited: these are CSVs and the slugs contain spaces but never pipes.
            "defillama_slugs": "|".join(a.defillama_slugs),
            "coverage_note": a.coverage_note,
            "genesis_date": a.genesis_date.isoformat() if a.genesis_date else "",
        }
        for a in assets
    ]

    source_rows: list[dict] = []
    for a in assets:
        if a.is_equity:
            continue
        for group in METRIC_GROUPS:
            source = a.source_for(group)
            if source == "none":
                # A row pointing at nothing would join to nothing and read as a config
                # error downstream. Absence is the honest encoding of "unavailable".
                continue
            if source.startswith("cm:"):
                source_rows.append(
                    {
                        "symbol": a.symbol,
                        "metric_group": group,
                        "source_kind": "cm",
                        "source_id": source[len("cm:") :],
                    }
                )
            else:
                source_rows.append(
                    {
                        "symbol": a.symbol,
                        "metric_group": group,
                        "source_kind": "coingecko",
                        "source_id": a.gecko_id or "",
                    }
                )

    return [
        _write(out / "assets.csv", ASSET_COLUMNS, asset_rows),
        _write(out / "asset_metric_sources.csv", SOURCE_COLUMNS, source_rows),
    ]


def write_method_seed(specs: dict[str, MethodSpec], seed_dir: str) -> str:
    """dim_method's source, projected from config/methods/ the way assets.csv is from config.

    Sorted by method_id so the committed file does not churn on dict ordering.
    """
    out = pathlib.Path(seed_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "method_id": spec.method_id,
            "family": spec.family,
            # Pipe-delimited for the same reason as defillama_slugs: a CSV field holding a
            # list, read far more often than it is parsed.
            "requires": "|".join(spec.requires),
            "min_density": spec.min_density,
            "min_rows": spec.min_rows,
            "citation": spec.citation,
            "returns_fair_value": str(spec.returns_fair_value).lower(),
        }
        for spec in sorted(specs.values(), key=lambda s: s.method_id)
    ]
    return _write(out / "methods.csv", METHOD_COLUMNS, rows)
