"""Configuration and the asset universe, loaded from YAML."""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import re
from dataclasses import dataclass, field

import yaml

from cryptoval.paths import PathResolver

SECTORS = (
    "pow_monetary",
    "smart_contract_l1",
    "scaling",
    "defi",
    "oracle",
    "exchange",
    "metaverse",
    "stablecoin",
)
CONSENSUS = ("pow", "pos", "other", "na")
METRIC_GROUPS = ("network", "valuation", "volume")

# A metric source is one Coin Metrics id, CoinGecko, or an explicit refusal. The lowercase
# pattern is deliberate: Coin Metrics ids are lowercase, and silently accepting "CM:X" would
# produce a 404 on the first live fetch instead of a config error here.
_SOURCE_PATTERN = re.compile(r"^(cm:[a-z0-9_]+|coingecko|none)$")


@dataclass(frozen=True)
class Asset:
    symbol: str
    sector: str | None = None
    consensus: str = "na"
    cm_id: str | None = None
    gecko_id: str | None = None
    yahoo_id: str | None = None
    is_stablecoin: bool = False
    is_equity: bool = False
    # Group -> source. Unmapped groups fall back to the primary cm_id (spec 5.1), so only
    # the three assets whose vendor identity is split need an entry here.
    metric_sources: dict[str, str] = field(default_factory=dict)
    defillama_slugs: tuple[str, ...] = ()
    coverage_note: str = ""
    # The asset's real launch, a matter of record; absent when it is not. The power law
    # refuses an asset without one rather than borrowing the vendor's coverage start.
    genesis_date: dt.date | None = None

    def source_for(self, group: str) -> str:
        if group not in METRIC_GROUPS:
            raise ValueError(f"unknown metric group {group!r}; expected one of {METRIC_GROUPS}")
        explicit = self.metric_sources.get(group)
        if explicit:
            return explicit
        if self.cm_id:
            return f"cm:{self.cm_id}"
        return "none" if group == "network" else "coingecko"

    @property
    def cm_ids(self) -> tuple[str, ...]:
        ids = {s[len("cm:") :] for s in map(self.source_for, METRIC_GROUPS) if s.startswith("cm:")}
        return tuple(sorted(ids))


@dataclass(frozen=True)
class Settings:
    data_root: str = "./data"
    trailing_refetch_days: int = 30
    bigquery_monthly_byte_budget: int = 900_000_000_000
    bigquery_project: str = ""
    request_timeout_s: int = 30
    max_retries: int = 5

    @property
    def resolver(self) -> PathResolver:
        return PathResolver(self.data_root)


def _validate(asset: Asset) -> None:
    if asset.is_equity:
        if not asset.yahoo_id:
            raise ValueError(f"{asset.symbol}: equity has no usable source id")
        return
    if asset.sector not in SECTORS:
        raise ValueError(f"{asset.symbol}: unknown sector {asset.sector!r}")
    if asset.consensus not in CONSENSUS:
        raise ValueError(f"{asset.symbol}: unknown consensus {asset.consensus!r}")
    for group, source in asset.metric_sources.items():
        if group not in METRIC_GROUPS:
            raise ValueError(
                f"{asset.symbol}: unknown metric group {group!r}; expected one of {METRIC_GROUPS}"
            )
        if not _SOURCE_PATTERN.match(str(source)):
            raise ValueError(
                f"{asset.symbol}: unusable metric source {source!r} for group {group!r}; "
                "expected 'cm:<id>', 'coingecko' or 'none'"
            )
        if source == "coingecko" and not asset.gecko_id:
            raise ValueError(f"{asset.symbol}: group {group!r} names coingecko but has no gecko_id")
    if asset.genesis_date is not None and (
        not isinstance(asset.genesis_date, dt.date) or isinstance(asset.genesis_date, dt.datetime)
    ):
        # YAML itself rejects an impossible date like 2009-13-45; a non-date string or a
        # timestamp would parse and reach the power law as garbage instead of failing here.
        raise ValueError(
            f"{asset.symbol}: genesis_date must be a YYYY-MM-DD date, got {asset.genesis_date!r}"
        )
    if any(not str(slug).strip() for slug in asset.defillama_slugs):
        raise ValueError(f"{asset.symbol}: defillama_slugs contains an empty entry")
    if not asset.cm_id and not asset.gecko_id:
        raise ValueError(f"{asset.symbol}: no usable source id")


def load_config(config_dir: str = "config") -> tuple[Settings, list[Asset]]:
    base = pathlib.Path(config_dir)
    raw_settings = yaml.safe_load((base / "settings.yml").read_text(encoding="utf-8")) or {}
    data_root = os.environ.get("CRYPTOVAL_DATA_ROOT", raw_settings.get("data_root", "./data"))
    if not data_root or not str(data_root).strip():
        raise ValueError("data_root must be a non-empty path")
    raw_settings["data_root"] = data_root
    settings = Settings(**raw_settings)

    raw_assets = yaml.safe_load((base / "assets.yml").read_text(encoding="utf-8")) or {}
    assets: list[Asset] = []

    def _build(row: dict, *, is_equity: bool) -> Asset:
        row = dict(row)
        slugs = row.get("defillama_slugs") or ()
        if isinstance(slugs, str):
            # tuple("Lido") is ('L','i','d','o'): four slugs that match no protocol, so the
            # rollup silently emits no revenue for the asset and no check fails.
            raise ValueError(
                f"{row.get('symbol')}: defillama_slugs must be a list, got the string {slugs!r}"
            )
        row["defillama_slugs"] = tuple(slugs)
        row["metric_sources"] = dict(row.get("metric_sources") or {})
        return Asset(is_equity=is_equity, **row)

    for row in raw_assets.get("assets", []) or []:
        assets.append(_build(row, is_equity=False))
    for row in raw_assets.get("equities", []) or []:
        assets.append(_build(row, is_equity=True))
    for a in assets:
        _validate(a)
    return settings, assets
