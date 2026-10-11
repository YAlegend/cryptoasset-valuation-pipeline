"""Configuration and the asset universe, loaded from YAML."""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Settings:
    data_root: str = "./data"
    trailing_refetch_days: int = 30
    bigquery_monthly_byte_budget: int = 900_000_000_000
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
    for row in raw_assets.get("assets", []) or []:
        assets.append(Asset(**row))
    for row in raw_assets.get("equities", []) or []:
        assets.append(Asset(is_equity=True, **row))
    for a in assets:
        _validate(a)
    return settings, assets
