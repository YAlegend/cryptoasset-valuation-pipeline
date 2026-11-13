"""The factor chain's configuration: floors, lags and the exclusion list are data."""

from __future__ import annotations

import dataclasses
import pathlib

import yaml


@dataclasses.dataclass(frozen=True)
class FactorConfig:
    exclude: dict[str, str]  # listing id -> why it is not a risk asset of its own
    min_market_cap_usd: float = 1_000_000.0
    min_assets: int = 50
    momentum_weeks: int = 3
    newey_west_lags: int = 4
    winsor_quantile: float = 0.05
    deciles: int = 10
    tail_share: float = 0.3
    spike_ratio: float = 5.0


def load_factor_config(path: str = "config/factors.yml") -> FactorConfig:
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    exclude: dict[str, str] = {}
    for reason, ids in (raw.pop("exclude", None) or {}).items():
        for listing in ids:
            if listing in exclude:
                raise ValueError(f"{listing!r} is excluded twice ({exclude[listing]}, {reason})")
            exclude[listing] = reason
    return FactorConfig(exclude=exclude, **raw)
