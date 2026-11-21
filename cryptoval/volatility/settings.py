"""The volatility chain's configuration (design 8): windows and anchors are data."""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib

import yaml

_INTS = ("arch_lm_lags", "exhaustion_block_days", "engle_sheppard_lags", "markov_search_reps")


@dataclasses.dataclass(frozen=True)
class VolConfig:
    windows: dict[str, tuple[dt.date, dt.date]]
    peak_window: str
    crypto_anchors: tuple[str, ...]
    traditional_anchors: tuple[str, ...]
    arch_lm_lags: int = 7
    exhaustion_block_days: int = 30
    engle_sheppard_lags: int = 5
    markov_search_reps: int = 20


def load_vol_config(path: str = "config/volatility.yml") -> VolConfig:
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    windows: dict[str, tuple[dt.date, dt.date]] = {}
    for name, (start, end) in raw["windows"].items():
        start, end = dt.date.fromisoformat(str(start)), dt.date.fromisoformat(str(end))
        if end < start:
            raise ValueError(f"window {name}: end {end} is before start {start}")
        windows[name] = (start, end)
    if raw["peak_window"] not in windows:
        raise ValueError(f"peak_window {raw['peak_window']!r} is not one of {sorted(windows)}")
    return VolConfig(
        windows=windows,
        peak_window=raw["peak_window"],
        crypto_anchors=tuple(raw["crypto_anchors"]),
        traditional_anchors=tuple(raw["traditional_anchors"]),
        **{key: int(raw[key]) for key in _INTS if key in raw},
    )
