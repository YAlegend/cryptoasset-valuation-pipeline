from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class Contract:
    name: str
    version: str
    required_columns: list[str]
    columns: dict[str, dict[str, Any]] = field(default_factory=dict)
    unique_on: list[str] = field(default_factory=list)
    freshness_max_delay_hours: int | None = None
    volume_anomaly_threshold: float | None = None

    @property
    def ranges(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for col, spec in self.columns.items():
            bounds = {k: spec[k] for k in ("minimum", "maximum") if k in spec}
            if bounds:
                out[col] = bounds
        return out


def load_contract(path: str) -> Contract:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"contract not found: {path}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    meta = raw.get("contract", {})
    schema = raw.get("schema", {})
    quality = raw.get("quality", {})
    return Contract(
        name=meta.get("name", p.stem),
        version=meta.get("version", "0.0.0"),
        required_columns=list(schema.get("required", []) or []),
        columns=dict(schema.get("columns", {}) or {}),
        unique_on=list((quality.get("uniqueness", {}) or {}).get("combination", []) or []),
        freshness_max_delay_hours=(quality.get("freshness", {}) or {}).get("max_delay_hours"),
        volume_anomaly_threshold=(quality.get("volume", {}) or {}).get("anomaly_threshold"),
    )
