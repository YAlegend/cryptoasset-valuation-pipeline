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


_QUALITY_KEYS = {
    "freshness": {"max_delay_hours"},
    "uniqueness": {"combination"},
    "volume": {"anomaly_threshold"},
    "notes": None,  # free text, not validated
}
_COLUMN_KEYS = {"type", "minimum", "maximum", "description"}


def _reject_unknown(found, allowed: set[str], where: str) -> None:
    unknown = sorted(str(k) for k in found if k not in allowed)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {where}; allowed: {sorted(allowed)}")


def _validate(raw: dict[str, Any], source: str) -> None:
    quality = raw.get("quality", {}) or {}
    _reject_unknown(quality, set(_QUALITY_KEYS), f"quality of {source}")
    for section, allowed in _QUALITY_KEYS.items():
        if allowed is None or section not in quality:
            continue
        body = quality[section] or {}
        _reject_unknown(body, allowed, f"quality.{section} of {source}")
    columns = (raw.get("schema", {}) or {}).get("columns", {}) or {}
    for col, spec in columns.items():
        spec = spec or {}
        _reject_unknown(spec, _COLUMN_KEYS, f"schema.columns.{col} of {source}")
        for bound in ("minimum", "maximum"):
            if bound in spec:
                v = spec[bound]
                if isinstance(v, bool) or not isinstance(v, int | float):
                    raise ValueError(
                        f"{bound} in schema.columns.{col} of {source} must be numeric, got {v!r}"
                    )


def load_contract(path: str) -> Contract:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"contract not found: {path}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    _validate(raw, str(path))
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
