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


def _mapping(value: Any, where: str) -> dict[str, Any]:
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping, got {type(value).__name__}: {value!r}")
    return value


def _validate(raw: dict[str, Any], source: str) -> None:
    _reject_unknown(raw, {"contract", "schema", "quality", "versioning"}, f"top level of {source}")
    schema = _mapping(raw.get("schema"), f"schema of {source}")
    _reject_unknown(schema, {"required", "columns"}, f"schema of {source}")
    required = schema.get("required")
    if required is not None and not isinstance(required, list):
        raise ValueError(f"schema.required of {source} must be a list, got {required!r}")
    quality = _mapping(raw.get("quality"), f"quality of {source}")
    _reject_unknown(quality, set(_QUALITY_KEYS), f"quality of {source}")
    for section, allowed in _QUALITY_KEYS.items():
        if allowed is None or section not in quality:
            continue
        where = f"quality.{section} of {source}"
        body = _mapping(quality[section], where)
        _reject_unknown(body, allowed, where)
    combination = _mapping(quality.get("uniqueness"), "quality.uniqueness").get("combination")
    if combination is not None and not isinstance(combination, list):
        raise ValueError(
            f"quality.uniqueness.combination of {source} must be a list, got {combination!r}"
        )
    columns = _mapping(schema.get("columns"), f"schema.columns of {source}")
    for col, spec in columns.items():
        spec = _mapping(spec, f"schema.columns.{col} of {source}")
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
