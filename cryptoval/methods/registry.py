"""Method discovery, execution and persistence (spec 10.1, 8.1).

A method whose inputs are unavailable produces ONE refusal row rather than nothing:
"this method cannot be applied to this asset, and here is the measurement that says so"
is a research finding, and a silent gap is not. Same for a method that raises -- the
exception becomes a recorded refusal so one broken method cannot void a whole run.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pandas as pd
import yaml

from cryptoval.hashing import stable_hash
from cryptoval.methods.base import (
    Estimate,
    Method,
    MethodSpec,
    circularity,
    covered_span,
    resolve_applicability,
)

ESTIMATE_COLUMNS = [
    "symbol",
    "metric_date",
    "method_id",
    "param_set_hash",
    "fair_value",
    "band_low",
    "band_high",
    "applicable",
    "reason",
    "circular_inputs",
    "inputs_used",
    "diagnostics",
    "panel_vintage_ts",
]

METHODS: dict[str, Method] = {}


def register_instance(method: Method) -> Method:
    """Add an already-built method to the registry, refusing a duplicate id.

    Used where one class serves two specs -- QTM ships twice, once with an exogenous
    velocity and once with the panel's (spec 10.2).
    """
    if method.spec.method_id in METHODS:
        raise ValueError(f"method id already registered: {method.spec.method_id}")
    METHODS[method.spec.method_id] = method
    return method


def register(cls: type) -> type:
    """Class decorator: instantiate and register, returning the class unchanged.

    It registers an INSTANCE, not the class: run_methods calls method.estimate(...), which
    on a class would be an unbound call failing on the missing self. The class is returned
    untouched so tests can still construct it directly.
    """
    register_instance(cls())
    return cls


def param_set_hash(params: dict[str, Any]) -> str:
    """Stable hash of a parameter set, so a DCF at 12% and at 16% coexist as rows."""
    return stable_hash(params)


def load_method_specs(config_dir: str = "config/methods") -> dict[str, MethodSpec]:
    specs: dict[str, MethodSpec] = {}
    for path in sorted(pathlib.Path(config_dir).glob("*.yml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = MethodSpec(
            method_id=raw["method_id"],
            family=raw["family"],
            requires=tuple(raw.get("requires", []) or []),
            min_density=float(raw.get("min_density", 0.5)),
            min_rows=int(raw.get("min_rows", 30)),
            params=dict(raw.get("params", {}) or {}),
            citation=raw.get("citation", ""),
            returns_fair_value=bool(raw.get("returns_fair_value", True)),
        )
        if spec.method_id in specs:
            raise ValueError(f"duplicate method_id {spec.method_id} in {path}")
        specs[spec.method_id] = spec
    return specs


def _row(estimate: Estimate, circular: tuple[str, ...]) -> dict:
    return {
        "symbol": estimate.symbol,
        "metric_date": estimate.metric_date,
        "method_id": estimate.method_id,
        "param_set_hash": estimate.param_set_hash,
        "fair_value": estimate.fair_value,
        "band_low": estimate.band_low,
        "band_high": estimate.band_high,
        "applicable": estimate.applicable,
        "reason": estimate.reason,
        # Pipe-joined rather than a list column: Parquet list columns are awkward to
        # query from SQL and this is read far more often than it is parsed.
        "circular_inputs": "|".join(circular),
        "inputs_used": json.dumps(estimate.inputs_used, default=str, sort_keys=True),
        "diagnostics": json.dumps(estimate.diagnostics, default=str, sort_keys=True),
        "panel_vintage_ts": estimate.panel_vintage_ts,
    }


def _refusal(symbol: str, spec: MethodSpec, reason: str, frame: pd.DataFrame) -> dict:
    latest = frame["metric_date"].max() if not frame.empty else None
    vintage = (
        frame["panel_vintage_ts"].max()
        if "panel_vintage_ts" in frame.columns and not frame.empty
        else None
    )
    return _row(
        Estimate(
            symbol=symbol,
            metric_date=latest,
            method_id=spec.method_id,
            param_set_hash=param_set_hash(spec.params),
            fair_value=None,
            band_low=None,
            band_high=None,
            inputs_used={},
            diagnostics={},
            applicable=False,
            reason=reason,
            panel_vintage_ts=vintage,
        ),
        circularity(spec),
    )


def run_methods(
    panel: pd.DataFrame,
    methods: dict[str, Method],
    method_ids: list[str] | None = None,
) -> pd.DataFrame:
    selected = method_ids or sorted(methods)
    rows: list[dict] = []

    for method_id in selected:
        method = methods[method_id]
        spec = method.spec
        circular = circularity(spec)
        for symbol, group in panel.groupby("symbol", sort=True):
            frame = covered_span(
                spec, group.sort_values("metric_date").reset_index(drop=True)
            )
            ok, reason = resolve_applicability(spec, frame)
            if not ok:
                rows.append(_refusal(str(symbol), spec, reason, frame))
                continue
            try:
                estimates = method.estimate(frame, dict(spec.params))
            except Exception as exc:
                # One broken method must not void a whole run; the failure becomes a
                # recorded refusal naming the exception type and its message.
                rows.append(
                    _refusal(str(symbol), spec, f"method raised {type(exc).__name__}: {exc}", frame)
                )
                continue
            rows.extend(_row(e, circular) for e in estimates)

    return pd.DataFrame(rows, columns=ESTIMATE_COLUMNS)
