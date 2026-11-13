"""Quarantine for vendor values that are impossible rather than merely extreme.

Spec 9.1. The free tier served reported-spot-volume figures up to 7e55 USD on
2026-10-11 (14 rows) against a largest credible value of 7.5e10 in the same panel.
Carrying such a value poisons every velocity and NVT figure computed from it;
dropping the whole row discards the healthy market cap sitting beside it. So the
offending CELL is nulled and the raw value recorded here, with bronze untouched —
this is the one place silver deviates from the vendor's numbers, and it is logged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.contracts.loader import Contract

QUARANTINE_COLUMNS = [
    "source",
    "entity",
    "metric_date",
    "column",
    "raw_value",
    "reason",
    "run_id",
    "quarantined_at",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=QUARANTINE_COLUMNS)


def quarantine_impossible(
    df: pd.DataFrame,
    contract: Contract,
    source: str,
    run_id: str,
    key_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (clean, quarantined) against the contract's range bounds.

    ``clean`` is ``df`` with every out-of-bounds cell set to NaN. ``quarantined``
    carries one row per offending cell, including the raw value, so the panel can
    always be reconciled against what the vendor actually sent.
    """
    if df.empty:
        return df.copy(), _empty()

    clean = df.copy()
    records: list[dict] = []
    now = pd.Timestamp.now(tz="UTC")
    entity_col = next((c for c in key_cols if c in df.columns), None)

    for column, bounds in contract.ranges.items():
        if column not in clean.columns:
            continue
        numeric = pd.to_numeric(clean[column], errors="coerce")
        # A null is a gap, governed by the null-rate ceiling. Only a value the vendor
        # supplied that cannot be true belongs here.
        offending = pd.Series(False, index=clean.index)
        reasons: dict = {}
        if "minimum" in bounds:
            below = numeric.notna() & (numeric < bounds["minimum"])
            offending |= below
            for idx in clean.index[below]:
                reasons[idx] = f"below contract minimum {bounds['minimum']:g}"
        if "maximum" in bounds:
            above = numeric.notna() & (numeric > bounds["maximum"])
            offending |= above
            for idx in clean.index[above]:
                reasons[idx] = f"above contract maximum {bounds['maximum']:g}"
        if not offending.any():
            continue
        for idx in clean.index[offending]:
            records.append(
                {
                    "source": source,
                    "entity": clean.loc[idx, entity_col] if entity_col else None,
                    "metric_date": clean.loc[idx, "metric_date"]
                    if "metric_date" in clean.columns
                    else pd.NaT,
                    "column": column,
                    "raw_value": float(numeric.loc[idx]),
                    "reason": reasons[idx],
                    "run_id": run_id,
                    "quarantined_at": now,
                }
            )
        # float, not pd.NA: stable_hash cannot encode NAType, so an object-dtype column
        # here would silently break SCD2 change detection (Plan 1, Ruling A).
        clean[column] = numeric.astype("float64").mask(offending, np.nan)

    quarantined = pd.DataFrame(records, columns=QUARANTINE_COLUMNS) if records else _empty()
    return clean, quarantined
