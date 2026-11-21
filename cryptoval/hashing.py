"""Stable content hashing.

Used for bronze response fingerprints, SCD2 change detection and the
DefiLlama methodology guard. Must be stable across processes and runs, so
Python's salted builtin hash() is unsuitable.
"""

from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import json
from typing import Any

import numpy as np


def _encode(obj: Any) -> Any:
    # datetime is a subclass of date (and pd.Timestamp of datetime): check it first.
    if isinstance(obj, dt.datetime):
        return {"__type__": "datetime", "value": obj.isoformat()}
    if isinstance(obj, dt.date):
        return {"__type__": "date", "value": obj.isoformat()}
    if isinstance(obj, decimal.Decimal):
        return {"__type__": "decimal", "value": str(obj)}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"stable_hash cannot encode object of type {type(obj).__name__}")


def stable_hash(obj: Any) -> str:
    """sha256 hex of canonical JSON (sorted keys, compact separators, UTF-8).

    Supported: JSON primitives, dict, list/tuple, datetime/date/pandas.Timestamp and
    Decimal (type-tagged, so a date never collides with a plain string), and numpy
    scalars (converted to Python primitives). Anything else raises TypeError rather
    than being silently stringified.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_encode)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
