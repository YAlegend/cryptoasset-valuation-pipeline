"""Stable content hashing.

Used for bronze response fingerprints, SCD2 change detection and the
DefiLlama methodology guard. Must be stable across processes and runs, so
Python's salted builtin hash() is unsuitable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
