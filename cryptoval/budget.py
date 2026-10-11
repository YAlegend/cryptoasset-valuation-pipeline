# cryptoval/budget.py
"""Byte-budget gate for BigQuery (spec section 6.3).

The free-only decision is enforced by two guards: a sandbox project with
no billing account, and this gate. Every query dry-runs first and passes
its estimate here; the gate also tracks cumulative bytes for the calendar
month, because the sandbox allowance is monthly, not per query.
"""

from __future__ import annotations

import datetime as dt

from cryptoval.manifest import RunManifest


class BudgetExceeded(RuntimeError):
    """Raised instead of running a query that would breach the byte budget."""


def _utc(now: dt.datetime) -> dt.datetime:
    """Return `now` in UTC. Naive datetimes have no defined month, so they are refused."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware; naive datetimes are ambiguous")
    return now.astimezone(dt.UTC)


class ByteBudgetGate:
    def __init__(self, manifest: RunManifest, monthly_budget: int) -> None:
        self.manifest = manifest
        self.monthly_budget = monthly_budget

    def remaining(self, now: dt.datetime) -> int:
        utc = _utc(now)
        used = self.manifest.bytes_scanned_in_month(utc.year, utc.month)
        return max(self.monthly_budget - used, 0)

    def guard(self, estimated_bytes: int, now: dt.datetime) -> None:
        if estimated_bytes < 0:
            raise ValueError("estimated_bytes must be non-negative")
        left = self.remaining(now)
        if estimated_bytes > left:
            utc = _utc(now)
            raise BudgetExceeded(
                f"query would scan {estimated_bytes} bytes but only {left} bytes "
                f"remain in the {utc:%Y-%m} budget of {self.monthly_budget}. "
                "Stage this extraction into a later month (spec section 6.3)."
            )
