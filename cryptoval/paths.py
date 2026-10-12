"""Single source of truth for storage locations.

Local filesystem and object storage differ only in the root URI, so every
consumer asks this resolver for paths instead of joining them itself.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

LAYERS = ("bronze", "silver", "gold", "marts")


@dataclass(frozen=True)
class PathResolver:
    root: str

    def _join(self, *parts: str) -> str:
        return "/".join([self.root.rstrip("/"), *parts])

    @property
    def is_remote(self) -> bool:
        return "://" in self.root

    def layer(self, name: str) -> str:
        if name not in LAYERS:
            raise ValueError(f"unknown layer: {name!r}; expected one of {LAYERS}")
        return self._join(name)

    def table(self, layer: str, name: str) -> str:
        return "/".join([self.layer(layer), name])

    def bronze_date_prefix(self, source: str, ingest_date: dt.date) -> str:
        return "/".join(
            [self.layer("bronze"), f"source={source}", f"ingest_date={ingest_date.isoformat()}"]
        )

    def bronze_partition(self, source: str, ingest_date: dt.date, run_id: str) -> str:
        """One partition per RUN, not per date.

        Date-only partitioning made a same-day re-run replace the partition wholesale, so
        re-running a subset of assets after a partial failure destroyed the response record
        of everything fetched earlier that day (spec 7.4, revised).
        """
        if not run_id or not str(run_id).strip():
            raise ValueError("run_id must be a non-empty identifier")
        return "/".join([self.bronze_date_prefix(source, ingest_date), f"run_id={run_id}"])
