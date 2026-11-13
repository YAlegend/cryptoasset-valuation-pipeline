"""Append-only record of pipeline runs (spec section 9.4)."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from cryptoval.paths import PathResolver

COLUMNS = [
    "run_id",
    "git_sha",
    "source",
    "params",
    "started_at",
    "ended_at",
    "rows_in",
    "rows_out",
    "bytes_scanned",
    "status",
    "error",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class RunRecord:
    run_id: str
    git_sha: str
    source: str
    params: dict[str, Any]
    started_at: dt.datetime
    ended_at: dt.datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RunManifest:
    def __init__(self, resolver: PathResolver) -> None:
        self._path = pathlib.Path(resolver.table("gold", "run_manifest.jsonl"))

    def start(
        self, source: str, params: dict[str, Any], started_at: dt.datetime | None = None
    ) -> RunRecord:
        return RunRecord(
            run_id=str(uuid.uuid4()),
            git_sha=_git_sha(),
            source=source,
            params=params,
            started_at=started_at or dt.datetime.now(dt.UTC),
        )

    def finish(
        self,
        record: RunRecord,
        rows_in: int,
        rows_out: int,
        status: str,
        bytes_scanned: int = 0,
        error: str | None = None,
    ) -> None:
        record.ended_at = dt.datetime.now(dt.UTC)
        row = {
            "run_id": record.run_id,
            "git_sha": record.git_sha,
            "source": record.source,
            "params": record.params,
            "started_at": record.started_at.isoformat(),
            "ended_at": record.ended_at.isoformat(),
            "rows_in": rows_in,
            "rows_out": rows_out,
            "bytes_scanned": bytes_scanned,
            "status": status,
            "error": error,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def read(self) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame(columns=COLUMNS)
        with self._path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        return pd.DataFrame(rows, columns=COLUMNS)

    def bytes_scanned_in_month(self, year: int, month: int) -> int:
        df = self.read()
        if df.empty:
            return 0
        started = pd.to_datetime(df["started_at"], utc=True)
        mask = (started.dt.year == year) & (started.dt.month == month)
        return int(df.loc[mask, "bytes_scanned"].fillna(0).sum())
