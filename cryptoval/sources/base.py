from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
import pandas as pd

from cryptoval.config import Asset


@dataclass(frozen=True)
class FetchResult:
    frame: pd.DataFrame
    rows_in: int


@runtime_checkable
class Source(Protocol):
    name: str
    contract_path: str

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult: ...
