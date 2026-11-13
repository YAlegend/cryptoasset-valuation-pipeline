from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
import pandas as pd

from cryptoval.config import Asset


@dataclass(frozen=True)
class FetchResult:
    frame: pd.DataFrame
    rows_in: int
    # Per-item failures a source survived. A non-empty errors with a non-empty frame is a
    # PARTIAL fetch: the rows present are real, and what is missing is named here rather
    # than discarded (fix round 1, Ruling D).
    errors: tuple[str, ...] = ()


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


RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def get_with_retry(
    client: httpx.Client,
    url: str,
    params: dict | None = None,
    max_retries: int = 5,
    initial_delay: float = 1.0,
) -> dict:
    """GET ``url`` and return its JSON, retrying transient failures.

    Retries 429 and 5xx with doubling backoff. Any other error status fails
    fast via ``raise_for_status``; the final retryable failure is re-raised.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        response = client.get(url, params=params)
        if response.status_code in RETRYABLE_STATUS:
            if attempt == max_retries - 1:
                response.raise_for_status()
            time.sleep(delay)
            delay *= 2
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable")
