"""The three extraction queries and the only module that talks to BigQuery (spec 6.3).

Columns are pruned to the narrow fields each query needs and every query carries a
block_timestamp range, which prunes partitions through the public views (318 GB -> 5 GB for
one month of inputs, measured 2026-10-18).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import pandas as pd

DATASET = "bigquery-public-data.crypto_bitcoin"


def _bounds(start: dt.date, end: dt.date) -> tuple[str, str]:
    """Inclusive dates to a half-open timestamp interval."""
    return start.isoformat(), (end + dt.timedelta(days=1)).isoformat()


def spent_sql(start: dt.date, end: dt.date) -> str:
    lo, hi = _bounds(start, end)
    # Through transactions, not outputs: inputs carry the spent value themselves, so only
    # the creating block is needed, and this join bills 466 GB against 740 GB (Ruling 1).
    # GROUP BY hash collapses BIP30's two duplicated coinbase hashes to the later block.
    return f"""
WITH tx AS (
  SELECT `hash`, MAX(block_number) AS block_number, MAX(block_timestamp) AS block_timestamp
  FROM `{DATASET}.transactions`
  WHERE block_timestamp < TIMESTAMP('{hi}')
  GROUP BY `hash`
)
SELECT
  DATE(tx.block_timestamp) AS created_date,
  DATE(i.block_timestamp) AS spent_date,
  CAST(SUM(i.value) AS INT64) AS value_sat,
  CAST(SUM(i.value * (i.block_number - tx.block_number)) / 1e8 AS FLOAT64) AS coinblocks_destroyed,
  COUNT(*) AS n_inputs
FROM `{DATASET}.inputs` AS i
JOIN tx ON tx.`hash` = i.spent_transaction_hash
WHERE i.block_timestamp >= TIMESTAMP('{lo}') AND i.block_timestamp < TIMESTAMP('{hi}')
GROUP BY created_date, spent_date
"""


def created_sql(start: dt.date, end: dt.date) -> str:
    lo, hi = _bounds(start, end)
    return f"""
SELECT
  DATE(block_timestamp) AS created_date,
  CAST(SUM(value) AS INT64) AS value_sat,
  COUNT(*) AS n_outputs
FROM `{DATASET}.outputs`
WHERE block_timestamp >= TIMESTAMP('{lo}') AND block_timestamp < TIMESTAMP('{hi}')
GROUP BY created_date
"""


def blocks_sql(start: dt.date, end: dt.date) -> str:
    lo, hi = _bounds(start, end)
    # Per block, not per day: block timestamps are not monotonic in height, so a day's heights
    # are not a contiguous run and a (min, max) pair would misstate its coinblocks created.
    return f"""
SELECT number AS height, DATE(timestamp) AS block_date
FROM `{DATASET}.blocks`
WHERE timestamp >= TIMESTAMP('{lo}') AND timestamp < TIMESTAMP('{hi}')
"""


SQL_BUILDERS: dict[str, Callable[[dt.date, dt.date], str]] = {
    "btc_utxo_spent": spent_sql,
    "btc_utxo_created": created_sql,
    "btc_blocks": blocks_sql,
}


def plain_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """BigQuery DATE arrives as db_dtypes' extension type, which Parquet records by name and
    pandas cannot read back without db_dtypes imported. Bronze holds plain dates instead."""
    dates = [c for c in frame.columns if str(frame[c].dtype) == "dbdate"]
    return frame.astype({c: object for c in dates})


class BigQueryClient:
    """Dry-run and run against one named project, never the gcloud default."""

    def __init__(self, project: str) -> None:
        if not project or not project.strip():
            raise ValueError(
                "bigquery_project is not configured in config/settings.yml; refusing to fall "
                "back to the gcloud default project, which may carry a billing account"
            )
        from google.cloud import bigquery

        self._bigquery = bigquery
        self._client = bigquery.Client(project=project)

    def dry_run(self, sql: str) -> int:
        config = self._bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        return int(self._client.query(sql, job_config=config).total_bytes_processed or 0)

    def run(self, sql: str, max_bytes: int) -> tuple[pd.DataFrame, int]:
        # maximum_bytes_billed is BigQuery's own refusal, a third guard behind the sandbox
        # and our gate: a query whose real scan exceeds the dry run fails instead of billing.
        config = self._bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes)
        job = self._client.query(sql, job_config=config)
        frame = job.result().to_dataframe(create_bqstorage_client=False)
        return plain_dtypes(frame), int(job.total_bytes_billed or 0)
