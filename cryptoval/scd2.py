"""Bi-temporal (SCD Type 2) merge for the silver panel.

Coin Metrics revises published metrics, so the panel carries two time
axes: metric_date (what the value describes) and the
effective_start_ts/effective_end_ts pair (when we knew it). Research
queries use as_of_view so a backtest sees only what was knowable at the
time; the dashboard uses current_view (spec section 8.2).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from cryptoval.hashing import stable_hash

FAR_FUTURE = pd.Timestamp("9999-12-31", tz="UTC")
META_COLUMNS = ["effective_start_ts", "effective_end_ts", "is_current", "row_hash"]


def add_row_hash(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    present = [c for c in value_cols if c in out.columns]
    out["row_hash"] = [
        stable_hash({c: row[c] for c in present}) for _, row in out[present].iterrows()
    ]
    return out


def _dedupe_incoming(incoming: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    # A paginated response can repeat a key. Without this, the merge would
    # open two current rows for one key (Review Focus 2).
    return incoming.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def merge_scd2(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    key: list[str],
    value_cols: list[str],
    as_of: dt.datetime,
) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of)
    fresh = add_row_hash(_dedupe_incoming(incoming, key), value_cols)

    if existing is None or existing.empty:
        opened = fresh.copy()
        opened["effective_start_ts"] = as_of_ts
        opened["effective_end_ts"] = FAR_FUTURE
        opened["is_current"] = True
        return opened.reset_index(drop=True)

    history = existing.copy()
    current = history[history["is_current"]]
    lookup = current.set_index(key)["row_hash"].to_dict()

    to_open: list[pd.DataFrame] = []
    keys_to_close: list[tuple] = []

    for _, row in fresh.iterrows():
        k = tuple(row[c] for c in key)
        known = lookup.get(k if len(k) > 1 else k[0])
        if known == row["row_hash"]:
            continue
        if known is not None:
            keys_to_close.append(k)
        to_open.append(row.to_frame().T)

    if keys_to_close:
        mask = history["is_current"] & history.set_index(key).index.isin(
            [k if len(k) > 1 else k[0] for k in keys_to_close]
        )
        history.loc[mask, "effective_end_ts"] = as_of_ts
        history.loc[mask, "is_current"] = False

    if not to_open:
        return history.reset_index(drop=True)

    opened = pd.concat(to_open, ignore_index=True)
    opened["effective_start_ts"] = as_of_ts
    opened["effective_end_ts"] = FAR_FUTURE
    opened["is_current"] = True
    merged = pd.concat([history, opened], ignore_index=True)
    merged["is_current"] = merged["is_current"].astype(bool)
    return merged


def as_of_view(df: pd.DataFrame, at: dt.datetime) -> pd.DataFrame:
    at_ts = pd.Timestamp(at)
    mask = (df["effective_start_ts"] <= at_ts) & (df["effective_end_ts"] > at_ts)
    return df[mask].reset_index(drop=True)


def current_view(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_current"]].reset_index(drop=True)


def reconcile_report(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    key: list[str],
    value_cols: list[str],
) -> dict[str, int]:
    """Compare a refetch against stored state without writing anything.

    Scheduled runs only refetch a trailing window, so a revision landing
    outside it would otherwise be invisible (Review Focus 4). The reconcile
    command runs this over an arbitrary range and reports the drift.
    """
    fresh = add_row_hash(_dedupe_incoming(incoming, key), value_cols)
    if existing is None or existing.empty:
        return {"new": len(fresh), "changed": 0, "unchanged": 0}

    current = existing[existing["is_current"]]
    lookup = current.set_index(key)["row_hash"].to_dict()

    new = changed = unchanged = 0
    for _, row in fresh.iterrows():
        k = tuple(row[c] for c in key)
        known = lookup.get(k if len(k) > 1 else k[0])
        if known is None:
            new += 1
        elif known == row["row_hash"]:
            unchanged += 1
        else:
            changed += 1
    return {"new": new, "changed": changed, "unchanged": unchanged}
