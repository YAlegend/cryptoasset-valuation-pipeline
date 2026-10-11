from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from cryptoval.contracts.loader import Contract

# Spec 9.2 says "trailing average". It must be a WINDOW: an all-time expanding mean
# turns a decade of secular growth into a permanent anomaly — measured against the live
# Coin Metrics panel it flagged 68.8% of all asset-days (fix round 1, Ruling J).
TRAILING_WINDOW_DAYS = 30


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def check_required_columns(df: pd.DataFrame, contract: Contract) -> CheckResult:
    missing = [c for c in contract.required_columns if c not in df.columns]
    return CheckResult(
        "required_columns", not missing, "" if not missing else f"missing columns: {missing}"
    )


def check_unique(df: pd.DataFrame, contract: Contract) -> CheckResult:
    keys = [k for k in contract.unique_on if k in df.columns]
    if not keys:
        return CheckResult("uniqueness", True, "no key columns present")
    n_dupes = int(df.duplicated(subset=keys).sum())
    return CheckResult(
        "uniqueness", n_dupes == 0, "" if n_dupes == 0 else f"{n_dupes} duplicate rows on {keys}"
    )


def check_not_null(df: pd.DataFrame, contract: Contract) -> CheckResult:
    offenders = {
        c: int(df[c].isna().sum())
        for c in contract.required_columns
        if c in df.columns and df[c].isna().any()
    }
    return CheckResult("not_null", not offenders, "" if not offenders else f"nulls: {offenders}")


def check_ranges(df: pd.DataFrame, contract: Contract) -> CheckResult:
    offenders = []
    for col, bounds in contract.ranges.items():
        if col not in df.columns:
            continue
        raw = df[col]
        numeric = pd.to_numeric(raw, errors="coerce")
        unparseable = raw.notna() & numeric.isna()
        if unparseable.any():
            offenders.append(
                f"{col} has {int(unparseable.sum())} non-numeric value(s), "
                f"e.g. {raw[unparseable].iloc[0]!r}"
            )
        series = numeric.dropna()
        if "minimum" in bounds and (series < bounds["minimum"]).any():
            offenders.append(f"{col} < {bounds['minimum']}")
        if "maximum" in bounds and (series > bounds["maximum"]).any():
            offenders.append(f"{col} > {bounds['maximum']}")
    return CheckResult("ranges", not offenders, "; ".join(offenders))


def check_freshness(
    df: pd.DataFrame, contract: Contract, date_col: str, now: dt.datetime
) -> CheckResult:
    if contract.freshness_max_delay_hours is None or df.empty:
        return CheckResult("freshness", True, "no SLA or no rows")
    latest = pd.to_datetime(df[date_col], utc=True).max()
    delay_h = (now - latest.to_pydatetime()).total_seconds() / 3600
    ok = delay_h <= contract.freshness_max_delay_hours
    return CheckResult(
        "freshness", ok, "" if ok else f"latest {latest.date()} is {delay_h:.1f}h old"
    )


def check_volume_anomaly(
    df: pd.DataFrame, contract: Contract, value_col: str, date_col: str
) -> CheckResult:
    threshold = contract.volume_anomaly_threshold
    if threshold is None or len(df) < 5:
        return CheckResult("volume_anomaly", True, "no SLA or too few rows")
    ordered = df.sort_values(date_col)
    values = pd.to_numeric(ordered[value_col], errors="coerce")
    trailing = values.shift(1).rolling(TRAILING_WINDOW_DAYS, min_periods=3).mean()
    # trailing means of zero would give an infinite deviation; treat them as unknown.
    safe_trailing = trailing.where(trailing != 0)
    deviation = (values - safe_trailing).abs() / safe_trailing
    worst = deviation.max()
    if pd.isna(worst):
        return CheckResult("volume_anomaly", True, "no valid trailing baseline")
    if worst <= threshold:
        return CheckResult("volume_anomaly", True, "")
    idx = deviation.idxmax()
    return CheckResult(
        "volume_anomaly",
        False,
        f"anomalous {value_col}={values.loc[idx]:.0f} deviates {worst:.1%} from trailing mean",
    )


def run_checks(df: pd.DataFrame, contract: Contract, now: dt.datetime) -> list[CheckResult]:
    results = [
        check_required_columns(df, contract),
        check_unique(df, contract),
        check_not_null(df, contract),
        check_ranges(df, contract),
    ]
    if "metric_date" in df.columns:
        results.append(check_freshness(df, contract, "metric_date", now))
    return results
