from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from cryptoval.contracts.loader import Contract

# Spec 9.2 says "trailing average". It must be a WINDOW: an all-time expanding mean
# turns a decade of secular growth into a permanent anomaly — measured against the live
# Coin Metrics panel it flagged 68.8% of all asset-days (fix round 1, Ruling J).
# ROWS, not days: for a dense daily panel the two coincide, but a gapped series compares
# against whatever its previous 30 observations were, however long ago.
TRAILING_WINDOW_ROWS = 30


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


def check_null_rate(df: pd.DataFrame, contract: Contract) -> CheckResult:
    """Fail when a declared column is far nullier than the vendor's normal sparsity.

    This is the detector spec 9.2 asks for: a metric that silently leaves the free tier
    arrives present-but-null, which passes required_columns, not_null, ranges and both
    identity assertions (they drop nulls before comparing).
    """
    if df.empty:
        return CheckResult("null_rate", True, "no rows")
    offenders = []
    for col, ceiling in contract.null_rate_bounds.items():
        if col not in df.columns:
            continue
        rate = float(df[col].isna().mean())
        if rate > ceiling:
            offenders.append(f"{col} is {rate:.1%} null, ceiling {ceiling:.1%}")
    return CheckResult("null_rate", not offenders, "; ".join(offenders))


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
    trailing = values.shift(1).rolling(TRAILING_WINDOW_ROWS, min_periods=3).mean()
    # trailing means of zero would give an infinite deviation; treat them as unknown.
    safe_trailing = trailing.where(trailing != 0)
    deviation = (values - safe_trailing).abs() / safe_trailing
    # Only the newest observation is judged: this runs on every validate, so scanning the
    # whole history would re-report a years-old event forever (see
    # docs/runbooks/impossible-metric-value.md for why that narrowing is safe).
    latest = deviation.iloc[-1]
    if pd.isna(latest):
        return CheckResult("volume_anomaly", True, "no valid trailing baseline")
    if latest <= threshold:
        return CheckResult("volume_anomaly", True, "")
    return CheckResult(
        "volume_anomaly",
        False,
        f"anomalous {value_col}={values.iloc[-1]:.0f} on "
        f"{pd.Timestamp(ordered[date_col].iloc[-1]).date()} deviates {latest:.1%} "
        "from trailing mean",
    )


def run_checks(df: pd.DataFrame, contract: Contract, now: dt.datetime) -> list[CheckResult]:
    results = [
        check_required_columns(df, contract),
        check_unique(df, contract),
        check_not_null(df, contract),
        check_null_rate(df, contract),
        check_ranges(df, contract),
    ]
    if "metric_date" in df.columns:
        results.append(check_freshness(df, contract, "metric_date", now))
    return results
