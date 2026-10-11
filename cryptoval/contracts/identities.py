"""Cross-column identity assertions (spec section 9.2).

Realized capitalisation is derived, not sourced: realized_cap =
CapMrktCurUSD / CapMVRVCur. If either input is redefined upstream, every
downstream cost-approach metric is silently wrong. These checks are the
tripwire for that, and they are blocking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.contracts.checks import CheckResult

DEFAULT_RTOL = 1e-6


def _relative_mismatch(
    left: pd.Series, right: pd.Series, rtol: float
) -> tuple[int, float]:
    both = pd.concat([left, right], axis=1).dropna()
    if both.empty:
        return 0, 0.0
    a = both.iloc[:, 0].to_numpy(dtype=float)
    b = both.iloc[:, 1].to_numpy(dtype=float)
    denom = np.where(np.abs(b) > 0, np.abs(b), np.nan)
    rel = np.abs(a - b) / denom
    rel = rel[~np.isnan(rel)]
    if rel.size == 0:
        return 0, 0.0
    return int((rel > rtol).sum()), float(rel.max())


def check_mvrv_identity(df: pd.DataFrame, rtol: float = DEFAULT_RTOL) -> CheckResult:
    needed = {"CapMrktCurUSD", "CapMVRVCur", "realized_cap"}
    if not needed.issubset(df.columns):
        return CheckResult("identity_mvrv", True, "columns absent")
    mvrv = pd.to_numeric(df["CapMVRVCur"], errors="coerce").where(lambda s: s > 0)
    implied = pd.to_numeric(df["CapMrktCurUSD"], errors="coerce") / mvrv
    n_bad, worst = _relative_mismatch(
        pd.to_numeric(df["realized_cap"], errors="coerce"), implied, rtol
    )
    return CheckResult(
        "identity_mvrv",
        n_bad == 0,
        ""
        if n_bad == 0
        else f"{n_bad} rows where realized_cap != market_cap/mvrv (worst {worst:.2e})",
    )


def check_supply_identity(df: pd.DataFrame, rtol: float = 1e-3) -> CheckResult:
    needed = {"SplyCur", "PriceUSD", "CapMrktCurUSD"}
    if not needed.issubset(df.columns):
        return CheckResult("identity_supply", True, "columns absent")
    implied = pd.to_numeric(df["SplyCur"], errors="coerce") * pd.to_numeric(
        df["PriceUSD"], errors="coerce"
    )
    n_bad, worst = _relative_mismatch(
        pd.to_numeric(df["CapMrktCurUSD"], errors="coerce"), implied, rtol
    )
    return CheckResult(
        "identity_supply",
        n_bad == 0,
        "" if n_bad == 0 else f"{n_bad} rows where supply*price != market_cap (worst {worst:.2e})",
    )


def check_monotonic_nondecreasing(
    df: pd.DataFrame, col: str, group_col: str = "asset", date_col: str = "metric_date"
) -> CheckResult:
    if col not in df.columns or group_col not in df.columns:
        return CheckResult(f"monotonic_{col}", True, "columns absent")
    offenders = []
    for key, grp in df.sort_values(date_col).groupby(group_col):
        series = pd.to_numeric(grp[col], errors="coerce").dropna()
        if (series.diff().dropna() < 0).any():
            offenders.append(str(key))
    return CheckResult(
        f"monotonic_{col}",
        not offenders,
        "" if not offenders else f"{col} decreases for: {offenders}",
    )


def check_cointime_complement(df: pd.DataFrame, rtol: float = 1e-9) -> CheckResult:
    needed = {"liveliness", "vaultedness"}
    if not needed.issubset(df.columns):
        return CheckResult("identity_cointime", True, "columns absent")
    total = pd.to_numeric(df["liveliness"], errors="coerce") + pd.to_numeric(
        df["vaultedness"], errors="coerce"
    )
    n_bad, worst = _relative_mismatch(total, pd.Series(1.0, index=df.index), rtol)
    return CheckResult(
        "identity_cointime",
        n_bad == 0,
        "" if n_bad == 0 else f"{n_bad} rows where liveliness+vaultedness != 1 (worst {worst:.2e})",
    )


def run_identity_checks(df: pd.DataFrame) -> list[CheckResult]:
    return [
        check_mvrv_identity(df),
        check_supply_identity(df),
        check_monotonic_nondecreasing(df, "thermocap"),
        check_cointime_complement(df),
    ]
