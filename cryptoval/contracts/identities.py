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


def _relative_mismatch(left: pd.Series, right: pd.Series, rtol: float) -> tuple[int, float]:
    """Count rows where left deviates from right by more than rtol (relative to right).

    Nulls are *unknown* and are dropped: the vendor legitimately omits metrics. A zero
    expected value is not unknown: zero vs non-zero is a mismatch (reported as infinite
    deviation), zero vs zero agrees. No division by the expected value, so no warnings.
    """
    both = pd.concat([left, right], axis=1).dropna()
    if both.empty:
        return 0, 0.0
    a = both.iloc[:, 0].to_numpy(dtype=float)
    b = both.iloc[:, 1].to_numpy(dtype=float)
    diff = np.abs(a - b)
    scale = np.abs(b)
    bad = diff > rtol * scale
    rel = np.divide(diff, scale, out=np.full_like(diff, np.inf), where=scale > 0)
    rel[diff == 0] = 0.0
    return int(bad.sum()), float(rel.max())


def check_mvrv_identity(df: pd.DataFrame, rtol: float = DEFAULT_RTOL) -> CheckResult:
    needed = {"CapMrktCurUSD", "CapMVRVCur", "realized_cap"}
    if not needed.issubset(df.columns):
        return CheckResult("identity_mvrv", True, "columns absent")
    raw_mvrv = pd.to_numeric(df["CapMVRVCur"], errors="coerce")
    # Non-null MVRV <= 0 is impossible (market cap / realized cap, both positive): report it.
    n_nonpositive = int((raw_mvrv <= 0).sum())
    mvrv = raw_mvrv.where(lambda s: s > 0)  # guard: never divide by non-positive MVRV
    implied = pd.to_numeric(df["CapMrktCurUSD"], errors="coerce") / mvrv
    n_bad, worst = _relative_mismatch(
        pd.to_numeric(df["realized_cap"], errors="coerce"), implied, rtol
    )
    problems = []
    if n_nonpositive:
        problems.append(
            f"{n_nonpositive} rows with non-positive CapMVRVCur (realized_cap undefined)"
        )
    if n_bad:
        problems.append(f"{n_bad} rows where realized_cap != market_cap/mvrv (worst {worst:.2e})")
    return CheckResult("identity_mvrv", not problems, "; ".join(problems))


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
    if not {col, group_col, date_col}.issubset(df.columns):
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
