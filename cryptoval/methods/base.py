"""The method contract (spec 10.1).

Every method is a pure function over the panel with one signature, so estimates from
different families are directly comparable -- and the comparison IS the research output.

Two properties are computed here rather than restated by each method, because a property
restated eleven times is a property forgotten once:

* APPLICABILITY, from declared dependencies and measured density. The free tier is not
  uniform: BNB's market cap is 19% dense and TRX has no free MVRV at all. A fit on sparse
  data converges and returns a confident number, which is worse than a refusal.
* CIRCULARITY. Several panel columns are themselves computed from market cap. A method
  consuming one and returning a price in exchange is algebraically handing back the market
  price: QTM with panel velocity reduces to price/circulating_fraction, and every residual
  is zero by construction. The flag travels into the method card and the report so such a
  number is never presented as independent evidence (spec 14, "method circularity").
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd

# Columns in fct_asset_metrics_daily that are derived FROM market cap.
MARKET_CAP_DERIVED = frozenset(
    {
        "velocity_reported",  # volume / market_cap
        "velocity_total",  # volume / market_cap
        "mvrv",  # market_cap / realized_cap
        "mvtv",  # market_cap / thermocap
        "realized_cap",  # market_cap / mvrv
        "investor_cap_usd",  # realized_cap - thermocap
    }
)


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    family: str
    requires: tuple[str, ...]
    min_density: float
    min_rows: int
    params: dict[str, Any]
    citation: str
    returns_fair_value: bool = True


@dataclass(frozen=True)
class Estimate:
    symbol: str
    metric_date: dt.date
    method_id: str
    param_set_hash: str
    fair_value: float | None
    band_low: float | None
    band_high: float | None
    inputs_used: dict[str, Any]
    diagnostics: dict[str, Any]
    applicable: bool
    reason: str = ""
    panel_vintage_ts: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if not self.applicable and self.fair_value is not None:
            raise ValueError(
                f"{self.method_id}/{self.symbol}: an inapplicable estimate cannot carry a "
                f"fair_value (got {self.fair_value!r}); that is how a refusal becomes a number"
            )
        if self.band_low is not None and self.band_high is not None:
            if self.band_low > self.band_high:
                raise ValueError(
                    f"{self.method_id}/{self.symbol}: band_low {self.band_low} is above "
                    f"band_high {self.band_high}"
                )


def circularity(spec: MethodSpec) -> tuple[str, ...]:
    """The market-cap-derived columns this method consumes, in declaration order."""
    return tuple(c for c in spec.requires if c in MARKET_CAP_DERIVED)


def covered_span(spec: MethodSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """Trim to the span in which every required column has started.

    Coverage is not uniform in time: CoinGecko's keyless tier serves 365 days, so
    volume_total_usd begins in 2025 while Coin Metrics' columns begin in 2015. Density
    measured over the whole history then reads a late-starting column as 8% sparse and
    refuses a method that its own trailing window can answer honestly. The pre-coverage
    void is not sparsity, so it is dropped before applicability is judged -- and the
    method sees only the span it could legitimately fit on.
    """
    starts = [
        frame[c].first_valid_index() for c in spec.requires if c in frame.columns
    ]
    if not starts or any(s is None for s in starts):
        # A wholly empty required column: leave the frame alone so the 0% density is
        # what gets reported, rather than an empty-slice reason that hides the cause.
        return frame
    return frame.iloc[max(starts) :].reset_index(drop=True)


def resolve_applicability(spec: MethodSpec, frame: pd.DataFrame) -> tuple[bool, str]:
    """Decide whether ``spec`` can honestly run on ``frame``.

    Returns (applicable, reason). The reason is empty when applicable, and otherwise
    carries the measurement that made it inapplicable -- a reason without a number is
    not actionable by whoever reads the method card.
    """
    if frame.empty:
        return False, "no rows in the panel slice"

    missing = [c for c in spec.requires if c not in frame.columns]
    if missing:
        return False, f"required column(s) absent from the panel: {missing}"

    if len(frame) < spec.min_rows:
        return False, f"only {len(frame)} rows, below the method's minimum of {spec.min_rows}"

    for column in spec.requires:
        density = float(frame[column].notna().mean())
        if density < spec.min_density:
            return False, (
                f"{column} is {density:.1%} dense, below this method's floor "
                f"{spec.min_density:.1%}"
            )
    return True, ""


@runtime_checkable
class Method(Protocol):
    spec: MethodSpec

    def estimate(self, frame: pd.DataFrame, params: dict[str, Any]) -> list[Estimate]: ...
