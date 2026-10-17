"""The log levels panel, and the dates a pair is actually testable on.

LEVELS, not returns and not ratios. Cointegration is a statement about two
non-stationary LEVEL series sharing a stochastic trend; differencing them away is
exactly what the test is built to avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.econometrics.base import PairSpec, Unusable
from cryptoval.scoring.targets import _require_contiguous_days

# Log series name -> the panel column it is the log of. Non-positive values become NaN:
# an asset with no transactions that day has no log of zero, and -inf in a regression is
# not a measurement.
LEVEL_SERIES: dict[str, str] = {
    "log_market_cap": "market_cap_usd",
    "log_realized_cap": "realized_cap",
    "log_volume": "volume_total_usd",
    "log_addresses": "active_addresses",
}

# Two years. Below this a cointegration test on daily data is reporting a trend, not an
# equilibrium, and the refusal names the count so a reader sees which assets fell out.
# Measured on the 2026-10-14 vintage: 20 of 24 assets clear it on realized_cap; DOT (653)
# and BNB (647) do not, TRX has no realized_cap at all and SOL has 363 rows.
MIN_PAIR_ROWS = 730


def build_levels(panel: pd.DataFrame) -> pd.DataFrame:
    """(symbol, metric_date, <log series per LEVEL_SERIES>, panel_vintage_ts)."""
    keep = ["symbol", "metric_date"]
    if "panel_vintage_ts" in panel.columns:
        keep.append("panel_vintage_ts")
    out = panel[keep].copy()
    for name, column in LEVEL_SERIES.items():
        values = pd.to_numeric(panel[column], errors="coerce").astype("float64")
        out[name] = np.log(values.where(values > 0.0))
    return out.sort_values(["symbol", "metric_date"]).reset_index(drop=True)


def pair_rows(levels: pd.DataFrame, symbol: str, spec: PairSpec) -> pd.DataFrame:
    """The dates where BOTH series are finite, with y, x and the constrained ratio.

    Raises Unusable, with the measurement, when the symbol is absent, when the
    intersection is shorter than MIN_PAIR_ROWS, or when it has an interior gap. A gap is
    refused rather than stitched: a stitched series makes a one-day lag mean one thing
    before the hole and another after it, and nothing downstream could detect that.
    """
    group = levels[levels["symbol"] == symbol]
    if group.empty:
        raise Unusable(f"{spec.pair_id}: {symbol} is not in the panel")

    rows = group[["metric_date", spec.dependent, spec.independent]].rename(
        columns={spec.dependent: "y", spec.independent: "x"}
    )
    rows = rows[rows["y"].notna() & rows["x"].notna()].reset_index(drop=True)

    if len(rows) < MIN_PAIR_ROWS:
        raise Unusable(
            f"{spec.pair_id}: {symbol} has {len(rows)} rows where both "
            f"{spec.dependent} and {spec.independent} are finite; "
            f"{MIN_PAIR_ROWS} are needed for a cointegration test on daily data"
        )

    # The intersection, not the panel, is what the test reads, so contiguity is checked
    # here rather than inherited from the panel's own guard. Its ValueError becomes a
    # refusal: a gap in one asset's fundamental must not end the run for the others.
    try:
        _require_contiguous_days(rows.assign(symbol=symbol))
    except ValueError as exc:
        raise Unusable(f"{spec.pair_id}: {exc}") from exc

    rows["ratio"] = rows["y"] - spec.forced_coefficient * rows["x"]
    return rows
