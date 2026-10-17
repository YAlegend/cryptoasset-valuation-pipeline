"""The target and the features (spec 11.2).

TWO RULES, settled here so that seven models do not each decide them:

* The target is a FORWARD LOG RETURN, never a level. log(P[t+h] / P[t]), NaN in the last
  h rows because there is no future price to divide by. Spec 11.2 is explicit that a model
  predicting levels from levels is the dissertation's error restated.
* Every feature is LAGGED ONE DAY. The features are this project's own fair values, and
  two classes of them have already seen day t's price: crypto_qtm_panel_velocity returns
  the market price to machine epsilon, and mayer_multiple and puell_multiple are
  documented as using an inclusive window because their published definitions are
  inclusive. Lagging everything makes that leak impossible by construction instead of by
  per-method audit, at a cost of one day of signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS: tuple[int, ...] = (1, 7, 30, 90, 365)

FEATURE_LAG_DAYS = 1

# A signal flatter than this carries no information a model could use, and at least one
# is flat BY CONSTRUCTION (a circular method's fair value is the price). Dropping it with
# its measured variance beats handing a regression a constant column.
MIN_SIGNAL_VARIANCE = 1e-12


def forward_log_return(price: pd.Series, horizon: int) -> pd.Series:
    """log(P[t+h] / P[t]), with the last h rows NaN.

    Positive prices only: a non-positive or missing price yields NaN rather than a
    complex or infinite return.
    """
    p = pd.to_numeric(price, errors="coerce").astype("float64")
    p = p.where(p > 0.0)
    return np.log(p.shift(-horizon) / p)


def signal_matrix(estimates: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """One column per method: log(fair_value / price), NOT yet lagged.

    The ratio rather than the level, because a fair value in dollars is not comparable
    across assets and a model fitted per asset would otherwise learn the price scale.
    """
    usable = estimates[
        estimates["applicable"].astype("boolean").fillna(False) & estimates["fair_value"].notna()
    ].copy()
    usable["fair_value"] = pd.to_numeric(usable["fair_value"], errors="coerce")
    usable = usable[usable["fair_value"] > 0.0]

    wide = usable.pivot_table(
        index=["symbol", "metric_date"],
        columns="method_id",
        values="fair_value",
        aggfunc="median",  # a grid method emits one row per param set; the median is the cell
    ).reset_index()

    merged = price[["symbol", "metric_date", "price_usd"]].merge(
        wide, on=["symbol", "metric_date"], how="left"
    )
    methods = [c for c in wide.columns if c not in ("symbol", "metric_date")]
    base = pd.to_numeric(merged["price_usd"], errors="coerce").where(lambda s: s > 0.0)
    for method_id in methods:
        merged[method_id] = np.log(merged[method_id] / base)
    return merged.drop(columns=["price_usd"])


def _require_contiguous_days(panel: pd.DataFrame) -> None:
    """Raise unless each symbol's dates step by exactly one day.

    shift(1) and shift(-h) count rows, so they are days only while each symbol's panel is a
    contiguous daily grid. A missing day would stretch every lag and horizon across it with
    no test failing, and a duplicated day would give its second copy a 0-day lag, the
    same-day leak the lag exists to prevent. The steps are measured rather than assumed.
    The check runs per symbol, so the boundary between two symbols is never read as a step.
    """
    for symbol, dates in panel.groupby("symbol", sort=False)["metric_date"]:
        dates = dates.reset_index(drop=True)
        steps = dates.diff().dt.days
        broken = steps[(steps > 1) | (steps < 1)]
        if not broken.empty:
            row = broken.index[0]
            step = int(steps[row])
            if step < 1:
                cause = f"a duplicate date, {dates[row]:%Y-%m-%d} ({step}-day step)"
                effect = "give the second copy a 0-day lag, the same-day leak the lag prevents"
            else:
                cause = f"a {step}-day step into {dates[row]:%Y-%m-%d} ({step - 1} missing day(s))"
                effect = "silently stretch every lag and horizon across it"
            raise ValueError(f"{symbol} has {cause}; a row shift would {effect}")


def build_frame(panel: pd.DataFrame, estimates: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """(symbol, metric_date, y, <feature per method>) for one horizon, lagged and sorted.

    ``frame.attrs["dropped_features"]`` maps a dropped method id to the measurement that
    dropped it, so a missing column is explained rather than merely absent.
    """
    price = panel[["symbol", "metric_date", "price_usd"]].sort_values(["symbol", "metric_date"])
    _require_contiguous_days(price)
    signals = signal_matrix(estimates, price).sort_values(["symbol", "metric_date"])
    methods = [c for c in signals.columns if c not in ("symbol", "metric_date")]

    # The lag, applied within each symbol so one asset's last row cannot feed the next
    # asset's first. groupby(...).shift is the only correct way to say that.
    signals[methods] = signals.groupby("symbol", sort=False)[methods].shift(FEATURE_LAG_DAYS)

    out = price.merge(signals, on=["symbol", "metric_date"], how="left")
    out["y"] = out.groupby("symbol", sort=False)["price_usd"].transform(
        lambda s: forward_log_return(s, horizon)
    )
    # BACKWARD looking, so it is known at t: random_walk_drift estimates its drift from
    # this and scales it by h. Forward-looking would leak the thing being predicted.
    out["daily_log_return"] = out.groupby("symbol", sort=False)["price_usd"].transform(
        lambda s: np.log(s.where(s > 0.0) / s.where(s > 0.0).shift(1))
    )

    dropped: dict[str, str] = {}
    # Full-sample variance is harmless: the 1e-12 floor drops only signals constant on every row.
    for method_id in methods:
        variance = float(out[method_id].var(skipna=True)) if out[method_id].notna().any() else 0.0
        if not np.isfinite(variance) or variance < MIN_SIGNAL_VARIANCE:
            dropped[method_id] = f"variance {variance:.3e} below {MIN_SIGNAL_VARIANCE:.0e}"
            out = out.drop(columns=[method_id])

    out = out.drop(columns=["price_usd"]).reset_index(drop=True)
    out.attrs["dropped_features"] = dropped
    return out
