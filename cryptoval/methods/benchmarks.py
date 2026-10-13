# cryptoval/methods/benchmarks.py
"""Family F -- the falsified and contested benchmark shelf (spec 10.7).

These methods are published AS BROKEN. Stock-to-flow's residuals falsify it after 2021;
the power law regresses price on the calendar and carries nine documented objections.
Including them visibly is more honest than omitting them, and Plan 4's out-of-sample
tests need a deliberately weak benchmark to beat -- a model that beats nothing has not
been shown to be a model.

Every row from this module carries diagnostics["status"], so a number cannot travel out
of here without its own obituary attached.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cryptoval.methods.base import Estimate
from cryptoval.methods.cost import _finite
from cryptoval.methods.monetary import _refuse
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def fit_log_log(x: pd.Series, y: pd.Series, min_rows: int) -> dict[str, Any] | None:
    """OLS of log(y) on log(x), returning beta with its confidence interval.

    None rather than an exception below ``min_rows`` or when either series has too few
    strictly positive values: a caller's job is to turn that into a refusal naming the
    measurement, and an exception here would become a generic "method raised" row.

    Durbin-Watson travels with every fit because a log-log regression of two trending
    series reports a high R-squared almost regardless of whether one explains the other
    (spec 10.3's caveat). The statistic is how a reader sees that.
    """
    frame = pd.DataFrame(
        {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
    )
    frame = frame[(frame["x"] > 0) & (frame["y"] > 0)].dropna()
    if len(frame) < min_rows:
        return None

    design = sm.add_constant(np.log(frame["x"].to_numpy()))
    result = sm.OLS(np.log(frame["y"].to_numpy()), design).fit()
    low, high = result.conf_int()[1]
    return {
        "alpha": float(result.params[0]),
        "beta": float(result.params[1]),
        "beta_ci_low": float(low),
        "beta_ci_high": float(high),
        "r_squared": float(result.rsquared),
        "durbin_watson": float(sm.stats.stattools.durbin_watson(result.resid)),
        "n": int(len(frame)),
    }


@register
class StockToFlow:
    spec = _SPECS["stock_to_flow"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("flow_window_rows", 365))
        min_fit = int(params.get("min_fit_rows", 365))
        phash = param_set_hash(params)

        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        issuance_coins = pd.to_numeric(frame["issuance_usd"], errors="coerce") / price.where(
            price > 0
        )
        flow = issuance_coins.rolling(window, min_periods=window).sum()
        supply = pd.to_numeric(frame["supply"], errors="coerce")
        ratio = supply / flow.where(flow > 0)
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            # shift(1): this row's own ratio never enters the fit that values it.
            fit = fit_log_log(ratio.iloc[:position], market_cap.iloc[:position], min_fit)
            today = _finite(ratio.iloc[position])
            today_supply = _finite(row.supply)
            if fit is None:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        f"fewer than {min_fit} prior stock-to-flow observations",
                    )
                )
                continue
            if today is None or today <= 0 or today_supply is None or today_supply <= 0:
                out.append(
                    _refuse(self.spec, row, params, "no stock-to-flow ratio or supply on this day")
                )
                continue
            implied_cap = float(np.exp(fit["alpha"] + fit["beta"] * np.log(today)))
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=implied_cap / today_supply,
                    band_low=float(np.exp(fit["alpha"] + fit["beta_ci_low"] * np.log(today)))
                    / today_supply,
                    band_high=float(np.exp(fit["alpha"] + fit["beta_ci_high"] * np.log(today)))
                    / today_supply,
                    inputs_used={
                        "stock_to_flow": today,
                        "supply": today_supply,
                        "flow_window_rows": window,
                    },
                    diagnostics={**fit, "status": "falsified_post_2021"},
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class PowerLaw:
    spec = _SPECS["power_law"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        min_fit = int(params.get("min_fit_rows", 365))
        phash = param_set_hash(params)
        price = pd.to_numeric(frame["price_usd"], errors="coerce")

        genesis = frame["genesis_date"].dropna()
        if genesis.empty:
            # One refusal per row, not one per asset: the registry's grain is the row, and a
            # single refusal would silently drop the rest of the series.
            return [
                _refuse(self.spec, row, params, "no configured genesis date for this asset")
                for row in frame.itertuples(index=False)
            ]
        origin = pd.Timestamp(genesis.iloc[0]).tz_localize("UTC")
        days = (pd.to_datetime(frame["metric_date"], utc=True) - origin).dt.days.astype("float64")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            fit = fit_log_log(days.iloc[:position], price.iloc[:position], min_fit)
            today = _finite(days.iloc[position])
            if fit is None:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        f"fewer than {min_fit} prior observations since genesis",
                    )
                )
                continue
            if today is None or today <= 0:
                out.append(
                    _refuse(self.spec, row, params, "metric_date is not after the genesis date")
                )
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=float(np.exp(fit["alpha"] + fit["beta"] * np.log(today))),
                    band_low=float(np.exp(fit["alpha"] + fit["beta_ci_low"] * np.log(today))),
                    band_high=float(np.exp(fit["alpha"] + fit["beta_ci_high"] * np.log(today))),
                    inputs_used={"days_since_genesis": today, "genesis_date": origin.date()},
                    diagnostics={**fit, "status": "contested"},
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
