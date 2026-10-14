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


# Below this, residuals are strongly positively autocorrelated and the fit's R-squared
# tells you the series both trend, not that one explains the other. Lives here because
# fit_log_log is the only place this project computes a Durbin-Watson statistic.
DW_AUTOCORRELATION_FLOOR = 1.5


def fit_log_log(
    x: pd.Series, y: pd.Series, min_rows: int, predict_at: float | None = None
) -> dict[str, Any] | None:
    """OLS of log(y) on log(x), returning beta with its confidence interval.

    None rather than an exception below ``min_rows``, when either series has too few
    strictly positive finite values, or when x has no variation (beta is then not
    identified): a caller's job is to turn that into a refusal naming the measurement,
    and an exception here would become a generic "method raised" row.

    Durbin-Watson travels with every fit because a log-log regression of two trending
    series reports a high R-squared almost regardless of whether one explains the other
    (spec 10.3's caveat). The statistic is how a reader sees that.

    ``predict_at`` asks for the fitted value at one x, with the mean-prediction
    confidence interval that uses the alpha-beta covariance. Plugging beta's own interval
    into the line with alpha held fixed is NOT an interval for the fitted value: alpha and
    beta are negatively correlated, and the two bounds swap places whenever log(x) < 0.
    """
    frame = pd.DataFrame(
        {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
    ).replace([np.inf, -np.inf], np.nan)
    frame = frame[(frame["x"] > 0) & (frame["y"] > 0)].dropna()
    # nunique: a constant x makes conf_int()[1] an IndexError rather than a fit.
    if len(frame) < min_rows or frame["x"].nunique() < 2:
        return None

    design = sm.add_constant(np.log(frame["x"].to_numpy()))
    result = sm.OLS(np.log(frame["y"].to_numpy()), design).fit()
    low, high = result.conf_int()[1]
    dw = float(sm.stats.stattools.durbin_watson(result.resid))
    fit = {
        "alpha": float(result.params[0]),
        "beta": float(result.params[1]),
        "beta_ci_low": float(low),
        "beta_ci_high": float(high),
        "r_squared": float(result.rsquared),
        "durbin_watson": dw,
        # The flag that stops a 0.99 R-squared from being mistaken for evidence.
        "residuals_autocorrelated": bool(dw < DW_AUTOCORRELATION_FLOOR),
        "n": int(len(frame)),
    }
    if predict_at is not None and np.isfinite(predict_at) and predict_at > 0:
        prediction = result.get_prediction([1.0, float(np.log(predict_at))])
        ci_low, ci_high = prediction.conf_int()[0]
        fit["fit_y"] = float(np.exp(prediction.predicted_mean[0]))
        fit["fit_y_ci_low"] = float(np.exp(ci_low))
        fit["fit_y_ci_high"] = float(np.exp(ci_high))
        # Named on the row: OLS standard errors on autocorrelated residuals are
        # overconfident, so this band is narrower than the fit really supports.
        fit["ci_basis"] = "naive_ols_mean_prediction"
    return fit


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
        # Counted, not assumed: a refusal has to name which of the two things it lacks.
        prior_ratios = ratio.gt(0).cumsum().shift(1).fillna(0)
        prior_issuance_days = issuance_coins.gt(0).cumsum().shift(1).fillna(0)

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            today = _finite(ratio.iloc[position])
            today_supply = _finite(row.supply)
            # iloc[:position]: this row's own ratio never enters the fit that values it.
            fit = fit_log_log(
                ratio.iloc[:position],
                market_cap.iloc[:position],
                min_fit,
                predict_at=today if today is not None and today > 0 else None,
            )
            if fit is None:
                if prior_issuance_days.iloc[position] == 0:
                    reason = (
                        "no day before this one has positive issuance, so stock-to-flow "
                        "is undefined rather than short of history"
                    )
                else:
                    reason = (
                        f"only {int(prior_ratios.iloc[position])} prior stock-to-flow "
                        f"observations, need {min_fit}"
                    )
                out.append(_refuse(self.spec, row, params, reason))
                continue
            if today is None or today <= 0 or today_supply is None or today_supply <= 0:
                out.append(
                    _refuse(self.spec, row, params, "no stock-to-flow ratio or supply on this day")
                )
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=fit["fit_y"] / today_supply,
                    band_low=fit["fit_y_ci_low"] / today_supply,
                    band_high=fit["fit_y_ci_high"] / today_supply,
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
        prior_rows = (days.gt(0) & price.gt(0)).cumsum().shift(1).fillna(0)

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            today = _finite(days.iloc[position])
            fit = fit_log_log(
                days.iloc[:position],
                price.iloc[:position],
                min_fit,
                predict_at=today if today is not None and today > 0 else None,
            )
            if fit is None:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        f"only {int(prior_rows.iloc[position])} prior priced days since "
                        f"genesis, need {min_fit}",
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
                    fair_value=fit["fit_y"],
                    band_low=fit["fit_y_ci_low"],
                    band_high=fit["fit_y_ci_high"],
                    inputs_used={"days_since_genesis": today, "genesis_date": origin.date()},
                    diagnostics={**fit, "status": "contested"},
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
