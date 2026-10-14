"""Family E(a) -- the cost approach (spec 10.6).

Four of these are arithmetic over columns the mart already derives. All four inherit the
MVRV identity's fragility (spec 6.2): realized cap is market_cap / mvrv, so every method
here is flagged circular and is a COST BASIS rather than independent evidence of value.
The method cards say so; this module makes sure the flag is never absent.

Hayes is the exception: it is genuinely independent of market cap, and genuinely exposed
to a unit error in Coin Metrics' HashRate series, which is why Step 6 of its task
calibrates it against observed price before it ships.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.monetary import _refuse
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()

# Named on every row that uses hayes_cost_per_coin: the hardware and vendor unit these
# parameters were measured against. Anything else is out of calibration, not approximate.
PARAMETER_CALIBRATION = "sha256_asic_fleet_coinmetrics_hashrate_unit"


def _finite(value: Any) -> float | None:
    """None for anything that cannot be a price: NaN, inf, or a non-number."""
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def _per_unit_estimates(
    frame: pd.DataFrame,
    spec: MethodSpec,
    level_column: str,
    params: dict,
    diagnostics_for=None,
) -> list[Estimate]:
    """One estimate per row: ``level_column`` divided by supply.

    A row whose supply is missing, zero or negative yields an INAPPLICABLE estimate rather
    than an infinity -- the asset is fine, that day is not.
    """
    out: list[Estimate] = []
    phash = param_set_hash(params)
    for row in frame.itertuples(index=False):
        level = _finite(getattr(row, level_column))
        supply = _finite(row.supply)
        vintage = getattr(row, "panel_vintage_ts", None)
        if supply is None or supply <= 0 or level is None:
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=spec.method_id,
                    param_set_hash=phash,
                    fair_value=None,
                    band_low=None,
                    band_high=None,
                    inputs_used={},
                    diagnostics={},
                    applicable=False,
                    reason=(
                        f"supply is {supply!r} on this day"
                        if supply is None or supply <= 0
                        else f"{level_column} is missing on this day"
                    ),
                    panel_vintage_ts=vintage,
                )
            )
            continue
        diagnostics = diagnostics_for(row) if diagnostics_for else {}
        fair_value = level / supply
        lower_bound = bool(diagnostics.get("is_lower_bound"))
        out.append(
            Estimate(
                symbol=row.symbol,
                metric_date=row.metric_date,
                method_id=spec.method_id,
                param_set_hash=phash,
                fair_value=fair_value,
                # A lower bound is a one-sided band, not a point with no uncertainty.
                band_low=fair_value if lower_bound else None,
                band_high=None,
                inputs_used={level_column: level, "supply": supply},
                diagnostics=diagnostics,
                applicable=True,
                panel_vintage_ts=vintage,
            )
        )
    return out


@register
class RealizedPrice:
    spec = _SPECS["realized_price"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _per_unit_estimates(frame, self.spec, "realized_cap", params)


@register
class ThermocapFloor:
    spec = _SPECS["thermocap_floor"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def diagnostics(row) -> dict:
            gapped = bool(getattr(row, "thermocap_has_gap", False))
            return {
                "is_lower_bound": gapped,
                # Direction matters and is easy to state backwards: a gap means some
                # miner revenue was never added, so thermocap UNDERSTATES cumulative
                # cost, and MVTV and investor cap built on it are biased UPWARD.
                "bias_direction": (
                    "a gap makes thermocap understate cumulative cost, so this floor is "
                    "too low and MVTV/investor-cap built on it read too high"
                    if gapped
                    else "no gap"
                ),
            }

        return _per_unit_estimates(frame, self.spec, "thermocap_usd", params, diagnostics)


@register
class InvestorCapPrice:
    spec = _SPECS["investor_cap_price"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _per_unit_estimates(frame, self.spec, "investor_cap_usd", params)


def _reversion_estimates(
    frame: pd.DataFrame,
    spec: MethodSpec,
    level_column: str,
    multiple_column: str,
    params: dict,
) -> list[Estimate]:
    """Value the asset at its own historical multiple of a cost level.

    The multiple is a TRAILING median -- expanding, so no future information enters a
    past row (the no-lookahead rule that Plan 4's backtests depend on).
    """
    out: list[Estimate] = []
    phash = param_set_hash(params)
    minimum = int(params.get("min_history_rows", 180))
    multiples = pd.to_numeric(frame[multiple_column], errors="coerce")
    trailing_median = multiples.shift(1).expanding(min_periods=minimum).median()

    for position, row in enumerate(frame.itertuples(index=False)):
        level = _finite(getattr(row, level_column))
        supply = _finite(row.supply)
        target = _finite(trailing_median.iloc[position])
        vintage = getattr(row, "panel_vintage_ts", None)
        if None in (level, supply, target) or supply <= 0 or target <= 0:
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=spec.method_id,
                    param_set_hash=phash,
                    fair_value=None,
                    band_low=None,
                    band_high=None,
                    inputs_used={},
                    diagnostics={},
                    applicable=False,
                    reason=(
                        f"fewer than {minimum} prior observations of {multiple_column}"
                        if target is None
                        else "missing level or supply on this day"
                    ),
                    panel_vintage_ts=vintage,
                )
            )
            continue
        observed = _finite(getattr(row, multiple_column))
        out.append(
            Estimate(
                symbol=row.symbol,
                metric_date=row.metric_date,
                method_id=spec.method_id,
                param_set_hash=phash,
                fair_value=level * target / supply,
                band_low=None,
                band_high=None,
                inputs_used={
                    level_column: level,
                    "supply": supply,
                    f"trailing_median_{multiple_column}": target,
                },
                diagnostics={
                    f"observed_{multiple_column}": observed,
                    "deviation_from_trailing_median": (
                        observed / target if observed and target else None
                    ),
                },
                applicable=True,
                panel_vintage_ts=vintage,
            )
        )
    return out


@register
class MvrvReversion:
    spec = _SPECS["mvrv_reversion"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _reversion_estimates(frame, self.spec, "realized_cap", "mvrv", params)


@register
class MvtvReversion:
    spec = _SPECS["mvtv_reversion"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _reversion_estimates(frame, self.spec, "thermocap_usd", "mvtv", params)


def hayes_cost_per_coin(
    hash_rate: float,
    hashes_per_unit: float,
    joules_per_hash: float,
    electricity_usd_per_kwh: float,
    coins_per_day: float,
) -> float | None:
    """Marginal electricity cost of producing one coin in a day (Hayes 2017).

    energy = hash_rate x hashes_per_unit x joules_per_hash x 86400 seconds
    cost   = energy / 3.6e6 J-per-kWh x price-per-kWh
    result = cost / coins mined that day

    ``hashes_per_unit`` converts Coin Metrics' HashRate unit into hashes per second. It is
    a PARAMETER rather than a constant because the unit differs by asset and getting it
    wrong scales the answer by orders of magnitude without changing its shape -- which is
    exactly the error a plausible-looking chart hides. Step 6 of this task calibrates it.
    """
    coins = _finite(coins_per_day)
    rate = _finite(hash_rate)
    if coins is None or coins <= 0 or rate is None or rate <= 0:
        return None
    joules = rate * hashes_per_unit * joules_per_hash * 86_400
    kwh = joules / 3.6e6
    return kwh * electricity_usd_per_kwh / coins


@register
class HayesCostOfProduction:
    spec = _SPECS["hayes_cost_of_production"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        out: list[Estimate] = []
        phash = param_set_hash(params)
        joules = float(params["joules_per_hash"])
        price_kwh = float(params["electricity_usd_per_kwh"])
        hashes_per_unit = float(params["hashes_per_unit"])

        for row in frame.itertuples(index=False):
            vintage = getattr(row, "panel_vintage_ts", None)
            issuance = _finite(row.issuance_usd)
            price = _finite(row.price_usd)
            coins = issuance / price if issuance is not None and price not in (None, 0) else None
            cost = (
                hayes_cost_per_coin(row.hash_rate, hashes_per_unit, joules, price_kwh, coins)
                if coins is not None
                else None
            )
            if cost is None:
                out.append(
                    Estimate(
                        symbol=row.symbol,
                        metric_date=row.metric_date,
                        method_id=self.spec.method_id,
                        param_set_hash=phash,
                        fair_value=None,
                        band_low=None,
                        band_high=None,
                        inputs_used={},
                        diagnostics={},
                        applicable=False,
                        reason="no hash rate or no coins issued on this day",
                        panel_vintage_ts=vintage,
                    )
                )
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=cost,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "hash_rate": _finite(row.hash_rate),
                        "coins_per_day": coins,
                        "joules_per_hash": joules,
                        "electricity_usd_per_kwh": price_kwh,
                        "hashes_per_unit": hashes_per_unit,
                    },
                    diagnostics={
                        # Hayes is a FLOOR argument, not a fair value: miners exit below
                        # cost, which constrains price from beneath.
                        "interpretation": "marginal production cost, a floor rather than a target",
                        # Ruling 16: these parameters describe SHA-256 hardware and the unit
                        # Coin Metrics reports for it. On a Scrypt asset (LTC, DOGE) both the
                        # unit and the energy per hash differ, and the output is wrong by
                        # orders of magnitude rather than slightly off.
                        "parameters_calibrated_for": PARAMETER_CALIBRATION,
                        "price_to_cost": (price / cost) if price and cost else None,
                    },
                    applicable=True,
                    panel_vintage_ts=vintage,
                )
            )
        return out


def _window_stats(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Observations to date and observations inside the trailing window, per row.

    isfinite, not notna: an inf passes a null check and then poisons the mean, which
    produced the self-contradictory "only 200 observations in the last 200 rows".
    """
    seen = pd.Series(
        np.isfinite(pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")),
        index=series.index,
    )
    return seen.cumsum(), seen.rolling(window, min_periods=1).sum()


def _short_window_reason(what: str, to_date: float, in_window: float, window: int) -> str:
    """A refusal that names its measurement, and which of two causes produced it."""
    if to_date < window:
        return f"only {int(to_date)} prior {what} observations to date, need {window}"
    return (
        f"only {int(in_window)} {what} observations in the last {window} rows, need {window} "
        "(gaps in the window)"
    )


@register
class MayerMultiple:
    spec = _SPECS["mayer_multiple"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 200))
        target = float(params.get("target_multiple", 1.0))
        phash = param_set_hash(params)
        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        # shift(1): today's own price must not enter the average today's price is judged
        # against, or the signal leaks one day of the thing it is meant to predict.
        moving = price.shift(1).rolling(window, min_periods=window).mean()
        # The published multiple keeps the spec's inclusive definition; only the fair
        # value is prior-only, and both bases are named on the row.
        inclusive = price.rolling(window, min_periods=window).mean()
        to_date, in_window = _window_stats(price.shift(1), window)

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            average = _finite(moving.iloc[position])
            inclusive_mean = _finite(inclusive.iloc[position])
            today = _finite(row.price_usd)
            if average is None:
                reason = _short_window_reason(
                    "price",
                    to_date.iloc[position],
                    in_window.iloc[position],
                    window,
                )
                out.append(_refuse(self.spec, row, params, reason))
                continue
            if average <= 0:
                out.append(_refuse(self.spec, row, params, f"trailing mean price is {average}"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=average * target,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "moving_average_price": average,
                        "window_rows": window,
                        "target_multiple": target,
                    },
                    diagnostics={
                        "mayer_multiple": (today / inclusive_mean)
                        if today and inclusive_mean
                        else None,
                        "mayer_multiple_basis": "inclusive_window, the published definition",
                        "baseline_basis": "prior_only",
                        "status": "diagnostic_not_fair_value",
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class PuellMultiple:
    spec = _SPECS["puell_multiple"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        target = float(params.get("target_multiple", 1.0))
        phash = param_set_hash(params)
        issuance = pd.to_numeric(frame["issuance_usd"], errors="coerce")
        # shift(1): see MayerMultiple. Today's issuance is the numerator of the multiple,
        # so it cannot also sit inside the average that sets the denominator.
        average = issuance.shift(1).rolling(window, min_periods=window).mean()
        inclusive = issuance.rolling(window, min_periods=window).mean()
        to_date, in_window = _window_stats(issuance.shift(1), window)

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            baseline = _finite(average.iloc[position])
            inclusive_mean = _finite(inclusive.iloc[position])
            today_issuance = _finite(row.issuance_usd)
            price, supply = _finite(row.price_usd), _finite(row.supply)
            if baseline is None:
                reason = _short_window_reason(
                    "issuance",
                    to_date.iloc[position],
                    in_window.iloc[position],
                    window,
                )
                out.append(_refuse(self.spec, row, params, reason))
                continue
            if baseline <= 0:
                out.append(_refuse(self.spec, row, params, f"trailing mean issuance is {baseline}"))
                continue
            if today_issuance is None or today_issuance <= 0:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        f"no issuance on this day (issuance_usd={row.issuance_usd})",
                    )
                )
                continue
            if price is None or price <= 0:
                out.append(
                    _refuse(
                        self.spec, row, params, f"no price on this day (price_usd={row.price_usd})"
                    )
                )
                continue
            # Fair value = the price at which today's issuance revenue would equal its
            # trailing average, holding coins issued fixed: price x (baseline / today).
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=price * (baseline * target / today_issuance),
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "issuance_usd": today_issuance,
                        "trailing_mean_issuance_usd": baseline,
                        "window_rows": window,
                        "supply": supply,
                    },
                    diagnostics={
                        "puell_multiple": (today_issuance / inclusive_mean)
                        if inclusive_mean
                        else None,
                        "puell_multiple_basis": "inclusive_window, the published definition",
                        "baseline_basis": "prior_only",
                        "status": "diagnostic_not_fair_value",
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class MvrvZScore:
    spec = _SPECS["mvrv_z_score"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        target_z = float(params.get("target_z", 0.0))
        phash = param_set_hash(params)
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
        realized = pd.to_numeric(frame["realized_cap"], errors="coerce")
        # shift(1) for the same reason as Mayer; ddof=1 (pandas' default) over a rolling
        # window, which is NOT the canonical expanding population sigma -- named on the row.
        deviation = market_cap.shift(1).rolling(window, min_periods=window).std()
        to_date, in_window = _window_stats(market_cap.shift(1), window)

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            sigma = _finite(deviation.iloc[position])
            cap, real, supply = (
                _finite(market_cap.iloc[position]),
                _finite(realized.iloc[position]),
                _finite(row.supply),
            )
            if sigma is None:
                reason = _short_window_reason(
                    "market-cap",
                    to_date.iloc[position],
                    in_window.iloc[position],
                    window,
                )
                out.append(_refuse(self.spec, row, params, reason))
                continue
            if sigma <= 0:
                reason = f"market cap has zero standard deviation over the last {window} rows"
                out.append(_refuse(self.spec, row, params, reason))
                continue
            if cap is None or real is None or supply is None or supply <= 0:
                missing = [
                    name
                    for name, value in (("market_cap_usd", cap), ("realized_cap", real))
                    if value is None
                ]
                if supply is None or supply <= 0:
                    missing.append(f"supply (={row.supply})")
                out.append(
                    _refuse(self.spec, row, params, f"missing on this day: {', '.join(missing)}")
                )
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=(real + target_z * sigma) / supply,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "market_cap_usd": cap,
                        "realized_cap": real,
                        "stdev_market_cap": sigma,
                        "window_rows": window,
                        "supply": supply,
                    },
                    diagnostics={
                        "z_score": (cap - real) / sigma,
                        "sigma_basis": "prior_only_rolling_sample_stdev_ddof1",
                        # At target_z = 0 the fair value IS realized_price's, to the last
                        # digit: only z_score is new information. A composite that averages
                        # both double-counts realized cap.
                        "derived_from": "realized_price" if target_z == 0.0 else None,
                        "status": "circular_realized_cap",
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class EnergyValue:
    spec = _SPECS["energy_value"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        phash = param_set_hash(params)
        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        coins_per_day = pd.to_numeric(frame["issuance_usd"], errors="coerce") / price.where(
            price > 0
        )
        hash_rate = pd.to_numeric(frame["hash_rate"], errors="coerce")

        # Capriole's energy value is 2 x annual energy cost / annual coins issued, which on a
        # daily basis is 2 x the Hayes cost per coin. Calling hayes_cost_per_coin rather than
        # restating the arithmetic is the point: the two methods are one piece of evidence.
        daily_cost = pd.Series(
            [
                hayes_cost_per_coin(
                    hash_rate.iloc[i],
                    float(params["hashes_per_unit"]),
                    float(params["joules_per_hash"]),
                    float(params["electricity_usd_per_kwh"]),
                    coins_per_day.iloc[i],
                )
                for i in range(len(frame))
            ],
            index=frame.index,
            dtype="float64",
        )
        smoothed = daily_cost.shift(1).rolling(window, min_periods=window).mean()
        to_date, in_window = _window_stats(daily_cost.shift(1), window)

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            mean_cost = _finite(smoothed.iloc[position])
            if mean_cost is None:
                reason = _short_window_reason(
                    "computable production-cost",
                    to_date.iloc[position],
                    in_window.iloc[position],
                    window,
                )
                out.append(_refuse(self.spec, row, params, reason))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=2.0 * mean_cost,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "trailing_mean_production_cost": mean_cost,
                        "window_rows": window,
                        "joules_per_hash": params["joules_per_hash"],
                        "electricity_usd_per_kwh": params["electricity_usd_per_kwh"],
                        "hashes_per_unit": float(params["hashes_per_unit"]),
                    },
                    diagnostics={
                        "derived_from": "hayes_cost_of_production",
                        "multiplier": 2.0,
                        "parameters_calibrated_for": PARAMETER_CALIBRATION,
                        "status": "not_independent_of_hayes",
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
