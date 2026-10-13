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

import pandas as pd

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def _finite(value: Any) -> float | None:
    """None for anything that cannot be a price: NaN, inf, or a non-number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
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
                        "price_to_cost": (price / cost) if price and cost else None,
                    },
                    applicable=True,
                    panel_vintage_ts=vintage,
                )
            )
        return out
