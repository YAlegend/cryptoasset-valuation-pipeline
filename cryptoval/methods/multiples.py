"""Family D -- relative multiples (spec 10.5).

Every multiple here reverts to a TRAILING statistic, never a full-sample one: a median
over the whole series tells a 2017 row what 2026 looked like, and Plan 4's walk-forward
tests would then be measuring leakage rather than skill.

NVT records which volume column produced it on every row. The two series differ about
fivefold (spec 10.1), so "NVT" without that label is two different numbers wearing one
name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.cost import _finite
from cryptoval.methods.monetary import _refuse
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def dilution_rate(supply: pd.Series, window_rows: int = 365) -> pd.Series:
    """Trailing supply growth, NaN until a full window exists.

    NaN rather than 0.0 for the first window: a zero dilution rate is a claim about
    monetary policy, and "we cannot see a year back yet" is not that claim.
    """
    values = pd.to_numeric(supply, errors="coerce")
    previous = values.shift(window_rows)
    return (values / previous.where(previous > 0)) - 1.0


def _nvt_estimates(
    frame: pd.DataFrame,
    spec: MethodSpec,
    params: dict,
    column: str,
    target: pd.Series,
    insufficient: str,
    extra_inputs: Callable[[float], dict[str, Any]],
    diagnostics_for: Callable[[float | None, float], dict[str, Any]],
) -> list[Estimate]:
    """Value the asset at a trailing NVT statistic: volume x target NVT / supply.

    ``target`` is the trailing statistic the multiple reverts to, already shifted so no
    row's own NVT enters its own target. Both methods in this family differ only in that
    statistic, so the loop lives once.
    """
    out: list[Estimate] = []
    phash = param_set_hash(params)
    market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
    volume = pd.to_numeric(frame[column], errors="coerce")
    nvt = market_cap / volume.where(volume > 0)

    for position, row in enumerate(frame.itertuples(index=False)):
        multiple = _finite(target.iloc[position])
        today_volume = _finite(volume.iloc[position])
        supply = _finite(row.supply)
        if multiple is None or multiple <= 0:
            out.append(_refuse(spec, row, params, insufficient))
            continue
        if today_volume is None or today_volume <= 0 or supply is None or supply <= 0:
            out.append(_refuse(spec, row, params, "missing volume or supply on this day"))
            continue
        today_nvt = _finite(nvt.iloc[position])
        out.append(
            Estimate(
                symbol=row.symbol,
                metric_date=row.metric_date,
                method_id=spec.method_id,
                param_set_hash=phash,
                fair_value=today_volume * multiple / supply,
                band_low=None,
                band_high=None,
                inputs_used={
                    # Load-bearing: NVT on the two volume series is two different
                    # numbers, so a row that did not name its source would be
                    # unreproducible.
                    "volume_column": column,
                    "volume": today_volume,
                    "supply": supply,
                    **extra_inputs(multiple),
                },
                diagnostics=diagnostics_for(today_nvt, multiple),
                applicable=True,
                panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
            )
        )
    return out


@register
class NvtReversion:
    spec = _SPECS["nvt_reversion"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        column = params.get("volume_column", "volume_total_usd")
        minimum = int(params.get("min_history_rows", 180))
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
        volume = pd.to_numeric(frame[column], errors="coerce")
        nvt = market_cap / volume.where(volume > 0)
        # shift(1) then expanding: this row's own NVT never enters its own target.
        target = nvt.shift(1).expanding(min_periods=minimum).median()

        return _nvt_estimates(
            frame,
            self.spec,
            params,
            column,
            target,
            f"fewer than {minimum} prior NVT observations on {column}",
            lambda multiple: {"trailing_median_nvt": multiple},
            lambda today, multiple: {
                "nvt": today,
                "deviation_from_trailing_median": (today / multiple) if today else None,
            },
        )


@register
class NvtGoldenRatio:
    """NVT against its own two-year moving average (Kalichkin 2018)."""

    spec = _SPECS["nvt_golden_ratio"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        column = params.get("volume_column", "volume_total_usd")
        ma_rows = int(params.get("ma_rows", 730))
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
        volume = pd.to_numeric(frame[column], errors="coerce")
        nvt = market_cap / volume.where(volume > 0)
        nvt_ma = nvt.shift(1).rolling(ma_rows, min_periods=ma_rows).mean()

        return _nvt_estimates(
            frame,
            self.spec,
            params,
            column,
            nvt_ma,
            f"fewer than {ma_rows} prior NVT observations for the moving average",
            lambda average: {"ma_rows": ma_rows, "nvt_ma": average},
            lambda today, average: {
                "nvt": today,
                "nvt_ma": average,
                "golden_ratio": (today / average) if today else None,
            },
        )


@register
class PriceToUtility:
    """The dissertation's P/U ratio (section 3.6).

    token_utility = (velocity x staking_ratio) / dilution_rate

    Two honest caveats ride along. The staking ratio is not served by any free source, so
    it is an unverified parameter and the method declines without it rather than guessing.
    And the construction inverts velocity, which holding_time_correction shows to be
    biased by exp(sigma^2) -- so uses_velocity_inversion is set on every row.
    """

    spec = _SPECS["price_to_utility"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        staking_ratio = params.get("staking_ratio")
        minimum = int(params.get("min_history_rows", 180))
        window = int(params.get("dilution_window_rows", 365))
        phash = param_set_hash(params)

        if staking_ratio is None:
            return [
                _refuse(
                    self.spec,
                    row,
                    params,
                    "staking_ratio is not supplied and no free source serves one, so "
                    "token_utility is not computable",
                )
                for row in frame.itertuples(index=False)
            ]

        velocity = pd.to_numeric(frame["velocity_total"], errors="coerce")
        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        dilution = dilution_rate(frame["supply"], window)
        utility = (velocity * float(staking_ratio)) / dilution.where(dilution > 0)
        target = (price / utility).shift(1).expanding(min_periods=minimum).median()

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            today_utility = _finite(utility.iloc[position])
            multiple = _finite(target.iloc[position])
            if today_utility is None or today_utility <= 0:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        f"token_utility is not computable: needs {window} rows of supply "
                        "history and a positive dilution rate",
                    )
                )
                continue
            if multiple is None:
                out.append(
                    _refuse(self.spec, row, params, f"fewer than {minimum} prior P/U observations")
                )
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=multiple * today_utility,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "staking_ratio": float(staking_ratio),
                        "dilution_window_rows": window,
                        "velocity_total": _finite(velocity.iloc[position]),
                        "trailing_median_price_to_utility": multiple,
                    },
                    diagnostics={
                        "token_utility": today_utility,
                        "dilution_rate": _finite(dilution.iloc[position]),
                        # holding_time_correction measures this bias at exp(sigma^2).
                        "uses_velocity_inversion": True,
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


def _annualised(series: pd.Series, window: int, annualise: bool = True) -> pd.Series:
    """Trailing-window mean, scaled to a year for a flow and left alone for a stock.

    A mean, not a sum: the window may contain fewer non-null days than rows, and summing
    nulls-as-zero would understate a protocol whose feed had an outage.

    ``annualise`` is False for TVL, which is a stock: multiplying a deposit balance by 365
    would produce a denominator with no meaning and a target multiple nobody could
    reconstruct. Absorbing the 365 into target_multiple instead would hide the same problem
    behind a number whose units cannot be read off the row.
    """
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=max(1, window // 2)).mean()
    return mean * 365.0 if annualise else mean


def _supply_age_days(row) -> int | None:
    """How far this row sits from the day the supply snapshot was taken.

    Supply metadata is point-in-time: one figure per symbol, joined onto every date. The
    multiples that use it are therefore valuing a 2020 row with 2026's supply, and the only
    honest thing a row can do about that is say how large the gap is.
    """
    as_of = getattr(row, "supply_as_of", None)
    if as_of is None or pd.isna(as_of):
        return None
    return int((pd.Timestamp(as_of) - pd.Timestamp(row.metric_date)).days)


def _multiple_estimates(
    frame: pd.DataFrame,
    spec: MethodSpec,
    params: dict,
    numerator_supply: Callable[[Any], tuple[float | None, dict[str, Any], str]],
) -> list[Estimate]:
    """One loop for every multiple: annualise the denominator, divide by the supply the
    numerator implies, scale by the target multiple.

    ``numerator_supply`` returns (supply, extra inputs, refusal reason) for a row, which is
    the only thing the five methods disagree about -- circulating supply for P/F, P/S and
    P/TVL, maximum supply for FDV, circulating plus a haircut on locked for dFDV.

    Every row here is a DIAGNOSTIC wearing a fair value's clothes: the target multiple is an
    assumption, the estimate scales linearly in it, and diagnostics.status says so on every
    row so the number cannot be lifted out of context without the caveat coming with it.
    """
    column = params["denominator"]
    window = int(params.get("window_rows", 365))
    target = float(params.get("target_multiple", 15.0))
    phash = param_set_hash(params)
    annual = _annualised(frame[column], window, bool(params.get("annualise", True)))
    price = pd.to_numeric(frame["price_usd"], errors="coerce")

    out: list[Estimate] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        value = _finite(annual.iloc[position])
        supply, extra, refusal = numerator_supply(row)
        if refusal:
            out.append(_refuse(spec, row, params, refusal))
            continue
        if value is None or value <= 0:
            # <= 0, not just None: net revenue CAN be negative, and dividing a negative
            # denominator into a price would publish a negative fair value that reads as a
            # cheap asset rather than as a protocol paying out more than it earns.
            out.append(
                _refuse(spec, row, params, f"no positive annualised {column} over {window} rows")
            )
            continue
        if supply is None or supply <= 0:
            out.append(_refuse(spec, row, params, "no usable supply on this day"))
            continue
        today_price = _finite(price.iloc[position])
        out.append(
            Estimate(
                symbol=row.symbol,
                metric_date=row.metric_date,
                method_id=spec.method_id,
                param_set_hash=phash,
                fair_value=value * target / supply,
                band_low=None,
                band_high=None,
                inputs_used={
                    "denominator": column,
                    "annual_value_usd": value,
                    "target_multiple": target,
                    "window_rows": window,
                    **extra,
                },
                diagnostics={
                    "realised_multiple": (today_price * supply / value) if today_price else None,
                    "supply_age_days": _supply_age_days(row),
                    "status": "diagnostic_target_multiple_is_an_assumption",
                },
                applicable=True,
                panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
            )
        )
    return out


def _circulating(row) -> tuple[float | None, dict, str]:
    supply = _finite(row.supply)
    return supply, {"supply": supply, "supply_basis": "circulating"}, ""


def _maximum(row) -> tuple[float | None, dict, str]:
    supply = _finite(getattr(row, "max_supply", None))
    if supply is None or supply <= 0:
        # No fallback to total_supply: that would publish a market-cap multiple as an FDV one.
        return None, {}, "no max_supply for this asset, so a fully diluted valuation does not exist"
    return (
        supply,
        {
            "supply": supply,
            "supply_basis": "max_supply",
            "supply_as_of": getattr(row, "supply_as_of", None),
        },
        "",
    )


def _haircut_locked(haircut: float) -> Callable[[Any], tuple[float | None, dict, str]]:
    def inner(row) -> tuple[float | None, dict, str]:
        circulating, total = _finite(row.supply), _finite(getattr(row, "total_supply", None))
        if circulating is None or total is None or total < circulating:
            # A total below circulating would make locked negative and haircut the supply
            # UPWARDS, inflating the fair value. The two figures come from DIFFERENT vendors
            # -- circulating from Coin Metrics, total from CoinGecko -- so this fires on 7 of
            # 24 assets (BNB by 41%, UNI 13%, ALGO 10%, POL, SNX, MKR, MANA). Subtracting one
            # vendor's supply from another's is the defect; the refusal names it rather than
            # implying the data is corrupt.
            return (
                None,
                {},
                "circulating supply (Coin Metrics) exceeds total supply (CoinGecko), so the "
                "locked portion would be negative: the two vendors disagree about this asset",
            )
        locked = total - circulating
        return (
            circulating + haircut * locked,
            {
                "supply_basis": "circulating_plus_haircut_locked",
                "circulating_supply": circulating,
                "total_supply": total,
                "locked_supply": locked,
                "locked_haircut": haircut,
                "effective_supply": circulating + haircut * locked,
                "supply_as_of": getattr(row, "supply_as_of", None),
            },
            "",
        )

    return inner


@register
class PriceToFees:
    spec = _SPECS["price_to_fees"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _multiple_estimates(frame, self.spec, params, _circulating)


@register
class PriceToSales:
    spec = _SPECS["price_to_sales"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _multiple_estimates(frame, self.spec, params, _circulating)


@register
class FdvToRevenue:
    spec = _SPECS["fdv_to_revenue"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _multiple_estimates(frame, self.spec, params, _maximum)


@register
class DfdvToRevenue:
    spec = _SPECS["dfdv_to_revenue"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _multiple_estimates(
            frame, self.spec, params, _haircut_locked(float(params.get("locked_haircut", 0.5)))
        )


@register
class PriceToTvl:
    spec = _SPECS["price_to_tvl"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _multiple_estimates(frame, self.spec, params, _circulating)
