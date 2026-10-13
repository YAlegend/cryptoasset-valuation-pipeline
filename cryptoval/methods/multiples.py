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
