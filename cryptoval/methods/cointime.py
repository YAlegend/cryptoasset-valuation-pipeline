"""Family E(b) -- Cointime and coin-age signals (spec 10.6(b), Plan 3b).

Each is one ratio per day over columns derived from the BigQuery coin-age extraction. BTC is
the only asset with those columns, so every other asset is refused by the registry's density
check, which names the empty column. The four signals return no fair value; their value
lives in diagnostics under the method id. Active price and true market mean are prices.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from cryptoval.methods.base import Estimate, MethodSpec, _finite
from cryptoval.methods.monetary import _refuse
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def _ratio(num_name: str, num, den_name: str, den) -> tuple[float | None, dict, str]:
    num, den = _finite(num), _finite(den)
    if num is None or den is None or den <= 0:
        reason = f"{den_name}={den!r}, {num_name}={num!r}: the ratio is undefined on this day"
        return None, {}, reason
    return num / den, {num_name: num, den_name: den}, ""


def _estimates(
    frame: pd.DataFrame, spec: MethodSpec, params: dict, compute: Callable
) -> list[Estimate]:
    out, phash = [], param_set_hash(params)
    for row in frame.itertuples(index=False):
        value, inputs, reason = compute(row)
        if value is not None and spec.returns_fair_value and value <= 0:
            reason = f"{spec.method_id} is {value!r}; a price must be positive ({inputs})"
            value = None
        if value is None:
            out.append(_refuse(spec, row, params, reason))
            continue
        out.append(
            Estimate(
                symbol=row.symbol,
                metric_date=row.metric_date,
                method_id=spec.method_id,
                param_set_hash=phash,
                fair_value=value if spec.returns_fair_value else None,
                band_low=None,
                band_high=None,
                inputs_used=inputs,
                diagnostics={} if spec.returns_fair_value else {spec.method_id: value},
                applicable=True,
                panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
            )
        )
    return out


@register
class Slrv:
    spec = _SPECS["slrv"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def compute(r):
            return _ratio("hodl_rc_0d", r.hodl_rc_0d, "hodl_rc_6m_1y", r.hodl_rc_6m_1y)

        return _estimates(frame, self.spec, params, compute)


@register
class Rpv:
    spec = _SPECS["rpv"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def compute(r):
            return _ratio(
                "realized_profit_usd", r.realized_profit_usd, "market_cap_usd", r.market_cap_usd
            )

        return _estimates(frame, self.spec, params, compute)


@register
class Sthpl:
    spec = _SPECS["sthpl"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def compute(r):
            sth_price, _, reason = _ratio(
                "sth_realized_cap", r.sth_realized_cap, "sth_supply", r.sth_supply
            )
            if sth_price is None:
                return None, {}, reason
            return _ratio("price_usd", r.price_usd, "sth_realized_price", sth_price)

        return _estimates(frame, self.spec, params, compute)


@register
class Aviv:
    spec = _SPECS["aviv"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def compute(r):
            cap, live = _finite(r.market_cap_usd), _finite(r.liveliness)
            active = cap * live if cap is not None and live is not None else None
            return _ratio("active_cap_usd", active, "investor_cap_usd", r.investor_cap_usd)

        return _estimates(frame, self.spec, params, compute)


@register
class ActivePrice:
    spec = _SPECS["active_price"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def compute(r):
            return _ratio("realized_cap", r.realized_cap, "active_supply", r.active_supply)

        return _estimates(frame, self.spec, params, compute)


@register
class TrueMarketMean:
    spec = _SPECS["true_market_mean"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def compute(r):
            return _ratio("investor_cap_usd", r.investor_cap_usd, "active_supply", r.active_supply)

        return _estimates(frame, self.spec, params, compute)
