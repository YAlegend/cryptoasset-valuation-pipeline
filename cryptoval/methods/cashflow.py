"""Family C -- discounted cash flow, and the counter-model that disputes it (spec 10.4).

Two things here are deliberate and should not be "tidied".

THE OUTPUT IS A GRID. Spec 10.4 asks for a sensitivity grid over growth, discount rate and
terminal multiple rather than a single number, and the registry already hashes parameter
sets so cells coexist as rows. A grid cell is therefore a param set, and a single cell
quoted alone is not this family's answer.

THE COUNTER-MODEL IS A SIBLING METHOD. The standing critique -- that fee burn and staking
yield move the supply path rather than network value -- is implemented, not argued, so the
disagreement between it and the DCFs is measured. Where they diverge is the finding.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pandas as pd

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.cost import _finite
from cryptoval.methods.monetary import _refuse
from cryptoval.methods.multiples import _annualised, dilution_rate
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def grid(params: dict) -> list[dict]:
    """Expand a ``sensitivity`` block into one param set per combination.

    The cell does NOT carry the sensitivity block: if it did, every cell would hash to the
    same param_set_hash and the grid would collapse to one row in the fact table.
    """
    sensitivity = {k: v for k, v in (params.get("sensitivity") or {}).items() if v}
    base = {k: v for k, v in params.items() if k != "sensitivity"}
    if not sensitivity:
        return [base]
    names = sorted(sensitivity)
    return [
        {**base, **dict(zip(names, combination, strict=True))}
        for combination in itertools.product(*(sensitivity[n] for n in names))
    ]


def three_stage_pv(
    cash_flow: float,
    growth: tuple[float, float, float],
    discount: float,
    years: tuple[int, int, int],
    terminal_multiple: float | None,
) -> tuple[float, float]:
    """Present value of three growth stages, and the fraction of it that is terminal.

    ``terminal_multiple`` None uses a Gordon terminal value at the third stage's growth,
    which REQUIRES discount > that growth. Raising rather than returning a negative or an
    infinity is deliberate: a negative present value from an inverted Gordon term is
    arithmetically valid and economically meaningless, and it would reach the fact table
    as a fair value.
    """
    if discount <= 0:
        raise ValueError(f"discount rate must be positive, got {discount}")
    if terminal_multiple is None and discount <= growth[2]:
        raise ValueError(
            f"terminal growth {growth[2]} is at or above the discount rate {discount}; "
            f"the Gordon terminal value is undefined there"
        )

    explicit = 0.0
    flow = float(cash_flow)
    period = 0
    for stage_growth, stage_years in zip(growth, years, strict=True):
        for _ in range(int(stage_years)):
            period += 1
            flow *= 1.0 + stage_growth
            explicit += flow / (1.0 + discount) ** period

    if terminal_multiple is None:
        terminal_flow = flow * (1.0 + growth[2]) / (discount - growth[2])
    else:
        terminal_flow = flow * float(terminal_multiple)
    terminal = terminal_flow / (1.0 + discount) ** period
    total = explicit + terminal
    return total, (terminal / total if total > 0 else 1.0)


def _dcf_estimates(
    frame: pd.DataFrame,
    spec: MethodSpec,
    params: dict,
    cash_flow: Callable[[pd.DataFrame], pd.Series],
    label: str,
    gate=None,
) -> list[Estimate]:
    """One row per (panel row, grid cell). ``gate`` returns a refusal reason or ""."""
    out: list[Estimate] = []
    cells = grid(params)
    window = int(params.get("window_rows", 365))
    cash = _annualised(cash_flow(frame), window)
    supply = pd.to_numeric(frame["supply"], errors="coerce")

    for position, row in enumerate(frame.itertuples(index=False)):
        refusal = gate(row) if gate else ""
        annual_flow = _finite(cash.iloc[position])
        today_supply = _finite(supply.iloc[position])
        for cell in cells:
            if refusal:
                out.append(_refuse(spec, row, cell, refusal))
                continue
            if annual_flow is None or annual_flow <= 0:
                # <= 0 is the common case for ETH, not an edge: annualised issuance exceeds
                # the burn on 83% of post-EIP-1559 days. A negative cash flow discounts to a
                # negative present value, which is arithmetically fine and is not a price.
                out.append(
                    _refuse(spec, row, cell, f"no positive annualised {label} over {window} rows")
                )
                continue
            if today_supply is None or today_supply <= 0:
                out.append(_refuse(spec, row, cell, "no supply on this day"))
                continue
            try:
                present_value, terminal_share = three_stage_pv(
                    annual_flow,
                    tuple(cell["growth"]),
                    float(cell["discount_rate"]),
                    tuple(cell["stage_years"]),
                    cell.get("terminal_multiple"),
                )
            except ValueError as exc:
                out.append(_refuse(spec, row, cell, str(exc)))
                continue
            ceiling = float(cell.get("max_terminal_share", 0.8))
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=spec.method_id,
                    param_set_hash=param_set_hash(cell),
                    fair_value=present_value / today_supply,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "cash_flow_basis": label,
                        "annual_cash_flow_usd": annual_flow,
                        "supply": today_supply,
                        "discount_rate": cell["discount_rate"],
                        "terminal_multiple": cell.get("terminal_multiple"),
                        "growth": list(cell["growth"]),
                        "stage_years": list(cell["stage_years"]),
                    },
                    diagnostics={
                        "terminal_share": terminal_share,
                        "terminal_dominated": terminal_share > ceiling,
                        "present_value_usd": present_value,
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
    return out


@register
class EthThreeStageDcf:
    spec = _SPECS["eth_three_stage_dcf"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def cash_flow(f: pd.DataFrame) -> pd.Series:
            # BURN minus issuance, per Step 1's measurement -- not total fees minus issuance.
            # DefiLlama's Ethereum methodology defines Revenue as the ETH burned, and the
            # burn accrues to every holder by reducing supply. Priority fees and MEV accrue
            # to PROPOSERS; counting them here would credit holders with money that goes to
            # block builders.
            burn = pd.to_numeric(f["net_revenue_usd"], errors="coerce")
            issuance = pd.to_numeric(f["issuance_usd"], errors="coerce")
            return burn - issuance

        return _dcf_estimates(frame, self.spec, params, cash_flow, "burn_net_of_issuance")


@register
class DefiNetRevenueDcf:
    spec = _SPECS["defi_net_revenue_dcf"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def gate(row) -> str:
            if getattr(row, "revenue_regime", None) == "fees_fully_distributed":
                return (
                    "this protocol does not retain revenue (revenue_regime="
                    "fees_fully_distributed), so there is no cash flow to discount"
                )
            return ""

        return _dcf_estimates(
            frame,
            self.spec,
            params,
            lambda f: pd.to_numeric(f["net_revenue_usd"], errors="coerce"),
            "net_protocol_revenue",
            gate,
        )


@register
class BondTargetYield:
    spec = _SPECS["bond_target_yield"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        target = float(params["risk_free_rate"]) + float(params["staking_risk_premium"])
        phash = param_set_hash(params)
        annual = _annualised(frame[params.get("denominator", "total_fees_usd")], window)
        supply = pd.to_numeric(frame["supply"], errors="coerce")
        price = pd.to_numeric(frame["price_usd"], errors="coerce")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            flow, today_supply = _finite(annual.iloc[position]), _finite(supply.iloc[position])
            if target <= 0:
                out.append(
                    _refuse(self.spec, row, params, f"target yield {target} is not positive")
                )
                continue
            if flow is None or flow <= 0 or today_supply is None or today_supply <= 0:
                out.append(
                    _refuse(self.spec, row, params, "no annualised fee flow or supply on this day")
                )
                continue
            today_price = _finite(price.iloc[position])
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=flow / target / today_supply,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "annual_fees_usd": flow,
                        "target_yield": target,
                        "risk_free_rate": params["risk_free_rate"],
                        "staking_risk_premium": params["staking_risk_premium"],
                        "denominator": params.get("denominator", "total_fees_usd"),
                        "supply": today_supply,
                    },
                    diagnostics={
                        "implied_current_yield": (flow / (today_supply * today_price))
                        if today_price
                        else None,
                        "status": "flat_perpetuity",
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class EmissionsAdjustedRealYield:
    spec = _SPECS["emissions_adjusted_real_yield"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        incentive = float(params.get("incentive_share_usd", 0.0))
        phash = param_set_hash(params)
        fees = _annualised(frame["total_fees_usd"], window)
        issuance = _annualised(frame["issuance_usd"], window)
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            annual_fees = _finite(fees.iloc[position])
            annual_issuance = _finite(issuance.iloc[position])
            cap = _finite(market_cap.iloc[position])
            if annual_fees is None or annual_issuance is None or cap is None or cap <= 0:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        "no annualised fees, issuance or market cap on this day",
                    )
                )
                continue
            fee_yield = annual_fees / cap
            dilution = annual_issuance / cap
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=None,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "annual_fees_usd": annual_fees,
                        "annual_issuance_usd": annual_issuance,
                        "market_cap_usd": cap,
                        "incentive_share_usd": incentive,
                    },
                    diagnostics={
                        "fee_yield": fee_yield,
                        "dilution": dilution,
                        "real_yield": fee_yield + (incentive / cap) - dilution,
                        # Load-bearing: spec 10.4's incentive term has no free source, so the
                        # published figure is a lower bound on that side and says so per row.
                        "incentive_share_available": incentive > 0,
                        "status": "yield_not_a_price",
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class SupplyEffectCounterModel:
    spec = _SPECS["supply_effect_counter_model"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        horizon = int(params.get("horizon_years", 10))
        phash = param_set_hash(params)

        # The OBSERVED supply path, not a reconstruction of it. dilution_rate is the
        # trailing growth in the supply series itself, which already nets issuance against
        # burn; rebuilding it from issuance_usd minus a share of fees would need a
        # per-asset burn share that no free source serves, and assuming all fees are burned
        # is false for ETH (20-50% post-EIP-4844) and for every chain that burns nothing.
        growth = dilution_rate(frame["supply"], window)
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
        supply = pd.to_numeric(frame["supply"], errors="coerce")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            cap, today_supply = _finite(market_cap.iloc[position]), _finite(supply.iloc[position])
            rate = _finite(growth.iloc[position])
            if cap is None or today_supply is None or today_supply <= 0 or rate is None:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        f"no market cap, supply or {window}-day supply growth rate on this day",
                    )
                )
                continue
            projected = today_supply * (1.0 + rate) ** horizon
            if projected <= 0:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        f"projected supply over {horizon} years is not positive",
                    )
                )
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    # Today's network value over tomorrow's supply: the whole claim.
                    fair_value=cap / projected,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "market_cap_usd": cap,
                        "supply": today_supply,
                        "annual_supply_growth": rate,
                        "projected_supply": projected,
                        "horizon_years": horizon,
                    },
                    diagnostics={
                        "holds_network_value_constant": True,
                        "supply_change_fraction": (projected / today_supply) - 1.0,
                        "status": "null_hypothesis_against_the_dcfs",
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
