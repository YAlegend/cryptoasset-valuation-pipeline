import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.cashflow import (
    BondTargetYield,
    DefiNetRevenueDcf,
    EmissionsAdjustedRealYield,
    EthThreeStageDcf,
    SupplyEffectCounterModel,
    grid,
    three_stage_pv,
)


def _cash_frame(days: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2025-06-01", periods=days, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": "ETH",
            "metric_date": [d.date() for d in dates],
            "price_usd": np.full(days, 2500.0),
            "supply": np.linspace(1.2e8, 1.21e8, days),
            "market_cap_usd": np.full(days, 3.0e11),
            "total_fees_usd": np.full(days, 3.0e6),
            # Burn above issuance, so the ETH DCF has something to discount. Live data is
            # the other way round on 83% of days; test_eth_dcf_refuses... covers that.
            "net_revenue_usd": np.full(days, 3.0e6),
            "issuance_usd": np.full(days, 2.0e6),
            "revenue_regime": "retains_revenue",
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_three_stage_pv_is_a_discounted_sum_and_reports_its_terminal_share():
    pv, terminal_share = three_stage_pv(
        cash_flow=100.0,
        growth=(0.2, 0.1, 0.03),
        discount=0.15,
        years=(3, 3, 4),
        terminal_multiple=10.0,
    )
    assert pv > 0
    assert 0.0 <= terminal_share <= 1.0
    # A lower discount rate makes every future dollar worth more today: the PV must rise.
    pv_cheap, _ = three_stage_pv(100.0, (0.2, 0.1, 0.03), 0.05, (3, 3, 4), 10.0)
    assert pv_cheap > pv


def test_three_stage_pv_refuses_a_discount_rate_below_terminal_growth():
    # The Gordon terminal term is undefined when g >= r: it must raise, not return a
    # negative or an infinity that would travel into a fair value.
    with pytest.raises(ValueError, match="terminal growth"):
        three_stage_pv(100.0, (0.2, 0.1, 0.12), 0.10, (3, 3, 4), None)


def test_grid_expands_a_sensitivity_block_into_param_sets():
    cells = grid(
        {
            "discount_rate": 0.15,
            "sensitivity": {"discount_rate": [0.10, 0.20], "terminal_multiple": [10.0, 20.0]},
        }
    )
    assert len(cells) == 4
    assert {c["discount_rate"] for c in cells} == {0.10, 0.20}
    assert all("sensitivity" not in c for c in cells), "a cell must not carry the grid that made it"


def test_eth_dcf_emits_one_row_per_grid_cell_with_distinct_hashes():
    frame = _cash_frame()
    estimates = EthThreeStageDcf().estimate(frame, dict(EthThreeStageDcf.spec.params))
    latest = [
        e for e in estimates if e.metric_date == frame["metric_date"].iloc[-1] and e.applicable
    ]
    cells = len(grid(dict(EthThreeStageDcf.spec.params)))
    assert len(latest) == cells
    assert len({e.param_set_hash for e in latest}) == cells
    assert all("terminal_share" in e.diagnostics for e in latest)


def test_eth_dcf_discounts_the_burn_not_the_whole_fee_stream():
    """Step 1's measurement: DefiLlama's Ethereum Revenue IS the burn, and the burn is what
    accrues to every holder. Priority fees and MEV go to proposers, so a DCF built on total
    fees would credit holders with money that reaches block builders."""
    frame = _cash_frame()
    params = {**EthThreeStageDcf.spec.params, "sensitivity": {}}
    last = [e for e in EthThreeStageDcf().estimate(frame, params) if e.applicable][-1]
    assert last.inputs_used["cash_flow_basis"] == "burn_net_of_issuance"
    # burn 3.0e6 - issuance 2.0e6 = 1.0e6/day, annualised.
    assert last.inputs_used["annual_cash_flow_usd"] == pytest.approx(365 * 1.0e6)

    # Raising total fees alone must change nothing: they are not the cash flow.
    richer = frame.assign(total_fees_usd=9.0e6)
    same = [e for e in EthThreeStageDcf().estimate(richer, params) if e.applicable][-1]
    assert same.fair_value == pytest.approx(last.fair_value)


def test_eth_dcf_refuses_when_issuance_exceeds_the_burn():
    """The live case: annualised issuance of $2.49bn against a $0.05bn burn, on 83% of the
    days since EIP-1559. A negative cash flow is not a cheap asset."""
    frame = _cash_frame().assign(net_revenue_usd=1.0e6, issuance_usd=2.0e6)
    estimates = EthThreeStageDcf().estimate(frame, dict(EthThreeStageDcf.spec.params))
    assert all(not e.applicable for e in estimates)
    assert all(e.fair_value is None for e in estimates)
    assert "no positive annualised burn_net_of_issuance" in estimates[-1].reason


def test_eth_dcf_flags_a_cell_whose_value_is_mostly_terminal():
    frame = _cash_frame()
    params = {
        **EthThreeStageDcf.spec.params,
        "discount_rate": 0.06,
        "terminal_multiple": 60.0,
        "sensitivity": {},
        "max_terminal_share": 0.8,
    }
    last = [e for e in EthThreeStageDcf().estimate(frame, params) if e.applicable][-1]
    assert last.diagnostics["terminal_share"] > 0.8
    assert last.diagnostics["terminal_dominated"] is True
    # Flagged, not refused: the number is arithmetically right and interpretively weak.
    assert last.fair_value is not None


def test_defi_dcf_refuses_a_protocol_that_retains_no_revenue():
    frame = _cash_frame().assign(revenue_regime="fees_fully_distributed", net_revenue_usd=0.0)
    estimates = DefiNetRevenueDcf().estimate(frame, dict(DefiNetRevenueDcf.spec.params))
    assert all(not e.applicable for e in estimates)
    assert "retain" in estimates[-1].reason


def test_bond_target_yield_divides_annual_fees_by_an_explicit_target():
    params = {
        "window_rows": 365,
        "risk_free_rate": 0.04,
        "staking_risk_premium": 0.06,
        "denominator": "total_fees_usd",
    }
    last = [e for e in BondTargetYield().estimate(_cash_frame(), params) if e.applicable][-1]
    assert last.inputs_used["target_yield"] == pytest.approx(0.10)
    annual = 365 * 3.0e6
    assert last.fair_value == pytest.approx(annual / 0.10 / 1.21e8, rel=1e-3)


def test_real_yield_subtracts_dilution_and_says_the_incentive_term_is_missing():
    last = [
        e
        for e in EmissionsAdjustedRealYield().estimate(
            _cash_frame(), dict(EmissionsAdjustedRealYield.spec.params)
        )
        if e.applicable
    ][-1]
    assert last.fair_value is None, "a yield is not a price"
    assert last.diagnostics["incentive_share_available"] is False
    assert last.diagnostics["real_yield"] < last.diagnostics["fee_yield"]


def test_counter_model_holds_network_value_constant_and_says_so():
    last = [
        e
        for e in SupplyEffectCounterModel().estimate(
            _cash_frame(), dict(SupplyEffectCounterModel.spec.params)
        )
        if e.applicable
    ][-1]
    assert last.diagnostics["holds_network_value_constant"] is True
    assert last.inputs_used["projected_supply"] > 0
    # Supply is growing in the fixture, so the counter-model's value must sit BELOW price.
    assert last.fair_value < 2500.0


def test_counter_model_reads_the_observed_supply_path_not_a_fee_burn_assumption():
    """The brief reconstructed the supply path from issuance minus 100% of fees burned,
    which is false for ETH (the burn is 20-50% of fees) and for every chain burning none."""
    frame = _cash_frame()
    params = dict(SupplyEffectCounterModel.spec.params)
    base = [e for e in SupplyEffectCounterModel().estimate(frame, params) if e.applicable][-1]
    # Fees and issuance are not inputs at all: moving them must change nothing.
    moved = frame.assign(total_fees_usd=9.9e9, issuance_usd=9.9e9)
    after = [e for e in SupplyEffectCounterModel().estimate(moved, params) if e.applicable][-1]
    assert after.fair_value == pytest.approx(base.fair_value)
    assert "annual_supply_growth" in base.inputs_used

    # A shrinking supply must value the token ABOVE today's price, not below.
    deflating = frame.assign(supply=np.linspace(1.21e8, 1.2e8, len(frame)))
    down = [e for e in SupplyEffectCounterModel().estimate(deflating, params) if e.applicable][-1]
    assert down.inputs_used["annual_supply_growth"] < 0
    assert down.fair_value > down.inputs_used["market_cap_usd"] / down.inputs_used["supply"]
