import json

import pandas as pd
import pytest

import cryptoval.methods  # noqa: F401  registers every family
from cryptoval.methods.registry import METHODS, run_methods


def _panel(**over):
    row = dict(
        symbol="BTC",
        market_cap_usd=1000.0,
        price_usd=50.0,
        realized_cap=400.0,
        investor_cap_usd=300.0,
        liveliness=0.6,
        active_supply=12.0,
        utxo_supply=20.0,
        hodl_rc_0d=0.02,
        hodl_rc_6m_1y=0.1,
        sth_supply=4.0,
        sth_realized_cap=160.0,
        realized_profit_usd=5.0,
        panel_vintage_ts=pd.Timestamp("2024-02-15", tz="UTC"),
    )
    row.update(over)
    return pd.DataFrame([row] * 40).assign(metric_date=pd.date_range("2024-01-01", periods=40))


def _last(method_id, panel):
    return run_methods(panel, METHODS, [method_id]).iloc[-1]


@pytest.mark.parametrize(
    "method_id,expected",
    [("slrv", 0.2), ("rpv", 0.005), ("sthpl", 50 / 40), ("aviv", 1000 * 0.6 / 300)],
)
def test_signals_carry_their_value_in_diagnostics(method_id, expected):
    row = _last(method_id, _panel())
    assert row["applicable"] and pd.isna(row["fair_value"])
    assert json.loads(row["diagnostics"])[method_id] == pytest.approx(expected)


@pytest.mark.parametrize(
    "method_id,expected", [("active_price", 400 / 12), ("true_market_mean", 300 / 12)]
)
def test_cointime_prices_are_fair_values(method_id, expected):
    assert _last(method_id, _panel())["fair_value"] == pytest.approx(expected)


def test_a_non_positive_price_refuses_the_day():
    row = _last("true_market_mean", _panel(investor_cap_usd=-1.0))
    assert not row["applicable"] and "investor_cap_usd" in row["reason"]


def test_a_zero_denominator_refuses_the_day():
    row = _last("slrv", _panel(hodl_rc_6m_1y=0.0))
    assert not row["applicable"] and "hodl_rc_6m_1y" in row["reason"]


def test_an_asset_without_coin_age_data_refuses_with_its_density():
    row = _last("slrv", _panel(symbol="ETH", hodl_rc_0d=None, hodl_rc_6m_1y=None))
    assert not row["applicable"] and "0.0% dense" in row["reason"]


def test_methods_reading_market_cap_derived_columns_are_flagged_circular():
    out = run_methods(_panel(), METHODS, ["aviv", "active_price", "true_market_mean"])
    assert out.groupby("method_id")["circular_inputs"].first().str.len().gt(0).all()
