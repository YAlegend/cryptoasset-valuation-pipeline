import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.cost import (
    HayesCostOfProduction,
    RealizedPrice,
    ThermocapFloor,
    hayes_cost_per_coin,
)
from cryptoval.methods.registry import param_set_hash


def _pow_frame(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTC"] * n,
            "metric_date": pd.date_range("2025-01-01", periods=n, freq="D").date,
            "realized_cap": np.linspace(1.0e12, 1.08e12, n),
            "thermocap_usd": np.linspace(4.0e11, 4.2e11, n),
            "investor_cap_usd": np.linspace(6.0e11, 6.6e11, n),
            "market_cap_usd": np.linspace(1.5e12, 1.7e12, n),
            "mvrv": np.linspace(1.5, 1.6, n),
            "mvtv": np.linspace(3.7, 4.0, n),
            "supply": np.linspace(1.95e7, 1.96e7, n),
            "price_usd": np.linspace(77000, 87000, n),
            # ~900 TH/s... in Coin Metrics' own unit, which Step 5 pins empirically.
            "hash_rate": np.full(n, 9.0e8),
            "issuance_usd": np.full(n, 4.0e7),
            "thermocap_has_gap": False,
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )


def test_realized_price_is_realized_cap_per_unit():
    frame = _pow_frame()
    out = RealizedPrice().estimate(frame, {})
    assert len(out) == len(frame)
    expected = frame["realized_cap"].iloc[-1] / frame["supply"].iloc[-1]
    assert out[-1].fair_value == pytest.approx(expected)
    assert out[-1].inputs_used["realized_cap"] == pytest.approx(frame["realized_cap"].iloc[-1])


def test_the_thermocap_floor_flags_a_gap_as_a_lower_bound():
    frame = _pow_frame()
    frame.loc[10:, "thermocap_has_gap"] = True
    out = ThermocapFloor().estimate(frame, {})
    last = out[-1]
    # A gapped thermocap understates the cumulative cost, so the figure is a LOWER bound
    # and anything derived from it (MVTV, investor cap) is biased UPWARD.
    assert last.diagnostics["is_lower_bound"] is True
    assert "understate" in last.diagnostics["bias_direction"]
    assert last.band_high is None
    assert last.band_low == pytest.approx(last.fair_value)


def test_a_zero_supply_day_yields_no_estimate_rather_than_infinity():
    frame = _pow_frame(n=40)
    frame.loc[5, "supply"] = 0.0
    out = RealizedPrice().estimate(frame, {})
    at_zero = [e for e in out if e.metric_date == frame["metric_date"].iloc[5]][0]
    assert at_zero.fair_value is None
    assert not at_zero.applicable
    assert "supply" in at_zero.reason


def test_hayes_cost_per_coin_is_energy_cost_divided_by_coins_mined():
    # 1e9 hashes/s at 1e-10 J per hash = 0.1 W... deliberately tiny, so the arithmetic is
    # checkable by hand rather than by reimplementing the function.
    cost = hayes_cost_per_coin(
        hash_rate=1.0e9,
        hashes_per_unit=1.0,
        joules_per_hash=1.0e-10,
        electricity_usd_per_kwh=0.05,
        coins_per_day=2.0,
    )
    joules_per_day = 1.0e9 * 1.0 * 1.0e-10 * 86400
    kwh = joules_per_day / 3.6e6
    assert cost == pytest.approx(kwh * 0.05 / 2.0)


def test_hayes_returns_none_when_no_coins_were_mined():
    assert hayes_cost_per_coin(1.0e9, 1.0, 1.0e-10, 0.05, 0.0) is None
    assert hayes_cost_per_coin(1.0e9, 1.0, 1.0e-10, 0.05, float("nan")) is None


def test_hayes_is_inapplicable_without_hash_rate():
    frame = _pow_frame()
    frame["hash_rate"] = np.nan
    method = HayesCostOfProduction()
    from cryptoval.methods.base import resolve_applicability

    ok, reason = resolve_applicability(method.spec, frame)
    assert not ok
    assert "hash_rate" in reason


def test_hayes_records_both_parameters_in_inputs_used():
    out = HayesCostOfProduction().estimate(
        _pow_frame(),
        {
            "joules_per_hash": 2.0e-11,
            "electricity_usd_per_kwh": 0.05,
            "hashes_per_unit": 1.0e3,
        },
    )
    used = out[-1].inputs_used
    # A cost of production is only as meaningful as its two assumptions, so they travel
    # with every row rather than living only in the method card.
    assert used["electricity_usd_per_kwh"] == 0.05
    assert used["joules_per_hash"] == 2.0e-11


def test_param_sets_distinguish_two_electricity_assumptions():
    cheap = param_set_hash({"electricity_usd_per_kwh": 0.03, "joules_per_hash": 2.0e-11})
    dear = param_set_hash({"electricity_usd_per_kwh": 0.08, "joules_per_hash": 2.0e-11})
    assert cheap != dear


@pytest.mark.integration
def test_the_hayes_unit_puts_btc_cost_within_an_order_of_magnitude_of_price():
    """A unit error scales Hayes by 1000x while leaving the curve's shape plausible."""
    import duckdb

    from cryptoval.methods.registry import load_method_specs

    params = load_method_specs()["hayes_cost_of_production"].params
    con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
    hr, iss, px = con.execute(
        """
        select hash_rate, issuance_usd, price_usd from fct_asset_metrics_daily
        where symbol = 'BTC' and hash_rate is not null and issuance_usd is not null
          and price_usd is not null
        order by metric_date desc limit 1
        """
    ).fetchone()
    cost = hayes_cost_per_coin(
        hr,
        params["hashes_per_unit"],
        params["joules_per_hash"],
        params["electricity_usd_per_kwh"],
        iss / px,
    )
    assert 0.1 <= px / cost <= 10.0, f"implied cost {cost:,.2f} against price {px:,.2f}"
