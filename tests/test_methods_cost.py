import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.base import circularity
from cryptoval.methods.cost import (
    EnergyValue,
    HayesCostOfProduction,
    MayerMultiple,
    MvrvZScore,
    PuellMultiple,
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


def _cost_frame(days: int = 900) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D", tz="UTC")
    price = 100.0 + np.arange(days) * 0.5
    supply = 1.0e7 + np.arange(days) * 50.0
    return pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": [d.date() for d in dates],
            "price_usd": price,
            "supply": supply,
            "market_cap_usd": price * supply,
            "realized_cap": price * supply * 0.7,
            "issuance_usd": np.full(days, 1.0e6),
            "hash_rate": np.full(days, 1.0e9),
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_mayer_multiple_is_price_over_its_own_trailing_mean():
    frame = _cost_frame()
    estimates = MayerMultiple().estimate(frame, {"window_rows": 200, "target_multiple": 1.0})
    last = [e for e in estimates if e.applicable][-1]
    # The baseline the fair value is built from excludes today (base.py's window rule);
    # the published multiple keeps the inclusive definition, and both are on the row.
    prior_ma = frame["price_usd"].iloc[-201:-1].mean()
    inclusive_ma = frame["price_usd"].iloc[-200:].mean()
    assert last.inputs_used["moving_average_price"] == pytest.approx(prior_ma)
    assert last.fair_value == pytest.approx(prior_ma)
    assert last.diagnostics["mayer_multiple"] == pytest.approx(
        frame["price_usd"].iloc[-1] / inclusive_ma
    )
    assert last.diagnostics["baseline_basis"] == "prior_only"


def test_puell_multiple_refuses_before_a_full_365_row_window():
    estimates = PuellMultiple().estimate(_cost_frame(days=300), dict(PuellMultiple.spec.params))
    assert all(not e.applicable for e in estimates)
    # The refusal names the measured count and the threshold, not just the threshold.
    assert estimates[-1].reason == "only 299 prior issuance observations to date, need 365"


def test_a_refusal_distinguishes_too_short_from_a_gapped_window():
    frame = _cost_frame(days=300)
    frame.loc[250:260, "price_usd"] = np.nan  # 289 observations, but a hole inside the window
    last = MayerMultiple().estimate(frame, {"window_rows": 200})[-1]
    assert (
        last.reason
        == "only 189 price observations in the last 200 rows, need 200 (gaps in the window)"
    )
    short = MayerMultiple().estimate(_cost_frame(days=120), {"window_rows": 200})[-1]
    assert short.reason == "only 119 prior price observations to date, need 200"


def test_puell_names_which_input_is_missing_today():
    frame = _cost_frame(days=400)
    frame.loc[399, "issuance_usd"] = 0.0
    frame.loc[398, "price_usd"] = np.nan
    out = PuellMultiple().estimate(frame, {"window_rows": 365})
    assert out[-1].reason == "no issuance on this day (issuance_usd=0.0)"
    assert out[-2].reason == "no price on this day (price_usd=nan)"


def test_puell_multiple_is_today_over_trailing_mean():
    frame = _cost_frame(days=400)
    frame.loc[399, "issuance_usd"] = 2.0e6
    last = PuellMultiple().estimate(frame, {"window_rows": 365, "target_multiple": 1.0})[-1]
    inclusive_mean = frame["issuance_usd"].iloc[-365:].mean()
    prior_mean = frame["issuance_usd"].iloc[-366:-1].mean()
    assert last.diagnostics["puell_multiple"] == pytest.approx(2.0e6 / inclusive_mean)
    assert last.fair_value == pytest.approx(frame["price_usd"].iloc[-1] * prior_mean / 2.0e6)


def test_mvrv_z_score_is_flagged_circular_because_realized_cap_is_derived():
    assert "realized_cap" in circularity(MvrvZScore.spec)
    last = [
        e
        for e in MvrvZScore().estimate(_cost_frame(), dict(MvrvZScore.spec.params))
        if e.applicable
    ][-1]
    assert "z_score" in last.diagnostics


def test_mvrv_z_score_refusal_names_the_measurement():
    out = MvrvZScore().estimate(_cost_frame(days=100), {"window_rows": 365, "target_z": 0.0})
    assert out[-1].reason == "only 99 prior market-cap observations to date, need 365"
    flat = _cost_frame(days=400)
    flat["market_cap_usd"] = 5.0e9
    assert MvrvZScore().estimate(flat, {"window_rows": 365})[-1].reason == (
        "market cap has zero standard deviation over the last 365 rows"
    )


def test_energy_value_is_exactly_twice_smoothed_hayes():
    frame = _cost_frame()
    params = {
        "window_rows": 365,
        "joules_per_hash": 2.0e-11,
        "electricity_usd_per_kwh": 0.05,
        "hashes_per_unit": 1.0e12,
    }
    last = [e for e in EnergyValue().estimate(frame, params) if e.applicable][-1]

    coins_per_day = frame["issuance_usd"] / frame["price_usd"]
    daily = [
        hayes_cost_per_coin(1.0e9, 1.0e12, 2.0e-11, 0.05, c) for c in coins_per_day.iloc[-366:-1]
    ]
    assert last.fair_value == pytest.approx(2.0 * float(np.mean(daily)))
    # The card must not claim independence from Hayes; the diagnostics say so on every row.
    assert last.diagnostics["derived_from"] == "hayes_cost_of_production"


def test_energy_value_refusal_counts_computable_days():
    params = dict(EnergyValue.spec.params)
    out = EnergyValue().estimate(_cost_frame(days=300), params)
    assert out[-1].reason == (
        "only 299 prior computable production-cost observations to date, need 365"
    )


@pytest.mark.parametrize(
    "method, params",
    [
        (MayerMultiple(), {"window_rows": 200}),
        (PuellMultiple(), {"window_rows": 365}),
        (MvrvZScore(), {"window_rows": 365, "target_z": 0.0}),
        (EnergyValue(), dict(EnergyValue.spec.params)),
    ],
)
def test_no_window_method_values_a_row_with_a_later_one(method, params):
    # The property that matters most, measured per base.py's window rule: shocking the
    # last row moves no earlier row, and moves no earlier row's inputs either.
    frame = _cost_frame(days=500)
    before = [e for e in method.estimate(frame, params) if e.applicable]
    shocked = frame.copy()
    for column in ("price_usd", "market_cap_usd", "issuance_usd", "hash_rate"):
        shocked.loc[shocked.index[-1], column] *= 100.0
    after = [e for e in method.estimate(shocked, params) if e.applicable]
    assert len(before) == len(after)
    for b, a in zip(before[:-1], after[:-1], strict=True):
        assert b.metric_date == a.metric_date
        assert a.fair_value == pytest.approx(b.fair_value)


def test_the_mayer_baseline_ignores_its_own_day():
    # Mayer's fair value is compared against today's price, so today's price must not be
    # inside it. Shocking the last price moves the multiple, never the baseline.
    frame = _cost_frame(days=500)
    params = {"window_rows": 200, "target_multiple": 1.0}
    before = [e for e in MayerMultiple().estimate(frame, params) if e.applicable][-1]
    shocked = frame.copy()
    shocked.loc[shocked.index[-1], "price_usd"] *= 100.0
    after = [e for e in MayerMultiple().estimate(shocked, params) if e.applicable][-1]
    assert after.fair_value == pytest.approx(before.fair_value)
    assert after.diagnostics["mayer_multiple"] > before.diagnostics["mayer_multiple"]


def test_the_puell_baseline_ignores_its_own_day():
    frame = _cost_frame(days=500)
    params = {"window_rows": 365, "target_multiple": 1.0}
    before = [e for e in PuellMultiple().estimate(frame, params) if e.applicable][-1]
    shocked = frame.copy()
    shocked.loc[shocked.index[-1], "issuance_usd"] *= 100.0
    after = [e for e in PuellMultiple().estimate(shocked, params) if e.applicable][-1]
    assert after.inputs_used["trailing_mean_issuance_usd"] == pytest.approx(
        before.inputs_used["trailing_mean_issuance_usd"]
    )


def test_the_window_boundary_is_exact_once_today_is_excluded():
    # window prior observations are needed, so a frame of exactly `window` rows has none.
    params = {"window_rows": 200, "target_multiple": 1.0}
    assert not any(e.applicable for e in MayerMultiple().estimate(_cost_frame(days=200), params))
    first = [e for e in MayerMultiple().estimate(_cost_frame(days=201), params) if e.applicable]
    assert len(first) == 1


def test_the_vendor_lagged_newest_day_still_gets_an_estimate():
    # Consequence of the prior-only baseline, and an improvement: the newest panel row has
    # no price yet (the vendor lags a day) but a baseline built from prior rows exists, so
    # the method answers instead of refusing. Only the inclusive multiple is unavailable.
    frame = _cost_frame(days=400)
    frame.loc[399, "price_usd"] = np.nan
    last = MayerMultiple().estimate(frame, {"window_rows": 200})[-1]
    assert last.applicable and last.fair_value > 0
    assert last.diagnostics["mayer_multiple"] is None


def test_an_infinity_is_not_counted_as_an_observation():
    frame = _cost_frame(days=400)
    frame.loc[380, "price_usd"] = np.inf
    last = MayerMultiple().estimate(frame, {"window_rows": 200})[-1]
    assert last.reason == (
        "only 199 price observations in the last 200 rows, need 200 (gaps in the window)"
    )


def test_energy_value_shares_hayes_own_parameters():
    # The card claims "the same fleet efficiency and electricity price". Nothing enforced it.
    hayes = dict(HayesCostOfProduction.spec.params)
    energy = dict(EnergyValue.spec.params)
    for key in ("hashes_per_unit", "joules_per_hash", "electricity_usd_per_kwh"):
        assert energy[key] == hayes[key], key


def test_both_hash_rate_methods_name_the_hardware_they_were_calibrated_for():
    # Ruling 16: the parameters describe SHA-256 hardware, so a Scrypt asset's number is
    # wrong by orders of magnitude rather than slightly off, and the row has to say so.
    frame = _cost_frame()
    hayes = [
        e
        for e in HayesCostOfProduction().estimate(frame, dict(HayesCostOfProduction.spec.params))
        if e.applicable
    ][-1]
    energy = [
        e for e in EnergyValue().estimate(frame, dict(EnergyValue.spec.params)) if e.applicable
    ][-1]
    assert hayes.diagnostics["parameters_calibrated_for"].startswith("sha256")
    assert energy.diagnostics["parameters_calibrated_for"].startswith("sha256")


def test_mvrv_z_at_target_zero_declares_that_it_is_realized_price():
    frame = _cost_frame()
    z = [
        e
        for e in MvrvZScore().estimate(frame, {"window_rows": 365, "target_z": 0.0})
        if e.applicable
    ][-1]
    realized = [
        e for e in RealizedPrice().estimate(frame, dict(RealizedPrice.spec.params)) if e.applicable
    ][-1]
    assert z.fair_value == pytest.approx(realized.fair_value)
    assert z.diagnostics["derived_from"] == "realized_price"
    # At a non-zero target the standard deviation does reach the fair value, so the
    # duplication claim would be false and the row must not make it.
    shifted = [
        e
        for e in MvrvZScore().estimate(frame, {"window_rows": 365, "target_z": -0.5})
        if e.applicable
    ][-1]
    assert shifted.diagnostics["derived_from"] is None
    assert shifted.fair_value < realized.fair_value
