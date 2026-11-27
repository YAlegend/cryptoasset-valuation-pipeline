import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.multiples import (
    DfdvToRevenue,
    FdvToRevenue,
    NvtGoldenRatio,
    NvtReversion,
    PriceToFees,
    PriceToSales,
    PriceToTvl,
    PriceToUtility,
    dilution_rate,
)


def _frame(n: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTC"] * n,
            "metric_date": pd.date_range("2024-01-01", periods=n, freq="D").date,
            "market_cap_usd": np.linspace(1.0e12, 1.7e12, n),
            # Deliberately NOT a constant multiple of volume_total_usd: NVT reversion
            # is invariant to rescaling volume by a constant (V x median(M/V) cancels
            # it), so only a different SHAPE makes the two series distinguishable --
            # which is what the real pair does, growing at different rates.
            "volume_reported_spot_usd": np.linspace(2.0e9, 4.0e9, n),
            "volume_total_usd": np.linspace(1.0e10, 1.5e10, n),
            "velocity_total": np.linspace(0.009, 0.010, n),
            "supply": np.linspace(1.9e7, 1.96e7, n),
            "price_usd": np.linspace(52000, 87000, n),
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )


def test_dilution_rate_is_annualised_supply_growth():
    supply = pd.Series(np.linspace(100.0, 102.0, 731))
    rate = dilution_rate(supply, window_rows=365)
    # Roughly 1% over the trailing year at the midpoint.
    assert rate.iloc[365] == pytest.approx(0.01, rel=0.15)
    # The first year has no trailing year to compare against.
    assert rate.iloc[:365].isna().all()


def test_nvt_uses_the_declared_volume_series():
    frame = _frame()
    reported = NvtReversion().estimate(
        frame, {"volume_column": "volume_reported_spot_usd", "min_history_rows": 180}
    )
    total = NvtReversion().estimate(
        frame, {"volume_column": "volume_total_usd", "min_history_rows": 180}
    )
    last_reported = [e for e in reported if e.applicable][-1]
    last_total = [e for e in total if e.applicable][-1]
    # Spec 10.1: the two volume series differ ~5x, so NVT built on them must differ too.
    # If a model had coalesced them these would be equal and the discipline would be lost.
    assert last_reported.fair_value != pytest.approx(last_total.fair_value)
    assert last_reported.inputs_used["volume_column"] == "volume_reported_spot_usd"


def test_the_golden_ratio_compares_nvt_to_its_two_year_average():
    out = NvtGoldenRatio().estimate(_frame(), {"volume_column": "volume_total_usd", "ma_rows": 730})
    last = [e for e in out if e.applicable][-1]
    assert last.diagnostics["nvt"] > 0
    assert last.diagnostics["nvt_ma"] > 0
    assert last.diagnostics["golden_ratio"] == pytest.approx(
        last.diagnostics["nvt"] / last.diagnostics["nvt_ma"]
    )


def test_the_golden_ratio_needs_two_years_before_it_says_anything():
    out = NvtGoldenRatio().estimate(
        _frame(n=500), {"volume_column": "volume_total_usd", "ma_rows": 730}
    )
    assert not any(e.applicable for e in out)
    assert "730" in out[-1].reason


def test_price_to_utility_refuses_without_a_staking_ratio():
    """The free tier serves no staking ratio, so the method declines rather than guessing."""
    out = PriceToUtility().estimate(_frame(), {"staking_ratio": None, "min_history_rows": 180})
    assert not any(e.applicable for e in out)
    assert "staking_ratio" in out[-1].reason


def test_price_to_utility_computes_when_the_ratio_is_supplied():
    out = PriceToUtility().estimate(_frame(), {"staking_ratio": 0.25, "min_history_rows": 180})
    last = [e for e in out if e.applicable][-1]
    assert last.diagnostics["token_utility"] > 0
    assert last.inputs_used["staking_ratio"] == 0.25
    # The holding-time correction's finding applies here too: P/U inverts velocity.
    assert last.diagnostics["uses_velocity_inversion"] is True


def _wide_frame(days: int = 500, max_supply: float | None = 2.1e7) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")
    price = 100.0 + np.arange(days) * 0.1
    supply = np.full(days, 1.0e7)
    return pd.DataFrame(
        {
            "symbol": "AAVE",
            "metric_date": [d.date() for d in dates],
            "price_usd": price,
            "supply": supply,
            "market_cap_usd": price * supply,
            "total_fees_usd": np.full(days, 5.0e5),
            "net_revenue_usd": np.full(days, 2.0e5),
            "tvl_usd": np.full(days, 1.0e10),
            "total_supply": np.full(days, 1.6e7),
            "max_supply": np.full(days, max_supply if max_supply is not None else np.nan),
            "supply_as_of": [pd.Timestamp("2026-10-13").date()] * days,
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_price_to_fees_annualises_the_trailing_window_and_names_its_denominator():
    last = [
        e
        for e in PriceToFees().estimate(_wide_frame(), dict(PriceToFees.spec.params))
        if e.applicable
    ][-1]
    # 365 rows x 5.0e5 per day of fees, against a target multiple.
    assert last.inputs_used["denominator"] == "total_fees_usd"
    assert last.inputs_used["annual_value_usd"] == pytest.approx(365 * 5.0e5)
    assert last.diagnostics["realised_multiple"] > 0


def test_price_to_sales_uses_net_revenue_not_gross_fees():
    last = [
        e
        for e in PriceToSales().estimate(_wide_frame(), dict(PriceToSales.spec.params))
        if e.applicable
    ][-1]
    assert last.inputs_used["denominator"] == "net_revenue_usd"
    assert last.inputs_used["annual_value_usd"] == pytest.approx(365 * 2.0e5)


def test_price_to_sales_refuses_a_negative_trailing_revenue_rather_than_inverting_it():
    """Net revenue may be negative. A negative denominator would hand back a negative fair
    value, and the sign would read as a cheap asset rather than as a loss-making protocol."""
    frame = _wide_frame().assign(net_revenue_usd=-2.0e5)
    estimates = PriceToSales().estimate(frame, dict(PriceToSales.spec.params))
    assert all(not e.applicable for e in estimates)
    assert all(e.fair_value is None for e in estimates)


def test_fdv_to_revenue_refuses_when_max_supply_is_null():
    estimates = FdvToRevenue().estimate(
        _wide_frame(max_supply=None), dict(FdvToRevenue.spec.params)
    )
    assert all(not e.applicable for e in estimates)
    assert "max_supply" in estimates[-1].reason
    # And it must NOT have fallen back to total_supply.
    assert all(e.fair_value is None for e in estimates)


def test_fdv_uses_max_supply_and_dfdv_uses_less_so_fdv_values_each_token_lower():
    """The two differ only in the supply the numerator implies, and more supply over the
    same revenue is less value per token. If this inverts, the two have been swapped."""
    frame = _wide_frame()
    fdv = [
        e for e in FdvToRevenue().estimate(frame, dict(FdvToRevenue.spec.params)) if e.applicable
    ][-1]
    dfdv = [
        e for e in DfdvToRevenue().estimate(frame, dict(DfdvToRevenue.spec.params)) if e.applicable
    ][-1]
    assert fdv.inputs_used["supply"] == 2.1e7
    assert dfdv.inputs_used["effective_supply"] == pytest.approx(1.3e7)
    assert fdv.fair_value < dfdv.fair_value


def test_dfdv_haircuts_only_the_locked_portion():
    frame = _wide_frame()
    params = {
        **DfdvToRevenue.spec.params,
        "locked_haircut": 0.5,
        "window_rows": 365,
        "target_multiple": 20.0,
    }
    last = [e for e in DfdvToRevenue().estimate(frame, params) if e.applicable][-1]
    locked = 1.6e7 - 1.0e7  # total minus circulating
    assert last.inputs_used["locked_supply"] == pytest.approx(locked)
    assert last.inputs_used["effective_supply"] == pytest.approx(1.0e7 + 0.5 * locked)


def test_dfdv_refuses_a_circulating_supply_above_the_vendor_total():
    """A negative lock would haircut the supply UPWARDS. Fires on 7 of 24 live assets,
    because circulating is Coin Metrics' figure and total is CoinGecko's."""
    frame = _wide_frame().assign(total_supply=0.9e7)
    estimates = DfdvToRevenue().estimate(frame, dict(DfdvToRevenue.spec.params))
    assert all(not e.applicable for e in estimates)
    assert "exceeds total supply" in estimates[-1].reason


def test_dfdv_measures_how_stale_the_point_in_time_supply_is_on_each_row():
    """total_supply is one snapshot applied to every date, so the locked portion is an
    artefact of the join the further back the row sits. The row has to say by how much."""
    last = [
        e
        for e in DfdvToRevenue().estimate(_wide_frame(), dict(DfdvToRevenue.spec.params))
        if e.applicable
    ][-1]
    # Last row is 2026-05-15; the snapshot is 2026-10-13, so 151 days of drift on the
    # NEWEST row, and over a year on the oldest.
    assert last.diagnostics["supply_age_days"] == 151
    first = [
        e
        for e in DfdvToRevenue().estimate(_wide_frame(), dict(DfdvToRevenue.spec.params))
        if e.applicable
    ][0]
    assert first.diagnostics["supply_age_days"] > last.diagnostics["supply_age_days"]


def test_price_to_tvl_refuses_an_asset_with_no_tvl():
    frame = _wide_frame().assign(tvl_usd=np.nan)
    estimates = PriceToTvl().estimate(frame, dict(PriceToTvl.spec.params))
    assert all(not e.applicable for e in estimates)
    assert "tvl" in estimates[-1].reason.lower()


def test_price_to_tvl_refuses_a_zero_tvl_rather_than_dividing_by_it():
    """LINK reads 0.0 on all 1,039 days: DefiLlama's TVL excludes a protocol's own staked
    governance token, so the zero is the vendor's definition, not a gap."""
    frame = _wide_frame().assign(tvl_usd=0.0)
    estimates = PriceToTvl().estimate(frame, dict(PriceToTvl.spec.params))
    assert all(not e.applicable for e in estimates)
    assert all(e.fair_value is None for e in estimates)


def test_price_to_tvl_does_not_annualise_a_stock():
    last = [
        e
        for e in PriceToTvl().estimate(_wide_frame(), dict(PriceToTvl.spec.params))
        if e.applicable
    ][-1]
    assert last.inputs_used["annual_value_usd"] == pytest.approx(1.0e10), "TVL is a stock"
