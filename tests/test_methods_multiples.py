import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.multiples import (
    NvtGoldenRatio,
    NvtReversion,
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
