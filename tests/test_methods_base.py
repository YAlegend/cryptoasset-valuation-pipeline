import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.base import (
    MARKET_CAP_DERIVED,
    Estimate,
    MethodSpec,
    circularity,
    resolve_applicability,
)

SPEC = MethodSpec(
    method_id="realized_price",
    family="E_cost",
    requires=("realized_cap", "supply"),
    min_density=0.5,
    min_rows=30,
    params={},
    citation="Coinmetrics (2018)",
)


def _frame(n: int = 100, realized_nulls: int = 0) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "symbol": ["BTC"] * n,
            "metric_date": pd.date_range("2026-01-01", periods=n, freq="D").date,
            "realized_cap": np.linspace(1e12, 1.1e12, n),
            "supply": np.linspace(1.9e7, 1.95e7, n),
            "panel_vintage_ts": pd.Timestamp("2026-10-12", tz="UTC"),
        }
    )
    if realized_nulls:
        frame.loc[: realized_nulls - 1, "realized_cap"] = np.nan
    return frame


def test_a_complete_frame_is_applicable():
    ok, reason = resolve_applicability(SPEC, _frame())
    assert ok and reason == ""


def test_a_missing_column_is_not_applicable_and_names_it():
    ok, reason = resolve_applicability(SPEC, _frame().drop(columns=["supply"]))
    assert not ok
    assert "supply" in reason and "absent" in reason


def test_a_column_below_the_density_floor_is_not_applicable():
    """Review Focus 3: BNB's market cap is 19% dense. A fit on 646 scattered days
    converges and returns a confident number, so density gates the method."""
    ok, reason = resolve_applicability(SPEC, _frame(n=100, realized_nulls=81))
    assert not ok
    assert "realized_cap" in reason
    assert "19.0% dense" in reason
    assert "floor 50.0%" in reason


def test_too_few_rows_is_not_applicable():
    ok, reason = resolve_applicability(SPEC, _frame(n=10))
    assert not ok
    assert "10 rows" in reason and "30" in reason


def test_an_empty_frame_is_not_applicable_rather_than_an_error():
    ok, reason = resolve_applicability(SPEC, _frame().iloc[0:0])
    assert not ok
    assert "no rows" in reason


def test_a_method_consuming_a_market_cap_derived_column_is_flagged_circular():
    """Review Focus 1: QTM with panel velocity is volume/market_cap, so the expression
    collapses to price/circulating_fraction and returns the market price back."""
    qtm = MethodSpec(
        method_id="crypto_qtm",
        family="A_monetary",
        requires=("volume_total_usd", "supply", "velocity_total"),
        min_density=0.5,
        min_rows=30,
        params={},
        citation="dissertation 3.1",
    )
    assert circularity(qtm) == ("velocity_total",)


def test_a_method_on_independent_fundamentals_is_not_circular():
    metcalfe = MethodSpec(
        method_id="metcalfe_estimated",
        family="B_network",
        requires=("market_cap_usd", "active_addresses", "supply"),
        min_density=0.5,
        min_rows=180,
        params={},
        citation="Metcalfe (1995)",
    )
    # market_cap_usd itself is the REGRESSAND, not a derived input, so it is not circular.
    assert circularity(metcalfe) == ()


def test_every_market_cap_derived_column_is_named():
    # Each of these is literally computed from market cap in fct_asset_metrics_daily, so a
    # method consuming one and returning a price is reusing its own answer.
    assert MARKET_CAP_DERIVED == frozenset(
        {
            "velocity_reported",
            "velocity_total",
            "mvrv",
            "mvtv",
            "realized_cap",
            "investor_cap_usd",
        }
    )


def test_an_estimate_refuses_a_fair_value_when_inapplicable():
    with pytest.raises(ValueError, match="inapplicable estimate cannot carry a fair_value"):
        Estimate(
            symbol="BNB",
            metric_date=dt.date(2026, 10, 12),
            method_id="realized_price",
            param_set_hash="abc",
            fair_value=42.0,
            band_low=None,
            band_high=None,
            inputs_used={},
            diagnostics={},
            applicable=False,
            reason="market_cap_usd is 19.1% dense",
            panel_vintage_ts=pd.Timestamp("2026-10-12", tz="UTC"),
        )


def test_an_estimate_refuses_an_inverted_band():
    with pytest.raises(ValueError, match="band_low .* above band_high"):
        Estimate(
            symbol="BTC",
            metric_date=dt.date(2026, 10, 12),
            method_id="metcalfe_estimated",
            param_set_hash="abc",
            fair_value=50000.0,
            band_low=60000.0,
            band_high=40000.0,
            inputs_used={},
            diagnostics={},
            applicable=True,
            reason="",
            panel_vintage_ts=pd.Timestamp("2026-10-12", tz="UTC"),
        )


def test_a_late_starting_column_is_not_counted_as_sparsity():
    """CoinGecko's keyless tier serves 365 days, so a volume column starts in 2025 while
    Coin Metrics' start in 2015. The void before it is absence of history, not sparsity."""
    from cryptoval.methods.base import covered_span

    frame = _frame(n=100, realized_nulls=60)
    span = covered_span(SPEC, frame)
    assert len(span) == 40
    ok, reason = resolve_applicability(SPEC, span)
    assert ok, reason


def test_a_wholly_empty_required_column_still_reports_its_density():
    from cryptoval.methods.base import covered_span

    frame = _frame()
    frame["realized_cap"] = np.nan
    ok, reason = resolve_applicability(SPEC, covered_span(SPEC, frame))
    assert not ok and "0.0% dense" in reason
