# tests/test_methods_benchmarks.py
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.benchmarks import PowerLaw, StockToFlow, fit_log_log


def _frame(days: int = 800, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=days, freq="D", tz="UTC")
    # A deterministic exponential path: both methods have a defined answer on it.
    supply = np.linspace(1.0e7, 1.9e7, days)
    issuance_coins = np.gradient(supply)
    price = start_price * np.exp(np.linspace(0.0, 1.5, days))
    return pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": [d.date() for d in dates],
            "price_usd": price,
            "supply": supply,
            "market_cap_usd": price * supply,
            "issuance_usd": issuance_coins * price,
            "genesis_date": dt.date(2009, 1, 3),
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_fit_log_log_recovers_a_known_exponent():
    x = pd.Series(np.linspace(1.0, 100.0, 400))
    y = pd.Series(3.0 * x**1.5)
    fit = fit_log_log(x, y, min_rows=100)
    assert fit is not None
    assert fit["beta"] == pytest.approx(1.5, abs=1e-6)
    assert fit["beta_ci_low"] <= fit["beta"] <= fit["beta_ci_high"]
    assert fit["n"] == 400


def test_fit_log_log_refuses_below_min_rows():
    x = pd.Series(np.linspace(1.0, 10.0, 20))
    assert fit_log_log(x, 2.0 * x, min_rows=100) is None


def test_stock_to_flow_reports_beta_and_the_post_2021_label():
    estimates = StockToFlow().estimate(_frame(), dict(StockToFlow.spec.params))
    applicable = [e for e in estimates if e.applicable]
    assert applicable, "stock-to-flow should be applicable on a full synthetic series"
    last = applicable[-1]
    assert last.fair_value > 0
    assert "beta" in last.diagnostics and "r_squared" in last.diagnostics
    # The shelf's whole point: the row says it is a falsified model, on every row.
    assert last.diagnostics["status"] == "falsified_post_2021"
    assert last.inputs_used["stock_to_flow"] > 0


def test_power_law_uses_days_since_genesis_and_refuses_without_one():
    frame = _frame()
    estimates = PowerLaw().estimate(frame, dict(PowerLaw.spec.params))
    last = [e for e in estimates if e.applicable][-1]
    assert last.inputs_used["days_since_genesis"] > 4000
    assert last.diagnostics["status"] == "contested"

    no_genesis = frame.assign(genesis_date=None)
    refusals = PowerLaw().estimate(no_genesis, dict(PowerLaw.spec.params))
    assert all(not e.applicable for e in refusals)
    assert "genesis" in refusals[-1].reason


def test_neither_shelf_method_is_flagged_circular():
    from cryptoval.methods.base import circularity

    assert circularity(StockToFlow.spec) == ()
    assert circularity(PowerLaw.spec) == ()


def test_fit_log_log_refuses_rather_than_raising_on_degenerate_input():
    # Constant x: beta is not identified, and conf_int() would be an IndexError.
    flat = pd.Series(np.full(400, 7.0))
    assert fit_log_log(flat, pd.Series(np.linspace(1.0, 2.0, 400)), min_rows=100) is None
    # inf passes a `> 0` filter but not a regression.
    x = pd.Series(np.linspace(1.0, 100.0, 400))
    dirty = x.copy()
    dirty.iloc[5] = np.inf
    fit = fit_log_log(dirty, 3.0 * x**1.5, min_rows=100)
    assert fit is not None and fit["n"] == 399


def test_fit_log_log_min_rows_is_a_boundary_not_a_range():
    x = pd.Series(np.linspace(1.0, 100.0, 300))
    y = 3.0 * x**1.5
    assert fit_log_log(x, y, min_rows=301) is None
    assert fit_log_log(x, y, min_rows=300) is not None


def test_fit_log_log_bands_the_fitted_value_not_the_slope():
    x = pd.Series(np.linspace(1.0, 100.0, 400))
    rng = np.random.default_rng(7)
    y = pd.Series(3.0 * x**1.5 * np.exp(rng.normal(0.0, 0.1, 400)))
    fit = fit_log_log(x, y, min_rows=100, predict_at=50.0)
    assert fit["fit_y_ci_low"] < fit["fit_y"] < fit["fit_y_ci_high"]
    assert fit["ci_basis"] == "naive_ols_mean_prediction"


def test_the_band_stays_ordered_when_the_ratio_is_below_one():
    # Three live assets (AVAX, MKR, USDC) sit here: log(ratio) < 0, which inverted the
    # old band built from beta's interval with alpha held fixed, and raised ValueError.
    frame = _frame()
    # Issuance larger than supply for a year => a stock-to-flow ratio under 1.
    frame["issuance_usd"] = frame["supply"] * frame["price_usd"] / 100.0
    estimates = StockToFlow().estimate(frame, dict(StockToFlow.spec.params))
    applicable = [e for e in estimates if e.applicable]
    assert applicable, "a sub-1 ratio is still a ratio, so these rows are applicable"
    assert all(e.inputs_used["stock_to_flow"] < 1.0 for e in applicable)
    assert all(e.band_low <= e.fair_value <= e.band_high for e in applicable)


def test_stock_to_flow_never_fits_on_its_own_row():
    # The no-look-ahead claim, measured: moving the last row cannot move its own answer.
    frame = _frame()
    before = [
        e for e in StockToFlow().estimate(frame, dict(StockToFlow.spec.params)) if e.applicable
    ]
    shocked = frame.copy()
    shocked.loc[shocked.index[-1], "market_cap_usd"] *= 100.0
    after = [
        e for e in StockToFlow().estimate(shocked, dict(StockToFlow.spec.params)) if e.applicable
    ]
    assert before[-1].metric_date == after[-1].metric_date
    assert after[-1].fair_value == pytest.approx(before[-1].fair_value)


def test_the_shelf_reports_the_spurious_fit_signature():
    # Two independent random walks: a high R-squared with a Durbin-Watson near zero is
    # what Family F exists to make visible, so the flag has to fire on it.
    rng = np.random.default_rng(11)
    x = pd.Series(np.exp(np.cumsum(rng.normal(0.0, 0.02, 1200))))
    y = pd.Series(np.exp(np.cumsum(rng.normal(0.0, 0.02, 1200))))
    fit = fit_log_log(x, y, min_rows=100)
    assert fit["durbin_watson"] < 0.5
    assert fit["residuals_autocorrelated"] is True


def test_a_zero_issuance_asset_is_refused_for_the_right_reason():
    # UNI, LINK, AAVE, COMP, MANA and ALGO: thousands of priced days, no issuance, so
    # stock-to-flow is undefined rather than short of history.
    frame = _frame().assign(issuance_usd=0.0)
    refusals = StockToFlow().estimate(frame, dict(StockToFlow.spec.params))
    assert all(not e.applicable for e in refusals)
    assert "positive issuance" in refusals[-1].reason

    # And where issuance exists but history is short, the reason carries the count.
    short = _frame(days=400)
    reason = StockToFlow().estimate(short, dict(StockToFlow.spec.params))[-1].reason
    assert "prior stock-to-flow observations" in reason and "35" in reason


def test_power_law_refuses_dates_at_or_before_genesis():
    frame = _frame()
    frame["genesis_date"] = frame["metric_date"].iloc[-1]
    estimates = PowerLaw().estimate(frame, dict(PowerLaw.spec.params))
    assert all(not e.applicable for e in estimates)
    assert any("prior priced days" in e.reason for e in estimates)
