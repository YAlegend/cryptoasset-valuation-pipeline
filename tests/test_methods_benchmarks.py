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
    assert fit["beta_ci_low"] < fit["beta"] < fit["beta_ci_high"]
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
