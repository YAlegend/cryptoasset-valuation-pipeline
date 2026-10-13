import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.monetary import (
    HoldingTimeCorrection,
    JCurve,
    fit_lognormal_velocity,
    holding_time_ratio,
)
from cryptoval.methods.registry import METHODS


def _frame(n: int = 500, sigma: float = 0.6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    velocity = np.exp(rng.normal(np.log(0.04), sigma, n))
    market_cap = np.full(n, 1.0e12)
    return pd.DataFrame(
        {
            "symbol": ["ETH"] * n,
            "metric_date": pd.date_range("2025-01-01", periods=n, freq="D").date,
            "market_cap_usd": market_cap,
            "volume_total_usd": velocity * market_cap,
            "velocity_total": velocity,
            "supply": np.full(n, 1.2e8),
            "price_usd": market_cap / 1.2e8,
            "tx_count": np.full(n, 1.2e6),
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )


def test_the_two_holding_times_differ_by_exp_sigma_squared():
    """Spec 10.2's central result, stated exactly rather than simulated."""
    fit = fit_lognormal_velocity(_frame(n=20000, sigma=0.6)["velocity_total"])
    assert fit["sigma"] == pytest.approx(0.6, rel=0.05)
    assert fit["mean_holding_time"] / fit["naive_inverse_velocity"] == pytest.approx(
        np.exp(fit["sigma"] ** 2), rel=1e-9
    )
    assert holding_time_ratio(0.6) == pytest.approx(np.exp(0.36))


def test_the_closed_form_matches_monte_carlo():
    """Simulation used as verification, which is what it is good for."""
    rng = np.random.default_rng(11)
    mu, sigma, n = np.log(0.04), 0.5, 2_000_000
    draws = np.exp(rng.normal(mu, sigma, n))
    simulated_mean_holding = float(np.mean(1.0 / draws))
    closed_form = float(np.exp(-mu + sigma**2 / 2))
    assert simulated_mean_holding == pytest.approx(closed_form, rel=0.01)


def test_a_zero_sigma_collapses_the_correction_to_one():
    # With no dispersion the inversion is exact, which is the degenerate case the
    # dissertation implicitly assumed.
    assert holding_time_ratio(0.0) == pytest.approx(1.0)


def test_the_correction_returns_no_fair_value_but_records_the_diagnosis():
    out = HoldingTimeCorrection().estimate(_frame(), {"window_rows": 365})
    last = out[-1]
    assert last.fair_value is None
    assert last.applicable  # applicable, but it is a correction, not a valuation
    assert last.diagnostics["holding_time_ratio"] > 1.0
    assert "anderson_darling_p" in last.diagnostics


def test_a_degenerate_velocity_series_is_refused_rather_than_fitted():
    frame = _frame(n=400)
    frame["velocity_total"] = 0.0
    out = HoldingTimeCorrection().estimate(frame, {"window_rows": 365})
    assert not out[-1].applicable
    assert "positive" in out[-1].reason


def test_qtm_with_panel_velocity_returns_the_price_back():
    """Review Focus 1, demonstrated numerically: the expression collapses."""
    frame = _frame(n=400)
    method = METHODS["crypto_qtm_panel_velocity"]
    out = method.estimate(frame, {"circulating_fraction": 1.0, "velocity_source": "panel"})
    applicable = [e for e in out if e.applicable]
    prices = frame["price_usd"].to_numpy()
    values = np.array([e.fair_value for e in applicable])
    # fair_value == price / f exactly, so every residual is zero by construction.
    assert values == pytest.approx(prices[: len(values)], rel=1e-9)
    assert applicable[0].diagnostics["collapses_to_price"] is True


def test_qtm_with_a_parameter_velocity_is_a_genuine_sensitivity():
    frame = _frame(n=400)
    method = METHODS["crypto_qtm"]
    slow = method.estimate(
        frame, {"circulating_fraction": 1.0, "velocity_source": "parameter", "velocity_daily": 0.02}
    )
    fast = method.estimate(
        frame, {"circulating_fraction": 1.0, "velocity_source": "parameter", "velocity_daily": 0.08}
    )
    # Halving assumed velocity doubles the implied price; this is the one usable reading
    # of QTM on free data (the on-chain transfer value it really wants is not free).
    assert slow[-1].fair_value == pytest.approx(4.0 * fast[-1].fair_value, rel=1e-9)
    assert slow[-1].diagnostics["collapses_to_price"] is False


def test_the_j_curve_splits_price_into_utility_and_expectation():
    out = JCurve().estimate(_frame(n=400), {"velocity_source": "parameter", "velocity_daily": 0.04})
    last = out[-1]
    # CUV is the fair value; DEUV is the residual the market pays above it.
    assert last.fair_value is not None
    assert last.diagnostics["deuv"] == pytest.approx(
        last.inputs_used["price_usd"] - last.fair_value
    )
    assert -1.0 <= last.diagnostics["deuv_share"] <= 1.0
