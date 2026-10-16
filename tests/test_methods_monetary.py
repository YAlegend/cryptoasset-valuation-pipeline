import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.monetary import (
    HoldingTimeCorrection,
    JCurve,
    KampakisLogLinearEoe,
    fit_log_linear_eoe,
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


def _eoe_frame(days: int = 600) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")
    price = 100.0 * np.exp(np.linspace(0.0, 0.8, days))
    supply = np.full(days, 1.0e8)
    market_cap = price * supply
    # Volume must NOT be a constant fraction of market cap, or log(T) and log(M) are
    # collinear too and the design collapses to rank 2. On the live panel the turnover
    # ratio varies day to day, which leaves rank 3 of 4 -- the real degeneracy.
    volume = market_cap * 0.04 * np.exp(np.random.default_rng(11).normal(0.0, 0.3, days))
    return pd.DataFrame(
        {
            "symbol": "ETH",
            "metric_date": [d.date() for d in dates],
            "price_usd": price,
            "supply": supply,
            "market_cap_usd": market_cap,
            "volume_total_usd": volume,
            # Exactly as the mart defines it: volume / market cap. That identity is what
            # makes the three published coefficients unidentifiable.
            "velocity_total": volume / market_cap,
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_fit_log_linear_eoe_returns_three_betas_each_with_an_interval():
    fit = fit_log_linear_eoe(_eoe_frame(), min_rows=180)
    assert fit is not None
    for name in ("beta_t", "beta_m", "beta_inv_v"):
        assert name in fit
        assert fit[f"{name}_ci_low"] <= fit[name] <= fit[f"{name}_ci_high"]
    assert "adj_r_squared" in fit and "durbin_watson" in fit


def test_fit_reports_that_its_coefficients_are_not_identified():
    """velocity_total IS volume/market cap, so log(1/V) == log(M) - log(T) exactly and the
    design matrix is rank 3 of 4. statsmodels returns a minimum-norm pseudo-inverse, which
    looks like an estimate. The fit has to say that it is not one."""
    fit = fit_log_linear_eoe(_eoe_frame(), min_rows=180)
    assert fit["coefficients_identified"] is False
    assert fit["design_rank"] == 3
    assert fit["design_columns"] == 4
    assert fit["condition_number"] > 1e10
    # The pseudo-inverse signature. Minimising the norm subject to the two identified
    # combinations forces beta_M == beta_T + beta_invV exactly, and drives all three to
    # (1/3, 2/3, 1/3) whenever gamma_T is zero -- which is what the live ETH fit returns.
    assert fit["beta_m"] == pytest.approx(fit["beta_t"] + fit["beta_inv_v"], rel=1e-6)
    assert fit["beta_t"] == pytest.approx(1 / 3, abs=0.02)
    assert fit["beta_m"] == pytest.approx(2 / 3, abs=0.02)


def test_the_identified_reparameterisation_recovers_the_definition_of_price():
    """Only gamma_T = beta_T - beta_invV and gamma_M = beta_M + beta_invV are identified,
    and they recover price = market cap / supply with volume contributing nothing."""
    fit = fit_log_linear_eoe(_eoe_frame(), min_rows=180)
    assert fit["gamma_t"] == pytest.approx(0.0, abs=0.01), "volume carries no information"
    assert fit["gamma_m"] == pytest.approx(1.0, abs=0.01)
    # The intercept is -log(supply): price = M / supply.
    assert fit["alpha"] == pytest.approx(-np.log(1.0e8), rel=0.01)
    # An accounting identity fits better than the published 0.97 ever could.
    assert fit["adj_r_squared"] > 0.99


def test_kampakis_declares_its_own_circularity_because_the_registry_cannot_see_it():
    from cryptoval.methods.base import circularity

    # velocity_total is caught automatically...
    assert "velocity_total" in circularity(KampakisLogLinearEoe.spec)
    last = [
        e
        for e in KampakisLogLinearEoe().estimate(
            _eoe_frame(), dict(KampakisLogLinearEoe.spec.params)
        )
        if e.applicable
    ][-1]
    # ...but market cap as a REGRESSOR against price is not, so the method says it itself.
    assert last.diagnostics["structurally_circular"] is True
    assert "price" in last.diagnostics["circularity_note"]


def test_kampakis_reports_its_own_betas_not_the_published_ones():
    last = [
        e
        for e in KampakisLogLinearEoe().estimate(
            _eoe_frame(), dict(KampakisLogLinearEoe.spec.params)
        )
        if e.applicable
    ][-1]
    published = (0.88, 0.84, 1.15)
    estimated = (
        last.diagnostics["beta_t"],
        last.diagnostics["beta_m"],
        last.diagnostics["beta_inv_v"],
    )
    assert estimated != published, "the coefficients are re-estimated, never adopted"
    assert last.diagnostics["published_beta"] == list(published)


def test_kampakis_predicts_the_price_back_because_the_model_is_an_identity():
    """The prediction IS identified even though the coefficients are not. What it predicts
    is the price itself, which is the finding, not a success."""
    frame = _eoe_frame()
    last = [
        e
        for e in KampakisLogLinearEoe().estimate(frame, dict(KampakisLogLinearEoe.spec.params))
        if e.applicable
    ][-1]
    actual = frame.loc[frame["metric_date"] == last.metric_date, "price_usd"].iloc[0]
    assert last.fair_value == pytest.approx(actual, rel=0.05)
