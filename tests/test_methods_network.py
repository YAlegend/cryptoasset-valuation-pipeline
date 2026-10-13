import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.network import MetcalfeEstimated, fit_metcalfe


def _frame(n: int = 800, beta: float = 1.69, noise: float = 0.15, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    addresses = np.exp(np.linspace(np.log(2e5), np.log(1.1e6), n))
    log_cap = 4.0 + beta * np.log(addresses) + rng.normal(0, noise, n)
    return pd.DataFrame(
        {
            "symbol": ["ETH"] * n,
            "metric_date": pd.date_range("2024-01-01", periods=n, freq="D").date,
            "market_cap_usd": np.exp(log_cap),
            "active_addresses": addresses,
            "supply": np.full(n, 1.2e8),
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )


def test_beta_is_recovered_with_a_confidence_interval():
    frame = _frame(beta=1.69)
    fit = fit_metcalfe(frame["market_cap_usd"], frame["active_addresses"])
    assert fit["beta"] == pytest.approx(1.69, rel=0.05)
    assert fit["beta_ci_low"] < fit["beta"] < fit["beta_ci_high"]
    # Never assumed: the theoretical 2.0 must not be inside a correctly recovered 1.69 CI
    # on this much data, which is the whole point of estimating it.
    assert not (fit["beta_ci_low"] <= 2.0 <= fit["beta_ci_high"])


def test_the_fit_reports_durbin_watson_and_r_squared():
    """Review Focus 2: R-squared alone cannot distinguish a real relationship from two
    trending series, so the autocorrelation statistic travels with it."""
    fit = fit_metcalfe(_frame()["market_cap_usd"], _frame()["active_addresses"])
    assert 0.0 <= fit["r_squared"] <= 1.0
    assert 0.0 <= fit["durbin_watson"] <= 4.0
    assert fit["n"] == 800


def test_two_unrelated_trending_series_get_a_high_r_squared_and_a_damning_dw():
    """The failure mode itself, demonstrated: both series merely grow."""
    n = 800
    t = np.linspace(0.0, 1.0, n)
    addresses = np.exp(np.log(1e5) + t * (np.log(1e6) - np.log(1e5)))
    # Both series merely grow, and the second's growth bends independently of the first.
    # The bend matters: two EXACT exponentials would fit each other perfectly in log-log,
    # leaving only floating-point noise as residual and making DW meaningless.
    # Amplitude measured rather than guessed -- DW is ~0.0001 at every amplitude from 0.05
    # to 0.30, so the bend's only real trade-off is R-squared: 0.9997 at 0.05, 0.9954 at
    # 0.20, 0.9895 at 0.30. 0.20 keeps both halves of the demonstration with margin.
    unrelated_cap = np.exp(np.log(1e9) + t * (np.log(1e12) - np.log(1e9))) * (
        1.0 + 0.20 * np.sin(3.0 * np.pi * t)
    )
    fit = fit_metcalfe(pd.Series(unrelated_cap), pd.Series(addresses))
    assert fit["r_squared"] > 0.99
    # Strong positive autocorrelation: DW far below 2 says the residuals are not
    # independent, so the R-squared is not evidence.
    assert fit["durbin_watson"] < 0.5
    assert fit["residuals_autocorrelated"] is True


def test_the_estimate_uses_only_trailing_data():
    """No row may be fitted on information from its own future (spec 11.2)."""
    frame = _frame(n=600)
    out = MetcalfeEstimated().estimate(frame, {"min_fit_rows": 180, "refit_every_rows": 30})
    applicable = [e for e in out if e.applicable]
    # The first 180 rows cannot be fitted, so they are refusals, not estimates.
    assert len(applicable) == 600 - 180
    assert all(e.diagnostics["fit_rows"] <= i + 180 for i, e in enumerate(applicable))


def test_the_band_comes_from_the_beta_confidence_interval():
    out = MetcalfeEstimated().estimate(_frame(n=400), {"min_fit_rows": 180, "refit_every_rows": 30})
    last = [e for e in out if e.applicable][-1]
    assert last.band_low < last.fair_value < last.band_high
    assert last.diagnostics["beta"] > 1.0


def test_a_flat_address_series_is_refused_rather_than_fitted():
    frame = _frame(n=400)
    frame["active_addresses"] = 500000.0
    out = MetcalfeEstimated().estimate(frame, {"min_fit_rows": 180, "refit_every_rows": 30})
    assert not any(e.applicable for e in out)
    assert "no variation" in out[-1].reason
