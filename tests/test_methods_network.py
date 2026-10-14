import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares

from cryptoval.methods import network
from cryptoval.methods.network import (
    LpplsBubbleFlag,
    MetcalfeEstimated,
    NvmRatio,
    PetersonMetcalfe,
    fit_lppls,
    fit_metcalfe,
)


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


def _network_frame(days: int = 800) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D", tz="UTC")
    addresses = np.linspace(2.0e5, 1.1e6, days)
    price = 50.0 + np.arange(days) * 0.2
    supply = np.full(days, 1.0e8)
    return pd.DataFrame(
        {
            "symbol": "ETH",
            "metric_date": [d.date() for d in dates],
            "active_addresses": addresses,
            "price_usd": price,
            "supply": supply,
            "market_cap_usd": price * supply,
            "tx_count": addresses * 3.0,
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def _applicable(method, frame):
    return [e for e in method.estimate(frame, dict(method.spec.params)) if e.applicable]


def test_peterson_values_the_network_from_a_fitted_scale_times_n_squared():
    frame = _network_frame()
    last = _applicable(PetersonMetcalfe(), frame)[-1]
    n = frame["active_addresses"].iloc[-1]
    assert last.inputs_used["pairs"] == pytest.approx(n * (n - 1) / 2.0)
    assert last.fair_value > 0
    assert last.diagnostics["scale_constant_is_fitted"] is True


def test_peterson_and_nvm_do_not_value_a_row_with_its_own_market_cap():
    """Ruling 1: the scale constant and the Metcalfe line come from PRIOR rows only."""
    base = _network_frame()
    bumped = base.copy()
    bumped.loc[bumped.index[-1], "market_cap_usd"] *= 10.0
    for method in (PetersonMetcalfe(), NvmRatio()):
        before = _applicable(method, base)[-1]
        after = _applicable(method, bumped)[-1]
        assert before.fair_value == after.fair_value
        assert before.diagnostics.get("alpha") == after.diagnostics.get("alpha")
        assert before.inputs_used.get("scale_constant") == after.inputs_used.get("scale_constant")


def test_nvm_ratio_reports_the_deviation_and_is_not_a_fair_value():
    last = _applicable(NvmRatio(), _network_frame())[-1]
    assert last.fair_value is None, "NVM is a deviation series, not a price"
    assert "nvm" in last.diagnostics
    assert NvmRatio.spec.returns_fair_value is False


def test_nvm_is_market_cap_over_the_metcalfe_line_and_refusals_name_their_measurement():
    frame = _network_frame()
    estimates = NvmRatio().estimate(frame, dict(NvmRatio.spec.params))
    last = [e for e in estimates if e.applicable][-1]
    assert last.diagnostics["nvm"] == pytest.approx(
        last.inputs_used["market_cap_usd"] / last.inputs_used["metcalfe_implied_cap"]
    )
    assert "only 50 usable" in estimates[50].reason and "need 180" in estimates[50].reason

    constant = frame.assign(active_addresses=5.0e5)
    flat = NvmRatio().estimate(constant, dict(NvmRatio.spec.params))[400]
    assert not flat.applicable and "no variation" in flat.reason


def test_peterson_refusal_names_the_count_and_the_day_level_cause():
    frame = _network_frame()
    short = PetersonMetcalfe().estimate(frame, dict(PetersonMetcalfe.spec.params))[50]
    assert "only 50 usable" in short.reason and "need 180" in short.reason
    frame.loc[frame.index[500], "active_addresses"] = np.nan
    day = PetersonMetcalfe().estimate(frame, dict(PetersonMetcalfe.spec.params))[500]
    assert not day.applicable and "active_addresses" in day.reason


def _lppls_params(**over):
    return {
        "max_horizon_rows": 180,
        "max_rmse_ratio": 0.02,
        "window_rows": 365,
        "n_restarts": 8,
        "min_window_coverage": 0.9,
        **over,
    }


def _bubble(sign: float = -1.0, noise: float = 0.0) -> pd.Series:
    """A genuine LPPLS path: tc 60 rows after a 365-row window, m=0.5, omega=8."""
    t = np.arange(365, dtype="float64")
    dt = 425.0 - t
    path = 5.0 + sign * 0.4 * dt**0.5 * (1.0 + 0.05 * np.cos(8.0 * np.log(dt)))
    rng = np.random.default_rng(1)
    return pd.Series(path + rng.normal(0, noise, len(t)))


def test_lppls_refuses_when_the_critical_time_hits_its_bound():
    # A straight line has no singularity: any tc the optimiser finds is an artefact.
    straight = pd.Series(np.log(np.linspace(100.0, 200.0, 400)))
    params = _lppls_params(window_rows=400)
    assert fit_lppls(straight, params) is None
    # Which bound, and therefore which finding: beyond the horizon is an extrapolation,
    # at-or-before this row is a singularity the window cannot date. Neither fires.
    status = network._lppls_fit(straight, params)["fit_status"]
    assert status in {"tc_beyond_horizon", "tc_at_or_before_this_row", "poor_fit"}


def test_lppls_detects_a_planted_bubble_so_the_flag_is_not_a_constant_false():
    """The positive control: a flag that never fires is indistinguishable from a broken one."""
    fit = fit_lppls(_bubble(), _lppls_params())
    assert fit is not None
    assert 365 < fit["tc"] < 365 + 180
    assert fit["rmse_ratio"] <= 0.02


def test_lppls_does_not_call_a_falling_path_a_bubble():
    params = _lppls_params()
    assert fit_lppls(_bubble(sign=+1.0), params) is None
    fit = network._lppls_fit(_bubble(sign=+1.0), params)
    # The sign gate, named: b > 0 is an anti-bubble, and the row publishes b either way.
    assert fit["fit_status"] == "wrong_sign" and fit["b"] > 0


def test_lppls_flag_says_no_bubble_rather_than_a_date_when_the_fit_is_poor():
    estimates = LpplsBubbleFlag().estimate(_network_frame(), dict(LpplsBubbleFlag.spec.params))
    applicable = [e for e in estimates if e.applicable]
    assert applicable, "the flag is applicable even when it detects nothing"
    assert applicable[-1].fair_value is None
    assert applicable[-1].diagnostics["bubble_regime"] is False
    assert applicable[-1].diagnostics["critical_time"] is None
    assert applicable[-1].diagnostics["fit_status"] != "bubble_regime"


def test_lppls_optimiser_that_never_converges_is_a_refusal_not_a_no_bubble(monkeypatch):
    """The silent degrade this method must not have: failure looking like a negative."""

    class _Stalled:
        success, status, cost, x = False, 0, 1.0, np.zeros(7)

    monkeypatch.setattr(network, "least_squares", lambda *a, **k: _Stalled())
    estimates = LpplsBubbleFlag().estimate(_network_frame(), dict(LpplsBubbleFlag.spec.params))
    assert not any(e.applicable for e in estimates)
    assert "optimiser_failed" in estimates[-1].reason
    assert "0 of 8 restarts converged" in estimates[-1].reason


def test_lppls_carries_a_fit_forward_between_strides_and_says_how_old_it_is():
    stride = 5
    params = {**LpplsBubbleFlag.spec.params, "stride_rows": stride}
    frame = _network_frame()
    applicable = [e for e in LpplsBubbleFlag().estimate(frame, params) if e.applicable]
    # Never staler than the stride, and the fresh fits land on calendar phase rather than
    # on frame position, so backfilling earlier history cannot move which dates are fresh.
    assert all(0 <= e.diagnostics["fit_age_rows"] < stride for e in applicable)
    fresh = {e.metric_date for e in applicable if e.diagnostics["fit_age_rows"] == 0}
    assert fresh, "some rows must carry a fit of their own"
    assert all(d.toordinal() % stride == 0 for d in sorted(fresh)[1:])


def test_the_stride_phase_does_not_move_when_earlier_history_is_prepended():
    # Plan 4 compares runs across vintages, so a backfill must not change which dates
    # carry a freshly fitted verdict.
    stride = 5
    params = {**LpplsBubbleFlag.spec.params, "stride_rows": stride}
    frame = _network_frame()
    short = frame.iloc[200:].reset_index(drop=True)
    ages_long = {
        e.metric_date: e.diagnostics["fit_age_rows"]
        for e in LpplsBubbleFlag().estimate(frame, params)
        if e.applicable
    }
    ages_short = {
        e.metric_date: e.diagnostics["fit_age_rows"]
        for e in LpplsBubbleFlag().estimate(short, params)
        if e.applicable
    }
    # Each frame fits its own first applicable row, so the two agree only once the
    # shorter one has reached its first calendar phase date: one stride of warm-up.
    shared = sorted(set(ages_long) & set(ages_short))[stride:]
    assert len(shared) > 50
    assert all(ages_long[d] == ages_short[d] for d in shared)


def test_lppls_window_with_holes_is_refused_with_its_coverage():
    frame = _network_frame()
    frame.loc[frame.index[600:700], "price_usd"] = np.nan
    estimates = LpplsBubbleFlag().estimate(frame, dict(LpplsBubbleFlag.spec.params))
    assert "window_gaps" in estimates[750].reason
    assert "prices" in estimates[750].reason


def test_lppls_flag_reports_the_critical_time_from_the_row_being_valued():
    """A planted tc 60 rows past a 365-row window: the first valued row is 60 rows from it."""
    frame = _network_frame(days=366)
    frame["price_usd"] = np.exp(_bubble().to_numpy().tolist() + [5.0])
    row = LpplsBubbleFlag().estimate(frame, dict(LpplsBubbleFlag.spec.params))[-1]
    assert row.applicable and row.diagnostics["bubble_regime"] is True
    assert row.diagnostics["fit_status"] == "bubble_regime"
    assert row.diagnostics["critical_time"] == pytest.approx(60.0, abs=10.0)
    assert row.diagnostics["restarts_converged"] >= 1


def test_the_variable_projection_fit_is_the_brief_s_seven_parameter_model():
    """The riskiest reduction in this plan: four linear parameters solved by lstsq inside
    the residual instead of being free. Same model, or every LPPLS row is wrong."""
    window = 365
    values = _bubble().to_numpy()
    t = np.arange(window, dtype="float64")

    def residual_7(theta):
        tc, m, omega, a, b, c1, c2 = theta
        dt = np.clip(tc - t, 1e-8, None)
        power, phase = dt**m, omega * np.log(dt)
        return a + b * power + power * (c1 * np.cos(phase) + c2 * np.sin(phase)) - values

    rng = np.random.default_rng(0)  # the same seed the module uses
    best = None
    for _ in range(8):
        guess = [window + rng.uniform(1.0, 180.0), rng.uniform(0.1, 0.9), rng.uniform(6.0, 13.0)]
        fit = least_squares(
            residual_7,
            [*guess, 5.0, -0.4, 0.0, 0.0],
            method="trf",
            bounds=(
                [window + 1.0, 0.1, 6.0, -np.inf, -np.inf, -np.inf, -np.inf],
                [window + 180.0, 0.9, 13.0, np.inf, np.inf, np.inf, np.inf],
            ),
            max_nfev=2000,
        )
        if fit.success and (best is None or fit.cost < best.cost):
            best = fit
    assert best is not None
    spread = float(np.ptp(values))
    reference_rmse = float(np.sqrt(2.0 * best.cost / len(values))) / spread

    reduced = network._lppls_fit(_bubble(), _lppls_params())
    assert reduced["fit_status"] == "bubble_regime"
    assert reduced["rmse_ratio"] == pytest.approx(reference_rmse, abs=1e-4)
    assert reduced["tc"] == pytest.approx(float(best.x[0]), abs=1.0)
    assert np.sign(reduced["b"]) == np.sign(float(best.x[4]))


def test_the_lppls_fit_is_deterministic_across_runs():
    # A bubble flag that changes between runs of the same panel is not a measurement.
    params = _lppls_params()
    first = network._lppls_fit(_bubble(noise=0.02), params)
    second = network._lppls_fit(_bubble(noise=0.02), params)
    for key in ("fit_status", "tc", "m", "omega", "b", "rmse_ratio", "restarts_converged"):
        assert first[key] == second[key], key


def test_the_lppls_window_never_contains_the_row_it_values():
    # The standing prior-only ruling, measured for the third method: shocking the valued
    # row cannot change its own verdict, because the window ends before it.
    frame = _network_frame(days=400)
    frame["price_usd"] = np.exp(np.linspace(4.0, 6.0, 400))
    params = {**LpplsBubbleFlag.spec.params, "stride_rows": 1}
    before = LpplsBubbleFlag().estimate(frame, params)[-1]
    shocked = frame.copy()
    shocked.loc[shocked.index[-1], "price_usd"] *= 100.0
    after = LpplsBubbleFlag().estimate(shocked, params)[-1]
    assert before.applicable == after.applicable
    assert after.diagnostics["fit_status"] == before.diagnostics["fit_status"]
    assert after.diagnostics["best_rmse_ratio"] == pytest.approx(
        before.diagnostics["best_rmse_ratio"]
    )


def test_the_critical_time_is_counted_from_the_row_not_from_the_fit():
    """`fitted_at` and `position` are different frames, and a 366-row test cannot tell
    them apart because they coincide there. This one plants tc at an absolute frame row."""
    window, stride = 365, 5
    total = 420
    tc_absolute = 430.0  # a frame row, 10 rows past the frame's end
    t = np.arange(total, dtype="float64")
    dt = np.clip(tc_absolute - t, 1e-8, None)
    path = 5.0 + -0.4 * dt**0.5 * (1.0 + 0.05 * np.cos(8.0 * np.log(dt)))
    frame = _network_frame(days=total)
    frame["price_usd"] = np.exp(path)
    rows = {
        e.metric_date: e
        for e in LpplsBubbleFlag().estimate(
            frame, {**LpplsBubbleFlag.spec.params, "stride_rows": stride}
        )
        if e.applicable and e.diagnostics["bubble_regime"]
    }
    assert rows, "the planted bubble must be detected somewhere in the frame"
    for position in range(window, total):
        row = rows.get(frame["metric_date"].iloc[position])
        if row is None:
            continue
        # critical_time counts rows from THIS row to tc, whatever the fit's own age.
        assert row.diagnostics["critical_time"] == pytest.approx(
            tc_absolute - position, abs=12.0
        ), (position, row.diagnostics["fit_age_rows"])
    assert any(r.diagnostics["fit_age_rows"] > 0 for r in rows.values()), "test a carried row too"
