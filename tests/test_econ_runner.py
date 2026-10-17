import dataclasses

import numpy as np
import pandas as pd

from cryptoval.econometrics.base import PairSpec
from cryptoval.econometrics.runner import MAX_VAR_LAGS, TEST_COLUMNS, run_pairs
from cryptoval.econometrics.series import build_levels

SPECS = {
    "mvrv_anchor": PairSpec(
        pair_id="mvrv_anchor",
        dependent="log_market_cap",
        independent="log_realized_cap",
        forced_coefficient=1.0,
        params={"lag_criterion": "aic"},
    )
}


def _panel(days=1200, symbols=("BTC",), seed=4):
    rng = np.random.default_rng(seed)
    out = []
    for symbol in symbols:
        realized = np.cumsum(rng.normal(0.002, 0.01, days)) + 20.0
        out.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "metric_date": pd.date_range("2019-01-01", periods=days, freq="D"),
                    "market_cap_usd": np.exp(realized + rng.normal(0.0, 0.05, days)),
                    "realized_cap": np.exp(realized),
                    "volume_total_usd": np.exp(realized - 3.0),
                    "active_addresses": np.exp(realized / 2.0),
                    "panel_vintage_ts": pd.Timestamp("2026-10-12 08:30", tz="UTC"),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def test_the_runner_emits_the_documented_columns_and_grain():
    rows = run_pairs(build_levels(_panel()), SPECS)
    assert list(rows.columns) == TEST_COLUMNS
    assert not rows.duplicated(subset=["pair_id", "symbol", "metric"]).any()


def test_every_statistic_the_chain_produces_reaches_the_table():
    rows = run_pairs(build_levels(_panel()), SPECS)
    metrics = set(rows["metric"])
    assert {
        "adf_statistic",
        "kpss_p_value",
        "independent_adf_p_value",
        "ratio_adf_p_value",
        "cointegration_p_value",
        "cointegration_coefficient",
        "johansen_rank",
        "vecm_alpha",
    } <= metrics


def test_the_holm_row_counts_only_the_cointegration_tests():
    """Global Constraint: ADF/KPSS are pre-tests and VECM coefficients are estimates.
    Counting them would inflate the correction."""
    rows = run_pairs(build_levels(_panel(symbols=("BTC", "ETH"))), SPECS)
    count = rows[rows["metric"] == "hypotheses_tested"]["value"].iloc[0]
    assert count == 2.0  # one pair, two symbols
    assert (rows["metric"] == "cointegration_p_value_holm").sum() == 2


def test_a_refused_pair_is_a_row_with_the_measurement_not_a_missing_asset():
    panel = _panel(days=1200, symbols=("BTC", "SHORT"))
    panel = panel[(panel["symbol"] != "SHORT") | (panel.groupby("symbol").cumcount() < 100)]
    rows = run_pairs(build_levels(panel), SPECS)
    refused = rows[(rows["symbol"] == "SHORT") & (rows["metric"] == "refused")]
    assert len(refused) == 1
    assert refused["value"].isna().all()
    assert "100 rows" in refused["reason"].iloc[0]


def test_every_row_carries_the_vintage_and_the_params_hash():
    vintage = pd.Timestamp("2026-10-12 08:30", tz="UTC")
    rows = run_pairs(build_levels(_panel()), SPECS, panel_vintage_ts=vintage)
    assert (rows["panel_vintage_ts"] == vintage).all()  # the hypotheses_tested row too
    assert rows["params_hash"].notna().all()
    assert rows[rows["pair_id"] == "mvrv_anchor"]["params_hash"].nunique() == 1


def test_the_params_hash_tells_two_pairs_with_identical_params_apart():
    """The three shipped pairs carry the SAME params; a hash of params alone would make
    an edited pair indistinguishable from its predecessor in the mart."""
    other = dataclasses.replace(SPECS["mvrv_anchor"], pair_id="other", independent="log_volume")
    rows = run_pairs(build_levels(_panel()), {**SPECS, "other": other})
    hashes = rows[rows["pair_id"] != "*"].groupby("pair_id")["params_hash"].first()
    assert hashes["mvrv_anchor"] != hashes["other"]


def test_the_vecm_is_fitted_only_where_cointegration_survived_holm():
    """Design 7: a VECM on a pair with no equilibrium estimates the speed of return to a
    relationship that does not exist, so the unrelated asset gets a refusal row instead."""
    panel = _panel(symbols=("BTC", "NOISE"))
    walk = np.cumsum(np.random.default_rng(10).normal(0.002, 0.03, 1200)) + 21.0
    panel.loc[panel["symbol"] == "NOISE", "market_cap_usd"] = np.exp(walk)
    rows = run_pairs(build_levels(panel), SPECS)

    def metric(symbol, name):
        return rows[(rows["symbol"] == symbol) & (rows["metric"] == name)]

    assert metric("BTC", "cointegration_p_value_holm")["value"].iloc[0] < 0.05
    assert metric("NOISE", "cointegration_p_value_holm")["value"].iloc[0] >= 0.05
    assert metric("BTC", "vecm_alpha")["value"].notna().all()
    noise_alpha = metric("NOISE", "vecm_alpha")
    assert len(noise_alpha) == 1 and noise_alpha["value"].isna().all()
    assert "not fitted" in noise_alpha["reason"].iloc[0]
    assert metric("NOISE", "vecm_half_life_days").empty


def test_the_var_lag_order_is_chosen_by_the_pairs_criterion_and_recorded():
    """Design 9: the lag order comes from an information criterion, with the choice
    recorded -- not a hard-coded 1. The fixture's differences carry lag-3 dynamics."""
    from statsmodels.tsa.vector_ar.vecm import select_order

    rng = np.random.default_rng(2)
    n = 1500
    dx = np.zeros(n)
    for t in range(3, n):
        dx[t] = 0.6 * dx[t - 3] + rng.normal(0.0, 0.01)
    realized = np.cumsum(dx) + 20.0
    panel = _panel(days=n)
    panel["realized_cap"] = np.exp(realized)
    panel["market_cap_usd"] = np.exp(realized + rng.normal(0.0, 0.05, n))
    levels = build_levels(panel)
    rows = run_pairs(levels, SPECS)

    expected = select_order(
        levels[["log_market_cap", "log_realized_cap"]].to_numpy(),
        maxlags=MAX_VAR_LAGS,
        deterministic="ci",
    ).aic
    assert expected > 1  # the fixture really needs more than one lag
    for metric in ("johansen_trace_statistic", "vecm_alpha"):
        assert rows[rows["metric"] == metric]["lags"].iloc[0] == expected
