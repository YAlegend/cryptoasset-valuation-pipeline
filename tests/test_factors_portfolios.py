import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.factors.inference import fama_macbeth, nw_alpha, nw_mean
from cryptoval.factors.portfolios import decile_returns, ltw_factors
from cryptoval.factors.settings import FactorConfig

CONFIG = FactorConfig(exclude={}, min_assets=50)


def _week(n=50, week="2020-01-05", seed=0):
    rng = np.random.default_rng(seed)
    signal = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "week": pd.Timestamp(week),
            "asset": [f"a{i:02d}" for i in range(n)],
            "market_cap_usd": rng.uniform(1e6, 1e9, n),
            "ret": signal / 100.0,
            "size": signal,
            "momentum": signal,
            "value": signal,
            "mvrv": signal,
        }
    )


def test_deciles_are_equal_count_value_weighted_and_hl_is_d10_minus_d1():
    week = _week()
    out = decile_returns(week, "value", CONFIG).iloc[0]
    top = week.tail(5)
    expected = (top["ret"] * top["market_cap_usd"]).sum() / top["market_cap_usd"].sum()
    assert np.isclose(out["D10"], expected)
    assert np.isclose(out["HL"], out["D10"] - out["D1"])
    assert out["n_assets"] == 50


def test_a_week_below_min_assets_is_skipped():
    assert decile_returns(_week(n=49), "value", CONFIG).empty


def test_ties_are_broken_by_asset_id():
    week = _week().assign(mvrv=1.0)
    a = decile_returns(week, "mvrv", CONFIG)
    b = decile_returns(week.sample(frac=1.0, random_state=3), "mvrv", CONFIG)
    pd.testing.assert_frame_equal(a, b)
    assert a["HL"].iloc[0] > 0  # ties fall back to id order, and ret rises with id here


def test_ltw_signs():
    week = _week()
    out = ltw_factors(week, CONFIG).iloc[0]
    assert out["CMOM"] > 0  # high momentum earned more
    assert out["CSMB"] < 0  # small minus big; small earned less here
    w = week["market_cap_usd"]
    assert np.isclose(out["CMKT"], (week["ret"] * w).sum() / w.sum())


def test_nw_mean_detects_a_premium_and_not_noise():
    noise = pd.Series(np.random.default_rng(0).normal(0, 0.1, 500))
    assert nw_mean(noise, 4)["p_value"] > 0.05
    assert nw_mean(noise + 0.05, 4)["p_value"] < 1e-6
    with pytest.raises(Unusable, match="51 weeks"):
        nw_mean(noise.iloc[:51], 4)


def test_nw_alpha_recovers_alpha_and_beta():
    rng = np.random.default_rng(1)
    factors = pd.DataFrame(rng.normal(0, 0.1, (500, 3)), columns=["CMKT", "CSMB", "CMOM"])
    portfolio = 0.02 + 1.5 * factors["CMKT"] + rng.normal(0, 0.01, 500)
    fit = nw_alpha(portfolio, factors, 4)
    assert fit["alpha"] == pytest.approx(0.02, abs=0.003)
    assert fit["beta_cmkt"] == pytest.approx(1.5, abs=0.02)
    assert fit["beta_csmb"] == pytest.approx(0.0, abs=0.02)
    assert fit["p_value"] < 1e-6


def test_fama_macbeth_slope_is_the_rank_spread():
    week = _week()
    week[["size", "momentum", "mvrv"]] = np.random.default_rng(2).normal(size=(50, 3))
    # ret = i / 100 and value's rank x = i / 49, so ret = 0.49 x: an exact fit
    slopes = fama_macbeth(
        week,
        ("size", "momentum", "value", "mvrv"),
        FactorConfig(exclude={}, min_assets=50, winsor_quantile=0.0),
    )
    assert slopes["value"].iloc[0] == pytest.approx(0.49, abs=1e-9)
    assert slopes["size"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_fama_macbeth_winsorizes_an_outlier():
    week = _week().assign(ret=0.0)
    week.loc[0, "ret"] = 9.0  # one +900% micro-cap at the bottom of the value rank
    assert abs(fama_macbeth(week, ("value",), CONFIG)["value"].iloc[0]) < 1e-12
    unclipped = FactorConfig(exclude={}, min_assets=50, winsor_quantile=0.0)
    assert fama_macbeth(week, ("value",), unclipped)["value"].iloc[0] < -0.1


def test_fama_macbeth_skips_thin_weeks():
    assert fama_macbeth(_week(n=49), ("value",), CONFIG).empty
