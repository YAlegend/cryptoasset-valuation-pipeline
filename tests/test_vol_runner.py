import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.volatility import runner
from cryptoval.volatility.runner import TEST_COLUMNS, run_volatility
from cryptoval.volatility.settings import VolConfig

CONFIG = VolConfig(
    windows={
        "early": (dt.date(2019, 6, 1), dt.date(2019, 12, 31)),
        "peak": (dt.date(2020, 3, 1), dt.date(2020, 4, 30)),
        "before": (dt.date(2010, 1, 1), dt.date(2010, 12, 31)),
    },
    peak_window="peak",
    crypto_anchors=("BTC",),
    traditional_anchors=("BTC",),
    markov_search_reps=5,
)


def _garch_prices(n, seed):
    rng = np.random.default_rng(seed)
    out, variance = np.empty(n), 1.0
    for t in range(n):
        out[t] = np.sqrt(variance) * rng.standard_normal()
        variance = 0.1 + 0.1 * out[t] ** 2 + 0.85 * variance
    return 100.0 * np.exp(np.cumsum(out) / 100.0)


def _crypto(symbols=("BTC", "ETH", "USDC"), days=1200, start="2019-01-01"):
    frames = []
    for i, symbol in enumerate(symbols):
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "metric_date": pd.date_range(start, periods=days, freq="D"),
                    "price_usd": _garch_prices(days, seed=i),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _traditional():
    days = pd.bdate_range("2018-01-01", "2023-06-30")
    return pd.DataFrame(
        {"symbol": "AAPL", "metric_date": days, "price_usd": _garch_prices(len(days), seed=9)}
    )


@pytest.fixture(scope="module")
def table():
    return run_volatility(_crypto(), _traditional(), CONFIG, frozenset({"USDC"}))


def test_columns_and_the_hypothesis_count(table):
    assert list(table.columns) == TEST_COLUMNS
    count = table[table["metric"] == "hypotheses_tested"]["value"].item()
    assert count == ((table["metric"] == "p_value") & table["value"].notna()).sum()
    # BTC and ETH: arch_lm + egarch + exhaustion each; BTC|ETH and BTC|AAPL: engle-sheppard.
    assert count == 8
    assert (table["metric"] == "p_value_holm").sum() == 8


def test_a_stablecoin_is_refused_by_name(table):
    usdc = table[table["symbol"] == "USDC"]
    assert usdc["metric"].tolist() == ["refused"]
    assert "stablecoin" in usdc["reason"].item()


def test_both_pairs_are_fitted_with_windows_and_a_peak_rank(table):
    for pair in ("BTC|ETH", "BTC|AAPL"):
        rows = table[(table["symbol"] == pair) & (table["model_id"] == "dcc")]
        assert {"a", "b", "p_value"} <= set(rows["metric"])
        peak = rows[rows["metric"] == "peak_percentile"]["value"].item()
        assert 0.0 <= peak <= 1.0
        early = rows[(rows["metric"] == "pearson_correlation") & (rows["window_name"] == "early")]
        assert early["value"].notna().item()


def test_a_window_outside_the_history_is_a_nan_row_with_its_count(table):
    rows = table[(table["window_name"] == "before") & (table["symbol"] == "BTC")]
    assert rows["value"].isna().all()
    assert rows["reason"].str.contains("0 rows in window before").all()


def test_markov_rows_carry_each_window(table):
    rows = table[(table["model_id"] == "markov") & (table["symbol"] == "BTC")]
    assert set(rows["window_name"].dropna()) == {"early", "peak", "before"}


def test_a_garch_that_fails_to_converge_is_a_refusal_and_the_run_continues(monkeypatch):
    from cryptoval.econometrics.base import Unusable

    real = runner.fit_garch

    def flaky(returns):
        if len(returns) == 1199 and abs(returns.iloc[0] - _first_eth_return()) < 1e-9:
            raise Unusable("garch: optimizer did not converge (flag 4: test)")
        return real(returns)

    monkeypatch.setattr(runner, "fit_garch", flaky)
    out = run_volatility(_crypto(("BTC", "ETH")), _traditional(), CONFIG)
    eth = out[(out["symbol"] == "ETH")]
    assert eth[eth["model_id"] == "garch"]["metric"].tolist() == ["refused"]
    assert "needs the garch fit" in eth[eth["model_id"] == "exhaustion"]["reason"].item()
    assert (out[(out["symbol"] == "BTC") & (out["model_id"] == "garch")]["metric"] == "alpha").any()


def _first_eth_return():
    crypto = _crypto(("BTC", "ETH"))
    prices = crypto[crypto["symbol"] == "ETH"]["price_usd"].to_numpy()
    return 100.0 * np.log(prices[1] / prices[0])


def test_a_missing_anchor_refuses_its_pairs_by_name():
    out = run_volatility(_crypto(("ETH",)), _traditional(), CONFIG)
    refused = out[(out["model_id"] == "dcc") & (out["metric"] == "refused")]
    assert set(refused["symbol"]) == {"BTC|ETH", "BTC|AAPL"}
    assert refused["reason"].str.contains("BTC is not in the panel").all()
