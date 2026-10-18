import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.volatility.returns import MIN_RETURNS, crypto_returns, pair_returns, price_series
from cryptoval.volatility.settings import load_vol_config


def _panel(symbol, start, days, gap_at=None, leading_nulls=0):
    dates = pd.date_range(start, periods=days, freq="D")
    price = np.exp(np.cumsum(np.full(days, 0.001))) * 100.0
    frame = pd.DataFrame({"symbol": symbol, "metric_date": dates, "price_usd": price})
    if leading_nulls:
        frame.loc[: leading_nulls - 1, "price_usd"] = np.nan
    if gap_at is not None:
        frame.loc[gap_at, "price_usd"] = np.nan
    return frame


def test_returns_are_percent_log_returns():
    returns = crypto_returns(price_series(_panel("BTC", "2019-01-01", 800), "BTC"), "BTC")
    assert len(returns) == 799
    assert returns.iloc[0] == pytest.approx(0.1)


def test_leading_nulls_are_not_a_gap():
    returns = crypto_returns(
        price_series(_panel("ETH", "2015-01-01", 1000, leading_nulls=200), "ETH"), "ETH"
    )
    assert len(returns) == 799


def test_an_interior_gap_is_refused_with_its_date():
    prices = price_series(_panel("BTC", "2019-01-01", 1000, gap_at=500), "BTC")
    with pytest.raises(Unusable, match="2-day step"):
        crypto_returns(prices, "BTC")


def test_a_short_history_is_refused_with_its_count():
    prices = price_series(_panel("SOL", "2025-10-14", 363), "SOL")
    with pytest.raises(Unusable, match=f"362 returns, {MIN_RETURNS} needed"):
        crypto_returns(prices, "SOL")


def test_an_absent_symbol_is_refused():
    with pytest.raises(Unusable, match="XRP is not in the panel"):
        price_series(_panel("BTC", "2019-01-01", 10), "XRP")


def test_trading_calendar_returns_span_the_weekend():
    crypto = price_series(_panel("BTC", "2019-01-01", 1400), "BTC")
    days = pd.bdate_range("2019-01-01", "2022-10-31")
    equity = pd.Series(np.linspace(100.0, 200.0, len(days)), index=days)
    out = pair_returns(crypto, equity, "BTC|AAPL", trading_calendar=True)
    friday, monday = pd.Timestamp("2019-01-04"), pd.Timestamp("2019-01-07")
    assert out.loc[monday, "a"] == pytest.approx(100.0 * np.log(crypto[monday] / crypto[friday]))


def test_trading_calendar_uses_only_the_overlap():
    crypto = price_series(_panel("AAVE", "2020-10-01", 1500), "AAVE")
    days = pd.bdate_range("2015-01-01", "2026-10-16")
    equity = pd.Series(100.0 + np.arange(len(days)), index=days, dtype="float64")
    out = pair_returns(crypto, equity, "AAVE|AAPL", trading_calendar=True)
    assert out.index.min() > pd.Timestamp("2020-10-01")


def test_trading_calendar_refuses_a_missing_crypto_price():
    crypto = price_series(_panel("BTC", "2019-01-01", 1400, gap_at=700), "BTC")
    days = pd.bdate_range("2019-01-01", "2022-10-31")
    equity = pd.Series(100.0, index=days)
    with pytest.raises(Unusable, match="no crypto price on 1 trading date"):
        pair_returns(crypto, equity, "BTC|AAPL", trading_calendar=True)


def test_daily_pairs_use_the_intersection():
    btc = price_series(_panel("BTC", "2019-01-01", 1400), "BTC")
    eth = price_series(_panel("ETH", "2020-01-01", 1000), "ETH")
    out = pair_returns(btc, eth, "BTC|ETH", trading_calendar=False)
    assert list(out.columns) == ["a", "b"]
    assert len(out) == 999


def test_the_shipped_config_names_the_dissertation_windows():
    config = load_vol_config()
    assert config.windows["peak_oct21_nov21"] == (dt.date(2021, 10, 1), dt.date(2021, 11, 30))
    assert config.peak_window == "peak_oct21_nov21"
    assert len(config.windows) == 8


def test_a_window_that_ends_before_it_starts_is_refused(tmp_path):
    path = tmp_path / "v.yml"
    path.write_text(
        "windows: {w: [2022-01-02, 2022-01-01]}\npeak_window: w\n"
        "crypto_anchors: [BTC]\ntraditional_anchors: [BTC]\n"
    )
    with pytest.raises(ValueError, match="before start"):
        load_vol_config(str(path))


def test_a_duplicated_date_is_refused_on_either_calendar():
    """Final review: on the trading calendar nothing else would catch a duplicate, and a
    repeated day becomes a silent zero return. 2026-10-18's iCloud conflict copies
    duplicated every panel row, so this input is not hypothetical."""
    days = pd.bdate_range("2019-01-01", periods=900)
    frame = pd.DataFrame({"symbol": "AAPL", "metric_date": days, "price_usd": 100.0})
    doubled = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
    with pytest.raises(Unusable, match="AAPL has 1 duplicated date"):
        price_series(doubled, "AAPL")
