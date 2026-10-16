import numpy as np
import pandas as pd
import pytest

from cryptoval.scoring.targets import (
    FEATURE_LAG_DAYS,
    HORIZONS,
    build_frame,
    forward_log_return,
    signal_matrix,
)


def _price_frame(days: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": dates,
            "price_usd": 100.0 * np.exp(np.linspace(0.0, 1.0, days)),
        }
    )


def _estimates(price: pd.DataFrame, method_id: str, ratio) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": price["symbol"],
            "metric_date": price["metric_date"],
            "method_id": method_id,
            "fair_value": price["price_usd"] * ratio,
            "applicable": True,
        }
    )


def test_the_target_is_a_forward_log_return_not_a_level():
    price = pd.Series([100.0, 110.0, 121.0, 133.1])
    y = forward_log_return(price, horizon=1)
    assert y.iloc[0] == pytest.approx(np.log(1.1))
    # The last h rows have no future price, so they are NaN rather than silently reused.
    assert np.isnan(y.iloc[-1])


def test_the_target_at_horizon_h_spans_h_days():
    price = pd.Series(100.0 * np.exp(np.arange(10) * 0.01))
    assert forward_log_return(price, horizon=3).iloc[0] == pytest.approx(0.03)


def test_horizons_are_the_five_the_spec_fixes():
    assert HORIZONS == (1, 7, 30, 90, 365)


def test_every_feature_is_lagged_at_least_one_day():
    """Review Focus 1: an inclusive-window signal must not see the day it predicts."""
    price = _price_frame()
    # A varying ratio: a constant one is dropped for zero variance by the test below.
    estimates = _estimates(price, "mayer_multiple", np.linspace(1.2, 1.8, len(price)))
    frame = build_frame(price, estimates, horizon=1)
    signals = signal_matrix(estimates, price).set_index("metric_date")["mayer_multiple"]
    merged = frame.set_index("metric_date")["mayer_multiple"].dropna()
    assert FEATURE_LAG_DAYS >= 1
    for date, value in merged.items():
        assert value == pytest.approx(signals.loc[date - pd.Timedelta(days=FEATURE_LAG_DAYS)])


def test_a_zero_variance_signal_is_dropped_with_its_measured_variance():
    """Review Focus 1: crypto_qtm_panel_velocity returns the price, so its signal is 0.0."""
    price = _price_frame()
    estimates = pd.concat(
        [
            _estimates(price, "crypto_qtm_panel_velocity", 1.0),  # fair_value == price
            _estimates(price, "metcalfe_estimated", np.linspace(0.5, 2.0, len(price))),
        ],
        ignore_index=True,
    )
    frame = build_frame(price, estimates, horizon=1)
    assert "crypto_qtm_panel_velocity" not in frame.columns
    assert "metcalfe_estimated" in frame.columns
    assert frame.attrs["dropped_features"]["crypto_qtm_panel_velocity"].startswith("variance ")


def test_an_inapplicable_estimate_contributes_no_signal():
    price = _price_frame()
    estimates = _estimates(price, "nvt_reversion", np.linspace(1.2, 1.8, len(price)))
    estimates.loc[estimates.index[:200], "applicable"] = False
    estimates.loc[estimates.index[:200], "fair_value"] = None
    frame = build_frame(price, estimates, horizon=1)
    assert frame["nvt_reversion"].isna().sum() >= 200


def test_a_non_positive_fair_value_yields_no_signal_rather_than_a_nan_log():
    price = _price_frame(60)
    estimates = _estimates(price, "thermocap_floor", 1.0)
    estimates.loc[estimates.index[:10], "fair_value"] = -5.0
    signals = signal_matrix(estimates, price)
    assert signals["thermocap_floor"].iloc[:10].isna().all()
    assert signals["thermocap_floor"].iloc[10:].notna().any()


def _two_symbol_panel(days: int = 400) -> pd.DataFrame:
    eth = _price_frame(days).assign(symbol="ETH", price_usd=lambda d: d["price_usd"] * 0.05)
    return pd.concat([_price_frame(days), eth], ignore_index=True)


def test_a_contiguous_two_symbol_panel_passes_the_gap_guard():
    """The symbol boundary must not read as a gap: BTC's last day is not ETH's first."""
    panel = _two_symbol_panel()
    estimates = _estimates(panel, "mayer_multiple", np.linspace(1.2, 1.8, len(panel)))
    frame = build_frame(panel, estimates, horizon=1)
    assert set(frame["symbol"]) == {"BTC", "ETH"}


def test_a_missing_day_in_one_symbol_raises_with_the_symbol_and_the_gap_size():
    panel = _two_symbol_panel()
    gap_row = (panel["symbol"] == "ETH") & (panel["metric_date"] == pd.Timestamp("2024-02-10"))
    panel = panel[~gap_row]
    estimates = _estimates(panel, "mayer_multiple", np.linspace(1.2, 1.8, len(panel)))
    with pytest.raises(ValueError, match="ETH") as err:
        build_frame(panel, estimates, horizon=1)
    assert "2-day step" in str(err.value)
