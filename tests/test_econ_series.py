import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import PairSpec, Unusable
from cryptoval.econometrics.series import LEVEL_SERIES, MIN_PAIR_ROWS, build_levels, pair_rows

MVRV = PairSpec(
    pair_id="mvrv_anchor",
    dependent="log_market_cap",
    independent="log_realized_cap",
    forced_coefficient=1.0,
)


def _panel(days=1200, symbols=("BTC",), **overrides):
    out = []
    for symbol in symbols:
        dates = pd.date_range("2019-01-01", periods=days, freq="D")
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "metric_date": dates,
                "price_usd": np.linspace(100.0, 300.0, days),
                "market_cap_usd": np.linspace(1e9, 3e9, days),
                "realized_cap": np.linspace(5e8, 2e9, days),
                "volume_total_usd": np.linspace(1e7, 5e7, days),
                "active_addresses": np.linspace(1e4, 5e4, days),
                "panel_vintage_ts": pd.Timestamp("2026-10-12 08:30", tz="UTC"),
            }
        )
        for column, value in overrides.items():
            frame[column] = value
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def test_build_levels_logs_every_declared_series():
    levels = build_levels(_panel())
    assert set(LEVEL_SERIES) <= set(levels.columns)
    panel = _panel()
    assert np.allclose(levels["log_market_cap"], np.log(panel["market_cap_usd"]))


def test_a_non_positive_value_becomes_nan_rather_than_minus_infinity():
    """Review Focus 1: active_addresses = 0 on a dead day, volume = 0 on an illiquid one."""
    panel = _panel()
    panel.loc[panel.index[5], "active_addresses"] = 0.0
    panel.loc[panel.index[6], "volume_total_usd"] = -1.0
    levels = build_levels(panel)
    assert levels["log_addresses"].isna().iloc[5]
    assert levels["log_volume"].isna().iloc[6]
    # The point of the guard: NaN is a missing measurement, -inf is a number a regression
    # will happily consume.
    assert not np.isneginf(levels[list(LEVEL_SERIES)].to_numpy()).any()


def test_pair_rows_uses_the_intersection_and_reports_its_length():
    """Review Focus 3: realized_cap starts later than price for most assets."""
    panel = _panel(days=1200)
    panel.loc[panel.index[:300], "realized_cap"] = np.nan
    rows = pair_rows(build_levels(panel), "BTC", MVRV)
    assert len(rows) == 900
    assert rows["metric_date"].min() == panel["metric_date"].iloc[300]
    assert rows[["y", "x"]].notna().all().all()


def test_the_ratio_is_the_dependent_minus_the_forced_multiple_of_the_independent():
    rows = pair_rows(build_levels(_panel()), "BTC", MVRV)
    assert np.allclose(rows["ratio"], rows["y"] - 1.0 * rows["x"])

    metcalfe = PairSpec(
        pair_id="metcalfe_anchor",
        dependent="log_market_cap",
        independent="log_addresses",
        forced_coefficient=2.0,
    )
    squared = pair_rows(build_levels(_panel()), "BTC", metcalfe)
    assert np.allclose(squared["ratio"], squared["y"] - 2.0 * squared["x"])


def test_an_interior_gap_in_the_intersection_is_refused_with_its_count():
    """Review Focus 2: DOT carries 653 MVRV rows inside 2,605 panel rows."""
    panel = _panel(days=1200)
    panel.loc[panel.index[600:610], "realized_cap"] = np.nan
    with pytest.raises(Unusable) as exc:
        pair_rows(build_levels(panel), "BTC", MVRV)
    assert "10 missing day(s)" in str(exc.value)


def test_a_pair_below_the_row_floor_is_refused_with_the_count():
    panel = _panel(days=MIN_PAIR_ROWS - 1)
    with pytest.raises(Unusable) as exc:
        pair_rows(build_levels(panel), "BTC", MVRV)
    assert str(MIN_PAIR_ROWS - 1) in str(exc.value)
    assert str(MIN_PAIR_ROWS) in str(exc.value)


def test_a_symbol_absent_from_the_panel_is_refused_not_empty():
    with pytest.raises(Unusable):
        pair_rows(build_levels(_panel()), "ETH", MVRV)
