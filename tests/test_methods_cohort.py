import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.cohort import CohortZScore
from cryptoval.methods.registry import run_methods

PARAMS = {"min_cohort_size": 3, "ratio": "market_cap_to_revenue", "window_rows": 365}


def _panel(days: int = 400) -> pd.DataFrame:
    """Three DeFi assets plus one lone oracle asset, with enough history for the
    trailing denominator to exist (the window needs half of 365 rows)."""
    dates = [d.date() for d in pd.date_range("2025-01-01", periods=days, freq="D")]
    rows = []
    for symbol, sector, cap, revenue in (
        ("AAVE", "defi", 1.5e9, 2.0e5),
        ("UNI", "defi", 4.8e9, 3.0e5),
        ("CRV", "defi", 5.0e8, 1.0e5),
        ("LINK", "oracle", 1.2e10, 5.0e4),
    ):
        rows.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "metric_date": dates,
                    "sector": sector,
                    "price_usd": np.full(days, 10.0),
                    "supply": np.full(days, 1.0e8),
                    "market_cap_usd": np.full(days, cap),
                    "net_revenue_usd": np.full(days, revenue),
                    "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _latest(estimates):
    newest = max(e.metric_date for e in estimates)
    return {e.symbol: e for e in estimates if e.metric_date == newest}


def test_cohort_zscore_scores_within_the_sector():
    by_symbol = _latest(CohortZScore().estimate(_panel(), PARAMS))
    defi = [by_symbol[s] for s in ("AAVE", "UNI", "CRV")]
    assert all(e.applicable for e in defi)
    assert all(e.inputs_used["cohort"] == "defi" for e in defi)
    assert all(e.inputs_used["cohort_size"] == 3 for e in defi)
    # A z-score set has mean zero by construction: that is the check that it is a z-score.
    assert sum(e.diagnostics["z_score"] for e in defi) == pytest.approx(0.0, abs=1e-9)
    assert all(e.fair_value is None for e in defi), "a z-score is not a price"


def test_cohort_of_one_is_refused_with_its_measured_size():
    link = _latest(CohortZScore().estimate(_panel(), PARAMS))["LINK"]
    assert not link.applicable
    assert "1" in link.reason and "3" in link.reason


def test_null_sector_is_its_own_refusal_not_a_cohort():
    panel = _panel().assign(sector=None)
    estimates = CohortZScore().estimate(panel, PARAMS)
    assert all(not e.applicable for e in estimates)
    assert "sector" in estimates[0].reason


def test_the_denominator_is_a_trailing_mean_not_one_day_of_revenue():
    """One day's revenue moved an asset's own z-score by more than a full sigma on 5% of
    live days. The ratio must be market cap over ANNUALISED trailing revenue."""
    panel = _panel()
    # A single enormous revenue day for UNI: a daily denominator would swing its ratio by
    # orders of magnitude, a 365-day mean absorbs it.
    spike = (panel["symbol"] == "UNI") & (panel["metric_date"] == panel["metric_date"].max())
    panel.loc[spike, "net_revenue_usd"] = 1.0e9

    uni = _latest(CohortZScore().estimate(panel, PARAMS))["UNI"]
    clean = _latest(CohortZScore().estimate(_panel(), PARAMS))["UNI"]
    assert uni.inputs_used["window_rows"] == 365

    # The spike is 3,333x the daily baseline. A one-day denominator would move the ratio by
    # that whole factor; a 365-day mean dilutes it to roughly 1/365 of that.
    spike_factor = 1.0e9 / 3.0e5
    moved = clean.inputs_used["ratio"] / uni.inputs_used["ratio"]
    assert moved < spike_factor / 100, f"a trailing mean should absorb it, moved {moved:.0f}x"
    # And the denominator really is annualised: 3.0e5/day over a year is ~1.1e8.
    assert clean.inputs_used["ratio"] == pytest.approx(4.8e9 / (365 * 3.0e5), rel=0.01)


def test_a_cohort_with_no_dispersion_is_refused_rather_than_divided_by_zero():
    panel = _panel()
    # Identical ratios across the cohort, so the standard deviation is exactly zero.
    defi = panel["sector"] == "defi"
    panel.loc[defi, "market_cap_usd"] = 1.0e9
    panel.loc[defi, "net_revenue_usd"] = 2.0e5
    estimates = [e for e in CohortZScore().estimate(panel, PARAMS) if e.symbol == "AAVE"]
    assert all(not e.applicable for e in estimates)
    assert "dispersion" in estimates[-1].reason


def test_run_methods_dispatches_a_cross_sectional_method_on_the_whole_panel():
    panel = _panel(days=400)
    rows = run_methods(panel, {"sector_cohort_zscore": CohortZScore()}, ["sector_cohort_zscore"])
    # Every panel row gets a row back -- and NOT four refusals from a per-symbol loop that
    # handed the method a one-row cohort.
    assert len(rows) == len(panel)
    assert rows["applicable"].sum() > 0
    newest = rows[rows["metric_date"] == rows["metric_date"].max()]
    assert int(newest["applicable"].sum()) == 3


def test_a_cross_sectional_method_that_raises_becomes_one_refusal_not_a_lost_run():
    class Boom(CohortZScore):
        def estimate(self, frame, params):
            raise RuntimeError("cohort exploded")

    rows = run_methods(_panel(), {"sector_cohort_zscore": Boom()}, ["sector_cohort_zscore"])
    assert len(rows) == 1
    assert not rows["applicable"].iloc[0]
    assert "method raised RuntimeError" in rows["reason"].iloc[0]


def test_a_saturated_cohort_says_so_rather_than_passing_off_one_bit_as_a_ranking():
    """A z-score over n members cannot exceed sqrt(n-1). When one member reaches that
    ceiling the rest are compressed into a point and the score cannot rank them."""
    panel = _panel()
    # One member orders of magnitude more expensive than the other two.
    panel.loc[panel["symbol"] == "UNI", "net_revenue_usd"] = 1.0e2
    by_symbol = _latest(CohortZScore().estimate(panel, PARAMS))
    assert by_symbol["UNI"].diagnostics["cohort_saturated"] is True
    assert by_symbol["UNI"].diagnostics["z_score"] == pytest.approx(2**0.5, rel=0.01)

    # The ordinary case is not flagged.
    clean = _latest(CohortZScore().estimate(_panel(), PARAMS))
    assert clean["AAVE"].diagnostics["cohort_saturated"] is False
