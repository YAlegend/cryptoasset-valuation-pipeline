import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.econometrics.cointegration import MIN_COINT_ROWS
from cryptoval.econometrics.vecm import speed_of_adjustment


def _mean_reverting(n=900, speed=0.1, seed=5):
    """y tracks x with an error-correction term pulling it back at `speed` per day, so
    the fitted alpha has a known sign and rough magnitude."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 1.0, n))
    y = np.zeros(n)
    for t in range(1, n):
        gap = y[t - 1] - x[t - 1]
        # Gap noise as large as the trend's: at sd 0.1 the gap is too small for alpha's
        # standard error (~0.15) to resolve a speed of 0.1, and the sign came back wrong.
        y[t] = y[t - 1] + (x[t] - x[t - 1]) - speed * gap + rng.normal(0.0, 1.0)
    return pd.Series(y), pd.Series(x)


def test_the_adjustment_term_is_negative_for_a_mean_reverting_pair():
    """A NEGATIVE alpha is what reversion means: a gap above equilibrium pulls down."""
    result = speed_of_adjustment(*_mean_reverting(speed=0.1))
    assert result["alpha"] < 0.0
    assert abs(result["alpha"] + 0.1) < 3.0 * result["alpha_stderr"]


def test_a_faster_pull_gives_a_larger_adjustment_and_a_shorter_half_life():
    """Not vacuous: the estimate tracks the simulated speed rather than its own sign."""
    slow = speed_of_adjustment(*_mean_reverting(speed=0.02))
    fast = speed_of_adjustment(*_mean_reverting(speed=0.30))
    assert abs(fast["alpha"]) > abs(slow["alpha"])
    assert fast["half_life_days"] < slow["half_life_days"]


def test_the_half_life_is_none_when_the_pull_is_the_wrong_sign():
    """A positive alpha is divergence, not reversion, and has no half life. Reporting a
    number there would read as a reversion speed."""
    # A NEGATIVE speed in the reverting fixture is a push: every gap is widened by 2% a day.
    result = speed_of_adjustment(*_mean_reverting(n=300, speed=-0.02))
    assert result["alpha"] > 0.0  # the fixture really diverges, so the next line can fail
    assert np.isnan(result["half_life_days"])


def test_the_reported_n_and_lags_are_what_was_used():
    result = speed_of_adjustment(*_mean_reverting(n=700), lags=2)
    assert result["n"] == 700
    assert result["lags"] == 2


def test_too_few_rows_is_refused_with_the_count():
    y, x = _mean_reverting(n=MIN_COINT_ROWS - 1)
    with pytest.raises(Unusable) as exc:
        speed_of_adjustment(y, x)
    assert str(MIN_COINT_ROWS - 1) in str(exc.value)


def test_an_overshooting_pull_still_has_a_half_life_under_a_day():
    """Alpha in (-2, -1) over-corrects, flipping the gap's sign each day while its SIZE
    shrinks by |1 + alpha|. That is reversion, faster than a day, not a missing number."""
    result = speed_of_adjustment(*_mean_reverting(speed=1.5))
    assert -2.0 < result["alpha"] < -1.0  # the fixture really overshoots
    assert 0.0 <= result["half_life_days"] < 1.0


def _fundamental_adjusts(n=900, speed=0.1, seed=6):
    """The mirror of _mean_reverting: market cap is a pure walk and the FUNDAMENTAL closes
    the gap, so market cap's own alpha is ~0 while the gap still reverts at `speed`."""
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0.0, 1.0, n))
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + (y[t] - y[t - 1]) + speed * (y[t - 1] - x[t - 1]) + rng.normal(0.0, 1.0)
    return pd.Series(y), pd.Series(x)


def test_the_half_life_is_the_gaps_not_market_caps_own():
    """Review C1: a gap the fundamental closes must not read as 'no reversion'. The half
    life comes from the gap's speed, beta'alpha, which counts both series' adjustment."""
    result = speed_of_adjustment(*_fundamental_adjusts(speed=0.1))
    assert abs(result["alpha"]) < 3.0 * result["alpha_stderr"]  # market cap does not adjust
    assert result["alpha_independent"] > 3.0 * result["alpha_independent_stderr"]
    assert abs(result["gap_speed"] + 0.1) < 0.05
    assert result["half_life_days"] < 10.0  # log(0.5)/log(0.9) is 6.6 days


def test_the_half_life_is_log_half_over_log_the_gaps_daily_shrink():
    """Pins the formula, not just its range: a -log(2)/alpha approximation passes a range."""
    for fixture in (_mean_reverting(speed=0.3), _mean_reverting(speed=1.5)):
        result = speed_of_adjustment(*fixture)
        expected = np.log(0.5) / np.log(abs(1.0 + result["gap_speed"]))
        assert result["half_life_days"] == pytest.approx(expected)
