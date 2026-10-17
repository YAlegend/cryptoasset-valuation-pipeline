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
