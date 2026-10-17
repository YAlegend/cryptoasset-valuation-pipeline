import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.econometrics.cointegration import MIN_COINT_ROWS, engle_granger, johansen


def _cointegrated(n=900, coefficient=2.0, seed=11):
    """x is a random walk; y is `coefficient` times x plus STATIONARY noise, so the two
    share one stochastic trend and y - coefficient*x is stationary by construction."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 1.0, n))
    y = coefficient * x + rng.normal(0.0, 1.0, n)
    return pd.Series(y), pd.Series(x)


def _independent_walks(n=900, seed=11):
    rng = np.random.default_rng(seed)
    return (
        pd.Series(np.cumsum(rng.normal(0.0, 1.0, n))),
        pd.Series(np.cumsum(rng.normal(0.0, 1.0, n))),
    )


def test_a_cointegrated_pair_is_found():
    assert engle_granger(*_cointegrated())["p_value"] < 0.01


def test_two_independent_random_walks_are_found_at_the_nominal_rate_only():
    """The other direction, so the test can fail. Two unrelated walks are the null case
    this whole chapter is up against. One draw cannot test that -- a correct 5% test
    rejects one pair in twenty -- so this is the SIZE over many independent pairs."""
    p_values = [
        engle_granger(*_independent_walks(n=400, seed=seed))["p_value"] for seed in range(200)
    ]
    rejection_rate = np.mean(np.array(p_values) < 0.05)
    assert 0.01 <= rejection_rate <= 0.10


def test_the_coefficient_is_estimated_not_assumed():
    """The substantive gain over the ratio form: Metcalfe's exponent becomes testable."""
    result = engle_granger(*_cointegrated(coefficient=2.0))
    assert abs(result["coefficient"] - 2.0) < 0.1
    assert result["coefficient_stderr"] > 0.0
    assert abs(result["coefficient"] - 2.0) < 3.0 * result["coefficient_stderr"]


def test_a_different_true_coefficient_is_recovered():
    """Not vacuous: the estimate tracks the truth rather than returning a constant."""
    assert abs(engle_granger(*_cointegrated(coefficient=0.5))["coefficient"] - 0.5) < 0.1


def test_the_reported_n_is_the_rows_used():
    assert engle_granger(*_cointegrated(n=700))["n"] == 700


def test_a_perfectly_collinear_pair_is_refused_with_its_residual_variance():
    """Review Focus 4: this design exists to avoid testing price against a price-derived
    series, so the guard that catches one must itself be tested. A p-value of 0 here
    would be published as a discovery."""
    x = pd.Series(np.cumsum(np.random.default_rng(3).normal(0.0, 1.0, 800)))
    y = 2.0 * x + 5.0
    with pytest.raises(Unusable) as exc:
        engle_granger(y, x)
    assert "residual variance" in str(exc.value)


def test_mismatched_lengths_are_refused():
    y, x = _cointegrated(n=800)
    with pytest.raises(Unusable):
        engle_granger(y, x.iloc[:700])


def test_too_few_rows_is_refused_with_the_count():
    y, x = _cointegrated(n=MIN_COINT_ROWS - 1)
    with pytest.raises(Unusable) as exc:
        engle_granger(y, x)
    assert str(MIN_COINT_ROWS - 1) in str(exc.value)


def test_the_stderr_survives_serially_correlated_equilibrium_errors():
    """The chapter tests Metcalfe's exponent against this interval, so it must not be the
    plain-OLS one: on a cointegrating regression the residual is autocorrelated and the
    naive standard error is several times too small, which would reject a true 2."""
    import statsmodels.api as sm

    rng = np.random.default_rng(5)
    n = 900
    x = np.cumsum(rng.normal(0.0, 1.0, n))
    error = np.zeros(n)
    for t in range(1, n):  # AR(1), rho 0.95: stationary, but slow to revert
        error[t] = 0.95 * error[t - 1] + rng.normal(0.0, 1.0)
    y = 2.0 * x + error

    naive = sm.OLS(y, sm.add_constant(x)).fit().bse[1]
    result = engle_granger(pd.Series(y), pd.Series(x))
    assert result["coefficient_stderr"] > 2.0 * naive


def test_johansen_finds_one_cointegrating_relationship_in_a_cointegrated_pair():
    result = johansen(*_cointegrated())
    assert result["rank"] == 1
    assert result["trace_statistic"] > result["critical_95"]


# statsmodels casts eigenvalues whose imaginary parts are rounding noise; 200 calls, 200 warnings.
@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
def test_johansen_finds_none_between_independent_walks_at_the_nominal_rate_only():
    """A size test, not one draw. det_order=0's critical values assume the walks DRIFT;
    driftless they reject ~12%, so this runs at drift/sd 0.05, the regime the tabulated
    values are for. The chapter reports the measured size in the other regime."""
    ranks = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        y, x = (pd.Series(np.cumsum(rng.normal(0.05, 1.0, 1000))) for _ in range(2))
        ranks.append(johansen(y, x)["rank"])
    assert 0.01 <= np.mean(ranks) <= 0.10


def test_johansen_refuses_too_few_rows_with_the_count():
    y, x = _cointegrated(n=MIN_COINT_ROWS - 1)
    with pytest.raises(Unusable) as exc:
        johansen(y, x)
    assert str(MIN_COINT_ROWS - 1) in str(exc.value)
