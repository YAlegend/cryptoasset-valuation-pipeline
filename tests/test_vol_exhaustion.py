import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.volatility.exhaustion import exhaustion_test


def _series(blocks=200, block_days=30, low_vol_continues=True, seed=0):
    """Low-volatility blocks are followed by a block of the same sign (when
    low_vol_continues); high-volatility blocks by a coin flip."""
    rng = np.random.default_rng(seed)
    high = rng.random(blocks) < 0.5
    signs = np.empty(blocks)
    signs[0] = 1.0
    for k in range(1, blocks):
        keep = low_vol_continues and not high[k - 1]
        signs[k] = signs[k - 1] if keep else rng.choice([-1.0, 1.0])
    days = blocks * block_days
    returns = np.repeat(signs * 0.5, block_days) + rng.normal(0.0, 0.1, days)
    vol = np.repeat(np.where(high, 5.0, 1.0), block_days)
    index = pd.date_range("2015-01-01", periods=days, freq="D")
    return pd.Series(returns, index=index), pd.Series(vol, index=index)


def test_it_finds_continuation_that_depends_on_volatility():
    result = exhaustion_test(*_series())
    assert result["p_value"] < 1e-6
    assert result["continuation_low_vol"] == pytest.approx(1.0)
    assert result["continuation_high_vol"] == pytest.approx(0.5, abs=0.15)
    assert result["block_pairs"] == 199


def test_it_finds_nothing_when_continuation_is_a_coin_flip():
    assert exhaustion_test(*_series(low_vol_continues=False))["p_value"] > 0.05


def test_too_few_blocks_are_refused_with_the_count():
    returns, vol = _series(blocks=15)
    with pytest.raises(Unusable, match="14 block pairs, 20 needed"):
        exhaustion_test(returns, vol)


def test_misaligned_volatility_is_refused():
    returns, vol = _series()
    with pytest.raises(Unusable, match="not on the returns' dates"):
        exhaustion_test(returns, vol.iloc[1:])
