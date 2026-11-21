"""The dissertation's 6.3 claim, as a hypothesis (design 5).

"Elevated volatility bespeaks an uncertain and unpredictable market in which the
direction of price resists forecasting; subdued volatility bespeaks a steadier market
whose price may be expected to continue upon its established course."

NON-OVERLAPPING blocks: overlapping windows share days, so their outcomes are
mechanically correlated and a count test over them is anti-conservative. Each pair of
consecutive blocks is one observation: did block k+1's summed return keep block k's
sign, classified by the GARCH conditional volatility on block k's last day, split at the
median. Fisher's exact test, two-sided; the two continuation rates give the direction.

The conditional volatility uses full-sample GARCH parameters (design 2), and the "profit
metrics" half of 6.3 is not tested: those are UTXO series deferred with the BigQuery work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from cryptoval.econometrics.base import Unusable

MIN_BLOCK_PAIRS = 20


def exhaustion_test(
    returns: pd.Series, cond_vol: pd.Series, block_days: int = 30
) -> dict[str, float]:
    if not cond_vol.index.equals(returns.index):
        raise Unusable("exhaustion: the conditional volatility is not on the returns' dates")
    blocks = len(returns) // block_days
    if blocks - 1 < MIN_BLOCK_PAIRS:
        raise Unusable(
            f"exhaustion: {max(blocks - 1, 0)} block pairs, {MIN_BLOCK_PAIRS} needed "
            f"({len(returns)} returns in {block_days}-day blocks)"
        )
    used = blocks * block_days
    sums = returns.to_numpy(dtype="float64")[:used].reshape(blocks, block_days).sum(axis=1)
    vol = cond_vol.to_numpy(dtype="float64")[:used].reshape(blocks, block_days)[:, -1]

    signs = np.sign(sums)
    continued = signs[1:] == signs[:-1]
    high = vol[:-1] > np.median(vol[:-1])
    table = [
        [int((high & continued).sum()), int((high & ~continued).sum())],
        [int((~high & continued).sum()), int((~high & ~continued).sum())],
    ]
    _, p_value = fisher_exact(table)
    return {
        "p_value": float(p_value),
        "continuation_high_vol": float(continued[high].mean()) if high.any() else float("nan"),
        "continuation_low_vol": float(continued[~high].mean()) if (~high).any() else float("nan"),
        "block_pairs": float(blocks - 1),
    }
