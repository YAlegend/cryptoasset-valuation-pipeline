"""Multiple-testing discipline (spec 11.6).

"Without this, the breadth of the method registry becomes a data-snooping machine."

Seven non-baseline models against 22 foldable assets at 5 horizons is up to 770
Diebold-Mariano tests. At alpha = 0.05 roughly 38 of them are expected to come back
significant with no skill present at all, so an uncorrected "beat persistence" table is a
list of coin flips with model names attached.

HOLM-BONFERRONI, not Benjamini-Hochberg. Spec 11.6 asks for a FAMILY-WISE correction,
which bounds the probability of making ANY false claim, and the headline question here is
existential: does ANYTHING beat a random walk. FDR bounds the expected share of false
claims among the rejections, which answers a different question and would let a handful
of spurious winners through by design.

The family is this plan's own hypotheses. The count is written to the table as a row so
Plans 4b and 4c can re-correct over the union without re-running anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cryptoval.scoring.runner import SCORE_COLUMNS

ALPHA = 0.05


def correct_family_wise(scores: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Append Holm-adjusted p-values and the hypothesis count to ``scores``.

    A refused DM test (NaN value, reason set) is NOT a hypothesis: counting it would
    inflate the correction and make the surviving claims harder to reject than the
    evidence warrants, which is a different error from the one 11.6 guards against but an
    error all the same.
    """
    tests = scores[(scores["metric"] == "dm_p_value") & scores["value"].notna()].copy()
    n = int(len(tests))

    count_row = dict.fromkeys(SCORE_COLUMNS)
    count_row.update(
        model_id="*",
        symbol="*",
        horizon=-1,
        fold=-1,
        metric="hypotheses_tested",
        value=float(n),
        reason=f"Holm-Bonferroni family-wise correction at alpha={alpha}",
    )

    if n == 0:
        return pd.concat([scores, pd.DataFrame([count_row])], ignore_index=True)[SCORE_COLUMNS]

    _, adjusted, _, _ = multipletests(
        tests["value"].to_numpy(dtype="float64"), alpha=alpha, method="holm"
    )
    corrected = tests.copy()
    corrected["metric"] = "dm_p_value_holm"
    corrected["value"] = np.asarray(adjusted, dtype="float64")
    corrected["reason"] = f"Holm over {n} hypotheses"

    return pd.concat(
        [scores, corrected, pd.DataFrame([count_row])], ignore_index=True
    )[SCORE_COLUMNS]
