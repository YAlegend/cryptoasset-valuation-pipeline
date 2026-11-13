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
Plans 4b and 4c can re-correct over the union of families -- from the RAW ``dm_p_value``
rows, which is all they need from this plan's run. Handing the corrected table back to
``correct_family_wise`` raises: a second pass over adjusted p-values is not a wider
family, it is a double correction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cryptoval.scoring.runner import SCORE_COLUMNS

ALPHA = 0.05


def correct_family_wise(
    scores: pd.DataFrame,
    alpha: float = ALPHA,
    *,
    p_metric: str = "dm_p_value",
    columns: list[str] | None = None,
    count_keys: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Append Holm-adjusted p-values and the hypothesis count to ``scores``.

    A refused test (NaN value, reason set) is NOT a hypothesis: counting it would inflate
    the correction and make the surviving claims harder to reject than the evidence
    warrants, which is a different error from the one 11.6 guards against but an error all
    the same.

    The keyword arguments exist so Plan 4b's econometrics table shares this
    implementation -- and with it the refusal of a second pass -- rather than carrying a
    second copy of Holm. Their defaults are Plan 4's behaviour exactly.
    """
    columns = list(columns if columns is not None else SCORE_COLUMNS)
    holm_metric = f"{p_metric}_holm"
    keys = dict(
        count_keys
        if count_keys is not None
        else {"model_id": "*", "symbol": "*", "horizon": -1, "fold": -1}
    )

    n_holm = int((scores["metric"] == holm_metric).sum())
    n_count = int((scores["metric"] == "hypotheses_tested").sum())
    if n_holm or n_count:
        raise ValueError(
            f"input already has {n_holm} {holm_metric} and {n_count} hypotheses_tested rows; "
            f"the correction was already applied, so re-correct from the raw {p_metric} rows"
        )

    tests = scores[(scores["metric"] == p_metric) & scores["value"].notna()].copy()
    n = int(len(tests))

    count_row = dict.fromkeys(columns)
    count_row.update(keys)
    count_row.update(
        metric="hypotheses_tested",
        value=float(n),
        reason=f"Holm-Bonferroni family-wise correction at alpha={alpha}",
    )

    if n == 0:
        return pd.concat([scores, pd.DataFrame([count_row])], ignore_index=True)[columns]

    _, adjusted, _, _ = multipletests(
        tests["value"].to_numpy(dtype="float64"), alpha=alpha, method="holm"
    )
    corrected = tests.copy()
    corrected["metric"] = holm_metric
    corrected["value"] = np.asarray(adjusted, dtype="float64")
    corrected["reason"] = f"Holm over {n} hypotheses"

    return pd.concat([scores, corrected, pd.DataFrame([count_row])], ignore_index=True)[columns]
