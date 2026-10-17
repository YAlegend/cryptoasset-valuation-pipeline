# tests/test_scoring_multiplicity.py
import numpy as np
import pandas as pd
import pytest

from cryptoval.scoring.multiplicity import ALPHA, correct_family_wise
from cryptoval.scoring.runner import SCORE_COLUMNS


def _p_rows(p_values, model_ids=None, symbols=None, horizons=None) -> pd.DataFrame:
    n = len(p_values)
    model_ids = model_ids or [f"m{i}" for i in range(n)]
    symbols = symbols or ["BTC"] * n
    horizons = horizons or [1] * n
    rows = []
    for p, model_id, symbol, horizon in zip(p_values, model_ids, symbols, horizons, strict=True):
        row = dict.fromkeys(SCORE_COLUMNS)
        row.update(
            model_id=model_id,
            symbol=symbol,
            horizon=horizon,
            fold=-1,
            metric="dm_p_value",
            value=p,
            reason="",
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def test_the_hypothesis_count_is_recorded_as_a_row():
    rows = _p_rows([0.01, 0.2, 0.5])
    rmse = rows.iloc[[0]].copy()
    rmse["metric"] = "rmse"
    rmse["value"] = 2.5
    out = correct_family_wise(pd.concat([rows, rmse], ignore_index=True))
    count = out[out["metric"] == "hypotheses_tested"]
    assert len(count) == 1
    assert count["value"].iloc[0] == 3.0


def test_holm_raises_every_p_value_and_never_lowers_one():
    raw = [0.001, 0.02, 0.3, 0.7]
    out = correct_family_wise(_p_rows(raw))
    adjusted = out[out["metric"] == "dm_p_value_holm"].sort_values("model_id")["value"].to_numpy()
    assert (adjusted >= np.array(raw)).all()
    # fdr_bh on this same input gives [0.004, 0.04, 0.4, 0.7]. Monotonicity holds for both, so
    # these exact Holm values are what separate the family-wise correction 11.6 requires from FDR.
    assert adjusted == pytest.approx([0.004, 0.06, 0.6, 0.7])


def test_holm_matches_its_definition_on_a_worked_example():
    # Sorted p = 0.01, 0.04; n = 2. Holm: p1 * 2 = 0.02, then max(0.02, 0.04 * 1) = 0.04.
    out = correct_family_wise(_p_rows([0.01, 0.04], model_ids=["a", "b"]))
    adjusted = out[out["metric"] == "dm_p_value_holm"].set_index("model_id")["value"]
    assert adjusted["a"] == pytest.approx(0.02)
    assert adjusted["b"] == pytest.approx(0.04)


def test_a_result_significant_alone_can_stop_being_significant_in_company():
    """The whole point of 11.6: breadth turns a registry into a data-snooping machine."""
    alone = correct_family_wise(_p_rows([0.03]))
    assert alone[alone["metric"] == "dm_p_value_holm"]["value"].iloc[0] < ALPHA

    crowd = correct_family_wise(_p_rows([0.03] + [0.6] * 99))
    survivors = crowd[(crowd["metric"] == "dm_p_value_holm") & (crowd["value"] < ALPHA)]
    assert survivors.empty


def test_a_refused_dm_test_is_not_counted_as_a_hypothesis():
    rows = _p_rows([0.01, 0.02])
    rows.loc[1, "value"] = np.nan
    rows.loc[1, "reason"] = "only 4 paired observations, minimum 10"
    out = correct_family_wise(rows)
    assert out[out["metric"] == "hypotheses_tested"]["value"].iloc[0] == 1.0
    assert len(out[out["metric"] == "dm_p_value_holm"]) == 1


def test_the_original_rows_are_preserved_alongside_the_corrected_ones():
    rows = _p_rows([0.01, 0.2])
    out = correct_family_wise(rows)
    assert len(out[out["metric"] == "dm_p_value"]) == 2
    assert list(out.columns) == SCORE_COLUMNS


def test_holm_follows_each_p_value_to_its_own_model_when_input_is_shuffled():
    # Expected values are NOT in input order: b is the smallest p, so it takes the smallest
    # adjusted value. A sorted-input bug lands the adjusted values on the wrong models here.
    out = correct_family_wise(_p_rows([0.7, 0.001, 0.3, 0.02], model_ids=["a", "b", "c", "d"]))
    adjusted = out[out["metric"] == "dm_p_value_holm"].set_index("model_id")["value"]
    assert list(adjusted[["a", "b", "c", "d"]]) == pytest.approx([0.7, 0.004, 0.6, 0.06])


def test_a_second_correction_refuses_rather_than_duplicating_rows():
    once = correct_family_wise(_p_rows([0.01, 0.2]))
    with pytest.raises(ValueError, match="2 dm_p_value_holm and 1 hypotheses_tested"):
        correct_family_wise(once)
