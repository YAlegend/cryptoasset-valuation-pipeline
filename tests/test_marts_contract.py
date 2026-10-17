"""The mart's select list must agree with the constant that writes gold.

The Plan 4 whole-branch review found nothing tied SCORE_COLUMNS to fct_model_score.sql, so
a column added to the constant would reach gold and never reach the mart, with no test
failing. This is that test, for both marts.
"""

import pathlib
import re

from cryptoval.econometrics.runner import TEST_COLUMNS
from cryptoval.scoring.runner import SCORE_COLUMNS


def _selected_columns(sql_path: str) -> list[str]:
    text = pathlib.Path(sql_path).read_text(encoding="utf-8")
    body = text[text.lower().index("select") : text.lower().rindex("from")]
    out = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--") or stripped.lower() == "select":
            continue
        match = re.fullmatch(r"([a-z_][a-z0-9_]*),?", stripped)
        if match:
            out.append(match.group(1))
    return out


def test_the_score_mart_selects_exactly_score_columns():
    assert _selected_columns("dbt/models/marts/fct_model_score.sql") == SCORE_COLUMNS


def test_the_econometrics_mart_selects_exactly_test_columns():
    assert _selected_columns("dbt/models/marts/fct_econometric_test.sql") == TEST_COLUMNS
