# tests/test_budget.py
import datetime as dt

import pytest

from cryptoval.budget import BudgetExceeded, ByteBudgetGate
from cryptoval.manifest import RunManifest
from cryptoval.paths import PathResolver

NOW = dt.datetime(2026, 10, 11, tzinfo=dt.UTC)
BUDGET = 1_000


def _gate(tmp_path, scanned: list[int] | None = None):
    m = RunManifest(PathResolver(str(tmp_path)))
    for b in scanned or []:
        rec = m.start("bigquery", {}, started_at=NOW)
        m.finish(rec, rows_in=0, rows_out=1, status="success", bytes_scanned=b)
    return ByteBudgetGate(m, monthly_budget=BUDGET)


def test_full_budget_remains_when_nothing_scanned(tmp_path):
    assert _gate(tmp_path).remaining(NOW) == BUDGET


def test_remaining_subtracts_this_month_only(tmp_path):
    assert _gate(tmp_path, [300, 200]).remaining(NOW) == 500


def test_guard_allows_a_query_inside_budget(tmp_path):
    _gate(tmp_path, [300]).guard(500, NOW)  # must not raise


def test_guard_allows_a_query_exactly_at_budget(tmp_path):
    _gate(tmp_path, [300]).guard(700, NOW)


def test_guard_refuses_a_query_over_budget(tmp_path):
    with pytest.raises(BudgetExceeded, match="would scan"):
        _gate(tmp_path, [300]).guard(800, NOW)


def test_refusal_message_reports_estimate_and_remaining(tmp_path):
    with pytest.raises(BudgetExceeded) as exc:
        _gate(tmp_path, [900]).guard(500, NOW)
    msg = str(exc.value)
    assert "500" in msg and "100" in msg


def test_guard_refuses_a_single_query_larger_than_the_whole_budget(tmp_path):
    with pytest.raises(BudgetExceeded):
        _gate(tmp_path).guard(BUDGET + 1, NOW)


def test_budget_resets_in_a_new_month(tmp_path):
    gate = _gate(tmp_path, [900])
    next_month = dt.datetime(2026, 11, 1, tzinfo=dt.UTC)
    assert gate.remaining(next_month) == BUDGET
    gate.guard(900, next_month)


def test_negative_estimate_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        _gate(tmp_path).guard(-1, NOW)
