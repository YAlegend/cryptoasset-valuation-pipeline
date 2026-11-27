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
    # A September record must not reduce October's remaining budget.
    m = RunManifest(PathResolver(str(tmp_path)))
    sep = dt.datetime(2026, 9, 30, 12, tzinfo=dt.UTC)
    for started, b in [(NOW, 300), (NOW, 200), (sep, 400)]:
        rec = m.start("bigquery", {}, started_at=started)
        m.finish(rec, rows_in=0, rows_out=1, status="success", bytes_scanned=b)
    assert ByteBudgetGate(m, monthly_budget=BUDGET).remaining(NOW) == 500


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
    assert "would scan 500 bytes" in msg
    assert "100 bytes remain" in msg
    assert "2026-10" in msg
    assert "budget of 1000" in msg


def test_guard_refuses_a_single_query_larger_than_the_whole_budget(tmp_path):
    with pytest.raises(BudgetExceeded):
        _gate(tmp_path).guard(BUDGET + 1, NOW)


def test_remaining_clamps_at_zero_when_ledger_exceeds_budget(tmp_path):
    gate = _gate(tmp_path, [1200])
    assert gate.remaining(NOW) == 0


def test_zero_byte_query_is_allowed_on_an_exhausted_gate(tmp_path):
    # A zero-byte query consumes nothing from the allowance, so it must not be refused.
    _gate(tmp_path, [1200]).guard(0, NOW)


def test_budget_resets_in_a_new_month(tmp_path):
    gate = _gate(tmp_path, [900])
    next_month = dt.datetime(2026, 11, 1, tzinfo=dt.UTC)
    assert gate.remaining(next_month) == BUDGET
    gate.guard(900, next_month)


def test_negative_estimate_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        _gate(tmp_path).guard(-1, NOW)


def test_naive_now_is_rejected(tmp_path):
    naive = dt.datetime(2026, 10, 11)
    with pytest.raises(ValueError, match="timezone-aware"):
        _gate(tmp_path).remaining(naive)


def test_aware_non_utc_now_is_charged_to_its_utc_month(tmp_path):
    # 2026-10-31T23:00-05:00 is 2026-11-01T04:00Z, so it belongs to November.
    # The October ledger (900) must not be charged against it.
    boundary = dt.datetime(2026, 10, 31, 23, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
    gate = _gate(tmp_path, [900])
    assert gate.remaining(boundary) == BUDGET
    gate.guard(BUDGET, boundary)
