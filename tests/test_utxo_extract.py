import datetime as dt

import pandas as pd
import pytest

from cryptoval.budget import BudgetExceeded, ByteBudgetGate
from cryptoval.io.bronze import write_bronze
from cryptoval.manifest import RunManifest
from cryptoval.paths import PathResolver
from cryptoval.utxo.extract import (
    GENESIS,
    Range,
    assert_contiguous,
    completed_runs,
    extract,
    next_range,
    read_history,
)

TODAY = dt.date(2009, 1, 10)
NOW = dt.datetime(2009, 1, 10, 12, tzinfo=dt.UTC)

FRAMES = {
    "inputs": pd.DataFrame(
        {
            "created_date": [GENESIS],
            "spent_date": [dt.date(2009, 1, 5)],
            "value_sat": [10],
            "coinblocks_destroyed": [1.0],
            "n_inputs": [1],
        }
    ),
    "outputs": pd.DataFrame(
        {"created_date": [GENESIS], "value_sat": [5_000_000_000], "n_outputs": [1]}
    ),
    "blocks": pd.DataFrame({"height": [0], "block_date": [GENESIS]}),
}


class FakeClient:
    def __init__(self, estimate=100, fail_on=None):
        self.estimate, self.fail_on = estimate, fail_on
        self.dry, self.ran = [], []

    def dry_run(self, sql):
        self.dry.append(sql)
        return self.estimate

    def run(self, sql, max_bytes):
        key = next(k for k in FRAMES if f"crypto_bitcoin.{k}" in sql)
        if key == self.fail_on:
            raise RuntimeError("boom")
        self.ran.append((key, max_bytes))
        return FRAMES[key].copy(), self.estimate


def _history(start, end):
    return pd.DataFrame(
        {
            "asset": ["btc"],
            "metric_date": [pd.Timestamp(end, tz="UTC")],
            "PriceUSD": [1.0],
            "SplyCur": [50.0],
            "realized_cap": [1.0],
        }
    )


def _run(tmp_path, client, budget=10_000, today=TODAY, dry_run=False):
    resolver = PathResolver(str(tmp_path))
    manifest = RunManifest(resolver)
    gate = ByteBudgetGate(manifest, budget)
    report = extract(resolver, client, gate, manifest, today, NOW, _history, dry_run=dry_run)
    return resolver, manifest, report


def test_the_first_run_covers_genesis_to_two_days_ago(tmp_path):
    resolver, manifest, report = _run(tmp_path, FakeClient())
    assert report["status"] == "extracted"
    [(rng, _)] = completed_runs(resolver)
    assert rng == Range(GENESIS, dt.date(2009, 1, 8))
    assert manifest.read()["bytes_scanned"].sum() == 300  # actual billed bytes, all three
    assert not read_history(resolver).empty


def test_the_next_run_starts_the_day_after_the_last_end(tmp_path):
    _run(tmp_path, FakeClient())
    resolver, _, _ = _run(tmp_path, FakeClient(), today=dt.date(2009, 1, 15))
    ranges = [r for r, _ in completed_runs(resolver)]
    assert ranges[1] == Range(dt.date(2009, 1, 9), dt.date(2009, 1, 13))
    assert assert_contiguous(completed_runs(resolver)) == dt.date(2009, 1, 13)


def test_up_to_date_issues_no_query(tmp_path):
    _run(tmp_path, FakeClient())
    client = FakeClient()
    _, _, report = _run(tmp_path, client)
    assert report["status"] == "up_to_date" and client.dry == [] and client.ran == []


def test_the_gate_sees_the_total_of_all_three_queries(tmp_path):
    client = FakeClient(estimate=400)  # each fits in 1000, the three together do not
    with pytest.raises(BudgetExceeded):
        _run(tmp_path, client, budget=1000)
    assert client.ran == []


def test_dry_run_prices_the_range_and_runs_nothing(tmp_path):
    client = FakeClient()
    resolver, _, report = _run(tmp_path, client, dry_run=True)
    assert report["status"] == "dry_run" and report["estimated_bytes"] == 300
    assert client.ran == [] and completed_runs(resolver) == []


def test_a_partial_run_is_ignored_and_its_range_redone(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        _run(tmp_path, FakeClient(fail_on="outputs"))
    resolver = PathResolver(str(tmp_path))
    assert completed_runs(resolver) == []
    assert next_range(resolver, TODAY) == Range(GENESIS, dt.date(2009, 1, 8))
    assert RunManifest(resolver).read()["status"].tolist() == ["success", "failed"]


def test_the_query_cap_is_what_remains_of_the_month(tmp_path):
    client = FakeClient()
    _run(tmp_path, client, budget=10_000)
    assert [cap for _, cap in client.ran] == [10_000, 9_900, 9_800]


def test_overlapping_runs_are_refused(tmp_path):
    resolver = PathResolver(str(tmp_path))
    spans = {"r1": (GENESIS, dt.date(2009, 1, 8)), "r2": (dt.date(2009, 1, 5), dt.date(2009, 1, 9))}
    for run_id, (a, b) in spans.items():
        for source, key in (
            ("btc_utxo_spent", "inputs"),
            ("btc_utxo_created", "outputs"),
            ("btc_blocks", "blocks"),
        ):
            frame = FRAMES[key].assign(range_start=a, range_end=b)
            write_bronze(resolver, source, b, frame, {}, run_id)
    with pytest.raises(ValueError, match="overlap"):
        completed_runs(resolver)


def test_a_gap_or_a_late_start_is_not_contiguous():
    runs = [
        (Range(GENESIS, dt.date(2009, 1, 8)), "a"),
        (Range(dt.date(2009, 1, 10), dt.date(2009, 1, 12)), "b"),
    ]
    with pytest.raises(ValueError, match="gap"):
        assert_contiguous(runs)
    with pytest.raises(ValueError, match="genesis"):
        assert_contiguous([(Range(dt.date(2009, 1, 4), dt.date(2009, 1, 8)), "a")])
