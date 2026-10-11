import datetime as dt

import pandas as pd

from cryptoval.scd2 import (
    FAR_FUTURE,
    add_row_hash,
    as_of_view,
    current_view,
    merge_scd2,
    reconcile_report,
)

KEY = ["asset", "metric_date"]
VALS = ["market_cap"]
T1 = dt.datetime(2026, 10, 11, 2, tzinfo=dt.UTC)
T2 = dt.datetime(2026, 10, 12, 2, tzinfo=dt.UTC)
T3 = dt.datetime(2026, 10, 13, 2, tzinfo=dt.UTC)


def _incoming(market_cap: float, asset: str = "btc", day: str = "2026-10-10"):
    return pd.DataFrame(
        {
            "asset": [asset],
            "metric_date": pd.to_datetime([day], utc=True),
            "market_cap": [market_cap],
        }
    )


def _empty():
    return pd.DataFrame(columns=[*KEY, *VALS])


def test_row_hash_is_stable_and_content_sensitive():
    a = add_row_hash(_incoming(100.0), VALS)
    b = add_row_hash(_incoming(100.0), VALS)
    c = add_row_hash(_incoming(101.0), VALS)
    assert a.loc[0, "row_hash"] == b.loc[0, "row_hash"]
    assert a.loc[0, "row_hash"] != c.loc[0, "row_hash"]


def test_first_load_inserts_one_open_current_row():
    out = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    assert len(out) == 1
    assert out.loc[0, "is_current"]
    assert out.loc[0, "effective_start_ts"] == pd.Timestamp(T1)
    assert out.loc[0, "effective_end_ts"] == FAR_FUTURE


def test_identical_reload_writes_nothing_new():
    first = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    second = merge_scd2(first, _incoming(100.0), KEY, VALS, T2)
    assert len(second) == 1
    assert second.loc[0, "effective_start_ts"] == pd.Timestamp(T1)
    assert second.loc[0, "is_current"]


def test_revision_closes_the_old_row_and_opens_a_new_one():
    first = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    second = merge_scd2(first, _incoming(110.0), KEY, VALS, T2)
    assert len(second) == 2
    closed = second[~second["is_current"]].iloc[0]
    openrow = second[second["is_current"]].iloc[0]
    assert closed["market_cap"] == 100.0
    assert closed["effective_end_ts"] == pd.Timestamp(T2)
    assert openrow["market_cap"] == 110.0
    assert openrow["effective_start_ts"] == pd.Timestamp(T2)
    assert openrow["effective_end_ts"] == FAR_FUTURE


def test_exactly_one_current_row_per_key_after_several_revisions():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(110.0), KEY, VALS, T2)
    hist = merge_scd2(hist, _incoming(120.0), KEY, VALS, T3)
    assert len(hist) == 3
    assert int(hist["is_current"].sum()) == 1
    assert hist[hist["is_current"]].iloc[0]["market_cap"] == 120.0


def test_duplicate_incoming_keys_cannot_create_two_current_rows():
    dupes = pd.concat([_incoming(100.0), _incoming(105.0)], ignore_index=True)
    out = merge_scd2(_empty(), dupes, KEY, VALS, T1)
    assert len(out) == 1
    assert int(out["is_current"].sum()) == 1
    # last row wins, mirroring an API that repeats a key across pages
    assert out.loc[0, "market_cap"] == 105.0


def test_as_of_view_reconstructs_the_old_vintage():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(110.0), KEY, VALS, T2)
    before = as_of_view(hist, dt.datetime(2026, 10, 11, 12, tzinfo=dt.UTC))
    after = as_of_view(hist, dt.datetime(2026, 10, 12, 12, tzinfo=dt.UTC))
    assert len(before) == 1 and before.iloc[0]["market_cap"] == 100.0
    assert len(after) == 1 and after.iloc[0]["market_cap"] == 110.0


def test_as_of_view_before_any_knowledge_is_empty():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    assert as_of_view(hist, dt.datetime(2026, 1, 1, tzinfo=dt.UTC)).empty


def test_current_view_returns_only_open_rows():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(110.0), KEY, VALS, T2)
    assert len(current_view(hist)) == 1
    assert current_view(hist).iloc[0]["market_cap"] == 110.0


def test_independent_keys_do_not_interfere():
    hist = merge_scd2(_empty(), _incoming(100.0, asset="btc"), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(50.0, asset="eth"), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(110.0, asset="btc"), KEY, VALS, T2)
    assert int(hist["is_current"].sum()) == 2
    eth = current_view(hist)
    eth = eth[eth["asset"] == "eth"].iloc[0]
    assert eth["market_cap"] == 50.0
    assert eth["effective_start_ts"] == pd.Timestamp(T1)


def test_reconcile_report_counts_changes_outside_the_trailing_window():
    hist = merge_scd2(_empty(), _incoming(100.0, day="2024-01-01"), KEY, VALS, T1)
    report = reconcile_report(hist, _incoming(999.0, day="2024-01-01"), KEY, VALS)
    assert report["changed"] == 1
    assert report["new"] == 0
    assert report["unchanged"] == 0


def test_reconcile_report_reports_a_clean_reconciliation():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    report = reconcile_report(hist, _incoming(100.0), KEY, VALS)
    assert report == {"new": 0, "changed": 0, "unchanged": 1}


def test_as_of_view_boundaries_are_start_inclusive_end_exclusive():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(110.0), KEY, VALS, T2)
    # the instant of a revision belongs to the new vintage only, never both
    assert as_of_view(hist, T1)["market_cap"].tolist() == [100.0]
    assert as_of_view(hist, T2)["market_cap"].tolist() == [110.0]
    assert as_of_view(hist, T2 - dt.timedelta(microseconds=1))["market_cap"].tolist() == [100.0]


def test_revision_intervals_abut_without_gap_or_overlap():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(110.0), KEY, VALS, T2)
    hist = merge_scd2(hist, _incoming(120.0), KEY, VALS, T3)
    ordered = hist.sort_values("effective_start_ts").reset_index(drop=True)
    assert ordered["effective_end_ts"].iloc[:-1].tolist() == (
        ordered["effective_start_ts"].iloc[1:].tolist()
    )


def test_missing_to_present_and_present_to_missing_are_revisions():
    nan = float("nan")
    a = add_row_hash(_incoming(nan), VALS).loc[0, "row_hash"]
    assert a == add_row_hash(_incoming(nan), VALS).loc[0, "row_hash"]
    assert a != add_row_hash(_incoming(1.0), VALS).loc[0, "row_hash"]
    filled = merge_scd2(
        merge_scd2(_empty(), _incoming(nan), KEY, VALS, T1), _incoming(5.0), KEY, VALS, T2
    )
    assert len(filled) == 2 and int(filled["is_current"].sum()) == 1
    dropped = merge_scd2(
        merge_scd2(_empty(), _incoming(5.0), KEY, VALS, T1), _incoming(nan), KEY, VALS, T2
    )
    assert len(dropped) == 2 and int(dropped["is_current"].sum()) == 1
    still_missing = merge_scd2(
        merge_scd2(_empty(), _incoming(nan), KEY, VALS, T1), _incoming(nan), KEY, VALS, T2
    )
    assert len(still_missing) == 1


def test_single_column_key_revises_only_the_changed_key():
    empty = pd.DataFrame(columns=["asset", "market_cap"])
    one = pd.DataFrame({"asset": ["a", "b"], "market_cap": [1.0, 2.0]})
    two = pd.DataFrame({"asset": ["a", "b"], "market_cap": [1.5, 2.0]})
    hist = merge_scd2(merge_scd2(empty, one, ["asset"], VALS, T1), two, ["asset"], VALS, T2)
    assert len(hist) == 3 and int(hist["is_current"].sum()) == 2
    assert sorted(current_view(hist)["market_cap"].tolist()) == [1.5, 2.0]


def test_reconcile_report_counts_new_keys_and_writes_nothing():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    snapshot = hist.copy(deep=True)
    both = pd.concat([_incoming(100.0), _incoming(7.0, asset="eth")], ignore_index=True)
    assert reconcile_report(hist, both, KEY, VALS) == {"new": 1, "changed": 0, "unchanged": 1}
    pd.testing.assert_frame_equal(hist, snapshot)
