# tests/test_bronze.py
import datetime as dt
import pathlib

import pandas as pd
import pytest

from cryptoval.io.bronze import read_bronze, write_bronze
from cryptoval.paths import PathResolver

DAY = dt.date(2026, 10, 11)


def _df():
    return pd.DataFrame({"asset": ["btc"], "value": [1.0]})


def test_writes_partition_and_round_trips(tmp_path):
    r = PathResolver(str(tmp_path))
    path = write_bronze(r, "coinmetrics", DAY, _df(), {"start": "2026-10-11"}, "run-1")
    assert pathlib.Path(path).exists()
    back = read_bronze(r, "coinmetrics", DAY)
    assert len(back) == 1
    assert back.loc[0, "asset"] == "btc"


def test_adds_provenance_columns(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "coinmetrics", DAY, _df(), {"start": "2026-10-11"}, "run-1")
    back = read_bronze(r, "coinmetrics", DAY)
    assert "_fetched_at" in back.columns
    assert back.loc[0, "_run_id"] == "run-1"
    assert back.loc[0, "_source"] == "coinmetrics"
    assert len(back.loc[0, "_request_hash"]) == 64


def test_identical_params_produce_identical_request_hash(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, _df(), {"a": 1, "b": 2}, "run-1")
    first = read_bronze(r, "s", DAY, "run-1").loc[0, "_request_hash"]
    write_bronze(r, "s", DAY, _df(), {"b": 2, "a": 1}, "run-2")
    assert read_bronze(r, "s", DAY, "run-2").loc[0, "_request_hash"] == first


def test_two_runs_on_the_same_date_both_survive(tmp_path):
    """The defect this fixes: re-running a subset used to delete the earlier run's record."""
    r = PathResolver(str(tmp_path))
    first = pd.DataFrame({"asset": ["btc", "eth"], "v": [1.0, 2.0]})
    second = pd.DataFrame({"asset": ["btc"], "v": [1.5]})

    write_bronze(r, "coinmetrics", dt.date(2026, 10, 12), first, {"a": 1}, "run-1")
    write_bronze(r, "coinmetrics", dt.date(2026, 10, 12), second, {"a": 2}, "run-2")

    runs = sorted(
        p.name
        for p in pathlib.Path(r.bronze_date_prefix("coinmetrics", dt.date(2026, 10, 12))).iterdir()
    )
    assert runs == ["run_id=run-1", "run_id=run-2"]
    # The partial re-run did not take eth's response with it.
    assert set(read_bronze(r, "coinmetrics", dt.date(2026, 10, 12), "run-1")["asset"]) == {
        "btc",
        "eth",
    }


def test_read_bronze_returns_the_newest_run_by_default(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "coinmetrics", dt.date(2026, 10, 12), pd.DataFrame({"v": [1.0]}), {}, "run-1")
    write_bronze(r, "coinmetrics", dt.date(2026, 10, 12), pd.DataFrame({"v": [9.0]}), {}, "run-2")
    assert read_bronze(r, "coinmetrics", dt.date(2026, 10, 12))["v"].iloc[0] == 9.0


def test_rewriting_the_same_run_id_is_refused(tmp_path):
    """A run_id is unique per run, so a collision means a bug, not an intentional rewrite."""
    r = PathResolver(str(tmp_path))
    write_bronze(r, "coinmetrics", dt.date(2026, 10, 12), pd.DataFrame({"v": [1.0]}), {}, "run-1")
    with pytest.raises(FileExistsError, match="run-1"):
        write_bronze(
            r, "coinmetrics", dt.date(2026, 10, 12), pd.DataFrame({"v": [2.0]}), {}, "run-1"
        )
    # Refused means untouched: the original response is still there, and no staging is left.
    assert read_bronze(r, "coinmetrics", dt.date(2026, 10, 12), "run-1")["v"].iloc[0] == 1.0
    assert [p.name for p in pathlib.Path(tmp_path).rglob("*.tmp*")] == []


def test_reading_a_date_with_no_runs_raises(tmp_path):
    r = PathResolver(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        read_bronze(r, "coinmetrics", dt.date(2026, 10, 12))


def test_reading_an_unknown_run_id_raises(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, _df(), {}, "run-1")
    with pytest.raises(FileNotFoundError, match="run-nope"):
        read_bronze(r, "s", DAY, "run-nope")


def test_other_dates_are_untouched_by_further_runs(tmp_path):
    r = PathResolver(str(tmp_path))
    other = dt.date(2026, 10, 10)
    write_bronze(r, "s", other, _df(), {}, "run-0")
    write_bronze(r, "s", DAY, _df(), {}, "run-1")
    write_bronze(r, "s", DAY, _df(), {}, "run-2")
    assert len(read_bronze(r, "s", other)) == 1


def test_empty_frame_is_rejected_so_a_failed_fetch_cannot_look_complete(tmp_path):
    r = PathResolver(str(tmp_path))
    with pytest.raises(ValueError, match="refusing to write an empty"):
        write_bronze(r, "s", DAY, pd.DataFrame(), {}, "run-1")
    with pytest.raises(FileNotFoundError):
        read_bronze(r, "s", DAY)


def test_no_temporary_directory_is_left_behind(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, _df(), {}, "run-1")
    leftovers = [p.name for p in pathlib.Path(tmp_path).rglob("*.tmp*")]
    assert leftovers == []


def test_remote_root_is_rejected_with_a_clear_message(tmp_path):
    r = PathResolver("gs://bucket/root")
    with pytest.raises(NotImplementedError, match="object storage"):
        write_bronze(r, "s", DAY, _df(), {}, "run-1")


def test_failed_later_run_keeps_the_previous_run_and_leaves_no_temporary(tmp_path, monkeypatch):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, _df(), {}, "run-1")

    def dies_mid_write(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", dies_mid_write)
    with pytest.raises(OSError, match="disk full"):
        write_bronze(r, "s", DAY, pd.DataFrame({"asset": ["eth"], "value": [2.0]}), {}, "run-2")

    back = read_bronze(r, "s", DAY)
    assert back.loc[0, "asset"] == "btc"
    leftovers = [p.name for p in pathlib.Path(tmp_path).rglob("*.tmp*")]
    assert leftovers == []


def test_failed_first_write_leaves_no_partition(tmp_path, monkeypatch):
    r = PathResolver(str(tmp_path))

    def dies_mid_write(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", dies_mid_write)
    with pytest.raises(OSError, match="disk full"):
        write_bronze(r, "s", DAY, _df(), {}, "run-1")

    assert not pathlib.Path(r.bronze_partition("s", DAY, "run-1")).exists()
    with pytest.raises(FileNotFoundError):
        read_bronze(r, "s", DAY)
    leftovers = [p.name for p in pathlib.Path(tmp_path).rglob("*.tmp*")]
    assert leftovers == []
