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
    path = write_bronze(r, "coinmetrics", DAY, _df(), {"start": "2026-10-11"})
    assert pathlib.Path(path).exists()
    back = read_bronze(r, "coinmetrics", DAY)
    assert len(back) == 1
    assert back.loc[0, "asset"] == "btc"


def test_adds_provenance_columns(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "coinmetrics", DAY, _df(), {"start": "2026-10-11"})
    back = read_bronze(r, "coinmetrics", DAY)
    assert "_fetched_at" in back.columns
    assert back.loc[0, "_source"] == "coinmetrics"
    assert len(back.loc[0, "_request_hash"]) == 64


def test_identical_params_produce_identical_request_hash(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, _df(), {"a": 1, "b": 2})
    first = read_bronze(r, "s", DAY).loc[0, "_request_hash"]
    write_bronze(r, "s", DAY, _df(), {"b": 2, "a": 1})
    assert read_bronze(r, "s", DAY).loc[0, "_request_hash"] == first


def test_rewriting_a_date_replaces_the_partition_wholesale(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, pd.DataFrame({"asset": ["btc"], "value": [1.0]}), {})
    write_bronze(r, "s", DAY, pd.DataFrame({"asset": ["eth"], "value": [2.0]}), {})
    back = read_bronze(r, "s", DAY)
    assert len(back) == 1
    assert back.loc[0, "asset"] == "eth"


def test_other_dates_are_untouched_by_a_rewrite(tmp_path):
    r = PathResolver(str(tmp_path))
    other = dt.date(2026, 10, 10)
    write_bronze(r, "s", other, _df(), {})
    write_bronze(r, "s", DAY, _df(), {})
    write_bronze(r, "s", DAY, _df(), {})
    assert len(read_bronze(r, "s", other)) == 1


def test_empty_frame_is_rejected_so_a_failed_fetch_cannot_look_complete(tmp_path):
    r = PathResolver(str(tmp_path))
    with pytest.raises(ValueError, match="refusing to write an empty"):
        write_bronze(r, "s", DAY, pd.DataFrame(), {})
    with pytest.raises(FileNotFoundError):
        read_bronze(r, "s", DAY)


def test_no_temporary_directory_is_left_behind(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, _df(), {})
    leftovers = [p.name for p in pathlib.Path(tmp_path).rglob("*.tmp*")]
    assert leftovers == []


def test_remote_root_is_rejected_with_a_clear_message(tmp_path):
    r = PathResolver("gs://bucket/root")
    with pytest.raises(NotImplementedError, match="object storage"):
        write_bronze(r, "s", DAY, _df(), {})


def test_failed_rewrite_keeps_the_previous_partition_and_leaves_no_temporary(tmp_path, monkeypatch):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, _df(), {})

    def dies_mid_write(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", dies_mid_write)
    with pytest.raises(OSError, match="disk full"):
        write_bronze(r, "s", DAY, pd.DataFrame({"asset": ["eth"], "value": [2.0]}), {})

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
        write_bronze(r, "s", DAY, _df(), {})

    assert not pathlib.Path(r.bronze_partition("s", DAY)).exists()
    with pytest.raises(FileNotFoundError):
        read_bronze(r, "s", DAY)
    leftovers = [p.name for p in pathlib.Path(tmp_path).rglob("*.tmp*")]
    assert leftovers == []
