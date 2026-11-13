# tests/test_paths.py
import datetime as dt

import pytest

from cryptoval.paths import PathResolver


def test_local_layer_path():
    r = PathResolver("./data")
    assert r.layer("bronze") == "./data/bronze"


def test_trailing_slash_in_root_is_normalised():
    assert PathResolver("./data/").layer("silver") == "./data/silver"


def test_remote_root_is_passed_through_unchanged():
    r = PathResolver("gs://my-bucket/cryptoval")
    assert r.layer("gold") == "gs://my-bucket/cryptoval/gold"
    assert r.is_remote is True


def test_local_root_is_not_remote():
    assert PathResolver("./data").is_remote is False


def test_bronze_partition_includes_the_run_id():
    r = PathResolver("/data")
    assert r.bronze_partition("coinmetrics", dt.date(2026, 10, 12), "run-abc") == (
        "/data/bronze/source=coinmetrics/ingest_date=2026-10-12/run_id=run-abc"
    )


def test_bronze_date_prefix_holds_every_run_for_that_date():
    r = PathResolver("/data")
    assert r.bronze_date_prefix("coinmetrics", dt.date(2026, 10, 12)) == (
        "/data/bronze/source=coinmetrics/ingest_date=2026-10-12"
    )


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_bronze_partition_rejects_an_empty_run_id(bad):
    with pytest.raises(ValueError, match="run_id"):
        PathResolver("/data").bronze_partition("coinmetrics", dt.date(2026, 10, 12), bad)


def test_table_path():
    r = PathResolver("./data")
    assert r.table("silver", "asset_metrics") == "./data/silver/asset_metrics"


def test_unknown_layer_is_rejected():
    with pytest.raises(ValueError, match="unknown layer"):
        PathResolver("./data").layer("platinum")
