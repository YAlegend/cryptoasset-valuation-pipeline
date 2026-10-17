import numpy as np
import pandas as pd

from cryptoval.contracts.loader import Contract
from cryptoval.quarantine import QUARANTINE_COLUMNS, quarantine_impossible

CONTRACT = Contract(
    name="t",
    version="1.0.0",
    required_columns=["asset", "metric_date"],
    columns={
        "volume_usd": {"type": "float", "minimum": 0, "maximum": 1.0e12},
        "market_cap_usd": {"type": "float", "minimum": 0},
    },
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset": ["usdc", "usdc", "btc"],
            "metric_date": pd.to_datetime(["2026-05-05", "2026-05-06", "2026-05-05"], utc=True),
            "volume_usd": [8.02e52, 3.4e9, 2.9e9],
            "market_cap_usd": [7.52e10, 7.5e10, 1.6e12],
        }
    )


def test_an_impossible_value_is_nulled_and_the_rest_of_the_row_survives():
    clean, bad = quarantine_impossible(
        _frame(), CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert np.isnan(clean.loc[0, "volume_usd"])
    # The healthy market cap beside it is untouched: this is why the cell, not the row, goes.
    assert clean.loc[0, "market_cap_usd"] == 7.52e10
    assert len(clean) == 3


def test_the_raw_value_and_the_bound_it_broke_are_recorded():
    _, bad = quarantine_impossible(
        _frame(), CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert list(bad.columns) == QUARANTINE_COLUMNS
    assert len(bad) == 1
    row = bad.iloc[0]
    assert row["source"] == "coinmetrics"
    assert row["column"] == "volume_usd"
    assert row["run_id"] == "run-1"
    assert row["raw_value"] == 8.02e52
    assert "maximum 1e+12" in row["reason"]
    assert row["entity"] == "usdc"


def test_values_inside_the_bounds_are_untouched():
    frame = _frame()
    frame.loc[0, "volume_usd"] = 5.0e9
    clean, bad = quarantine_impossible(
        frame, CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert bad.empty
    assert list(bad.columns) == QUARANTINE_COLUMNS
    pd.testing.assert_frame_equal(clean, frame)


def test_a_value_below_a_minimum_is_quarantined_too():
    frame = _frame()
    frame.loc[2, "market_cap_usd"] = -1.0
    _, bad = quarantine_impossible(
        frame, CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert set(bad["column"]) == {"volume_usd", "market_cap_usd"}
    assert "minimum 0" in bad.set_index("column").loc["market_cap_usd", "reason"]


def test_a_null_is_not_impossible():
    frame = _frame()
    frame.loc[0, "volume_usd"] = np.nan
    _, bad = quarantine_impossible(
        frame, CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    # A missing value is a gap, which the null-rate ceiling already governs. Quarantine is
    # only for values the vendor supplied that cannot be true.
    assert bad.empty


def test_a_contract_column_absent_from_the_frame_is_ignored():
    frame = _frame().drop(columns=["volume_usd"])
    clean, bad = quarantine_impossible(
        frame, CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert bad.empty
    assert "volume_usd" not in clean.columns


def test_nulling_keeps_the_column_numeric():
    clean, _ = quarantine_impossible(
        _frame(), CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    # An object-dtype column would break stable_hash and therefore SCD2 change detection.
    assert clean["volume_usd"].dtype.kind == "f"


def test_an_empty_frame_is_handled():
    empty = _frame().iloc[0:0]
    clean, bad = quarantine_impossible(
        empty, CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert clean.empty and bad.empty
    assert list(bad.columns) == QUARANTINE_COLUMNS
