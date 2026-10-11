import numpy as np
import pandas as pd

from cryptoval.contracts.identities import (
    check_cointime_complement,
    check_monotonic_nondecreasing,
    check_mvrv_identity,
    check_supply_identity,
    run_identity_checks,
)


def test_mvrv_identity_holds_for_consistent_rows():
    df = pd.DataFrame(
        {
            "CapMrktCurUSD": [1.725e12],
            "CapMVRVCur": [1.5967],
            "realized_cap": [1.725e12 / 1.5967],
        }
    )
    assert check_mvrv_identity(df).passed


def test_mvrv_identity_fails_when_realized_cap_is_inconsistent():
    df = pd.DataFrame(
        {"CapMrktCurUSD": [1.0e12], "CapMVRVCur": [2.0], "realized_cap": [9.9e11]}
    )
    res = check_mvrv_identity(df)
    assert not res.passed
    assert "realized_cap" in res.detail


def test_mvrv_identity_ignores_null_rows_rather_than_failing():
    df = pd.DataFrame(
        {
            "CapMrktCurUSD": [1.0e12, 1.0e12],
            "CapMVRVCur": [2.0, np.nan],
            "realized_cap": [5.0e11, np.nan],
        }
    )
    assert check_mvrv_identity(df).passed


def test_supply_identity_holds_and_fails():
    good = pd.DataFrame({"SplyCur": [19_000_000.0], "PriceUSD": [100.0],
                         "CapMrktCurUSD": [1.9e9]})
    assert check_supply_identity(good).passed
    bad = pd.DataFrame({"SplyCur": [19_000_000.0], "PriceUSD": [100.0],
                        "CapMrktCurUSD": [1.0e9]})
    assert not check_supply_identity(bad).passed


def test_monotonic_check_passes_for_nondecreasing_series_per_asset():
    df = pd.DataFrame(
        {
            "asset": ["btc", "btc", "eth", "eth"],
            "metric_date": pd.to_datetime(
                ["2026-10-01", "2026-10-02", "2026-10-01", "2026-10-02"], utc=True
            ),
            "thermocap": [100.0, 110.0, 50.0, 55.0],
        }
    )
    assert check_monotonic_nondecreasing(df, "thermocap").passed


def test_monotonic_check_fails_when_a_series_decreases():
    df = pd.DataFrame(
        {
            "asset": ["btc", "btc"],
            "metric_date": pd.to_datetime(["2026-10-01", "2026-10-02"], utc=True),
            "thermocap": [110.0, 100.0],
        }
    )
    res = check_monotonic_nondecreasing(df, "thermocap")
    assert not res.passed
    assert "btc" in res.detail


def test_monotonic_check_is_not_fooled_by_unsorted_input():
    df = pd.DataFrame(
        {
            "asset": ["btc", "btc"],
            "metric_date": pd.to_datetime(["2026-10-02", "2026-10-01"], utc=True),
            "thermocap": [110.0, 100.0],
        }
    )
    assert check_monotonic_nondecreasing(df, "thermocap").passed


def test_cointime_complement_requires_liveliness_plus_vaultedness_to_equal_one():
    good = pd.DataFrame({"liveliness": [0.4], "vaultedness": [0.6]})
    assert check_cointime_complement(good).passed
    bad = pd.DataFrame({"liveliness": [0.4], "vaultedness": [0.5]})
    assert not check_cointime_complement(bad).passed


def test_run_identity_checks_skips_checks_whose_columns_are_absent():
    df = pd.DataFrame({"CapMrktCurUSD": [1.0e12]})
    results = run_identity_checks(df)
    assert all(r.passed for r in results)
