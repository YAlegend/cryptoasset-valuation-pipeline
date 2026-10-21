import numpy as np
import pandas as pd

from cryptoval.factors.panel import weekly_panel
from cryptoval.factors.settings import FactorConfig

SUNDAYS = pd.date_range("2020-01-05", periods=8, freq="W-SUN")
CONFIG = FactorConfig(exclude={}, min_assets=1)


def _daily(n_assets=4, days=None):
    days = days if days is not None else pd.date_range(SUNDAYS[0], SUNDAYS[-1], freq="D")
    rows = []
    for i in range(n_assets):
        for day in days:
            rows.append(
                {
                    "asset": f"a{i}",
                    "metric_date": day,
                    "price_usd": 10.0 * (1 + i) * 1.01 ** (day - SUNDAYS[0]).days,
                    "market_cap_usd": 1e8 * (1 + i),
                    "active_addresses": 1000.0 * (1 + i),
                    "mvrv": 1.0 + i,
                }
            )
    return pd.DataFrame(rows)


def test_only_sundays_are_used_and_the_return_is_to_the_next_sunday():
    panel = weekly_panel(_daily(), CONFIG)
    assert set(panel["week"]) == set(SUNDAYS[:-1])  # the last Sunday has no t+1
    row = panel[(panel["asset"] == "a0") & (panel["week"] == SUNDAYS[0])].iloc[0]
    assert np.isclose(row["ret"], 1.01**7 - 1)
    assert np.isclose(row["value"], 1000.0 / 1e8)
    assert np.isclose(row["size"], np.log(1e8))


def test_momentum_looks_back_three_weeks():
    a0 = weekly_panel(_daily(), CONFIG).query("asset == 'a0'").set_index("week")
    assert a0["momentum"].iloc[:3].isna().all()
    assert np.isclose(a0.loc[SUNDAYS[3], "momentum"], 1.01**21 - 1)


def test_a_missing_sunday_never_spans_two_weeks():
    # momentum_weeks=1, so SUNDAYS[5]'s lookback is the absent SUNDAYS[4]: NaN, not the
    # two-week change; SUNDAYS[3]'s t+1 is the absent Sunday, so it has no return.
    days = pd.date_range(SUNDAYS[0], SUNDAYS[-1], freq="D").drop(SUNDAYS[4])
    config = FactorConfig(exclude={}, min_assets=1, momentum_weeks=1)
    panel = weekly_panel(_daily(days=days), config)
    weeks = set(panel["week"])
    assert SUNDAYS[3] not in weeks and SUNDAYS[4] not in weeks
    a0 = panel[panel["asset"] == "a0"].set_index("week")
    assert np.isnan(a0.loc[SUNDAYS[5], "momentum"])


def test_exclusions_floor_and_thin_weeks():
    daily = _daily(n_assets=3)
    daily.loc[daily["asset"] == "a1", "market_cap_usd"] = 5e5  # below the $1M floor
    panel = weekly_panel(daily, FactorConfig(exclude={"a2": "wrapper"}, min_assets=1))
    assert set(panel["asset"]) == {"a0"}
    assert weekly_panel(daily, FactorConfig(exclude={}, min_assets=3)).empty


def test_an_empty_universe_is_an_empty_panel():
    assert weekly_panel(_daily().iloc[0:0], CONFIG).empty
