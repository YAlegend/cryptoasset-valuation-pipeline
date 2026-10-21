import numpy as np
import pandas as pd

from cryptoval.factors.runner import FACTOR_COLUMNS, run_factors
from cryptoval.factors.settings import FactorConfig

CONFIG = FactorConfig(exclude={}, min_assets=50)


def _universe(n_assets=60, weeks=80, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2018-01-07", periods=weeks, freq="W-SUN")
    frames = []
    for i in range(n_assets):
        price = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.1, weeks)))
        frames.append(
            pd.DataFrame(
                {
                    "asset": f"a{i:02d}",
                    "metric_date": days,
                    "price_usd": price,
                    "market_cap_usd": price * 1e7 * (1 + i),
                    "active_addresses": 1e4 * rng.uniform(1, 2),
                    "mvrv": rng.uniform(0.5, 3.0, weeks),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_thirteen_hypotheses_one_holm_family():
    table, returns = run_factors(_universe(), CONFIG)
    assert list(table.columns) == FACTOR_COLUMNS
    assert (table["metric"] == "refused").sum() == 0
    assert table[table["metric"] == "hypotheses_tested"]["value"].item() == 13
    assert (table["metric"] == "p_value_holm").sum() == 13
    assert not table.duplicated(["model_id", "signal", "portfolio", "metric"]).any()
    assert {"HL_value", "CMKT"} <= set(returns["series"])
    assert not returns.duplicated(["week", "series"]).any()


def test_an_empty_universe_refuses_every_test():
    table, returns = run_factors(_universe().iloc[0:0], CONFIG)
    assert (table["metric"] == "refused").sum() == 13
    assert table[table["metric"] == "hypotheses_tested"]["value"].item() == 0
    assert returns.empty and list(returns.columns) == ["week", "series", "value"]
