# Cross-Sectional Asset Pricing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether size, momentum, active-address value and MVRV earn returns across the Coin Metrics community universe: decile portfolios, Liu-Tsyvinski-Wu factors, factor alphas and Fama-MacBeth, under one Holm family.

**Architecture:** A new `cryptoval/factors/` package: config loader, Coin Metrics universe fetch writing silver `cm_universe_daily`, a weekly panel, portfolios, Newey-West inference, and a runner writing gold `factor_test` + `factor_returns`, with dbt marts over both. CLI `factors fetch|run`, `make factors` inside `make all`.

**Tech Stack:** pandas 3.0.6, statsmodels 0.15 (OLS HAC), httpx, PyYAML, dbt-duckdb.

**Spec:** `docs/superpowers/specs/2026-10-21-factor-models-design.md`

## Global Constraints

- Universe = catalog ids serving daily `CapMrktCurUSD`; exclusions only by asset type (design §2).
- Weekly = observation dated Sunday, no forward fill; eligible needs market cap ≥ $1,000,000 and prices at t and t+1.
- Week used only with ≥ 50 assets (per signal for deciles, all four for Fama-MacBeth).
- Value-weighted portfolios; Newey-West 4 lags; FM returns winsorized at 5%/95% per week.
- Exactly 13 hypotheses in one Holm family via `cryptoval.scoring.multiplicity.correct_family_wise`.
- A test that cannot run is a `refused` row carrying the reason.
- Python 3.14, ruff clean (`make lint`), tests run with `.venv/bin/python -m pytest`.

## Review Focus

1. A Sunday missing for every asset mid-sample: returns and momentum must not silently span two weeks. Pinned in Task 3 (`test_a_missing_sunday_never_spans_two_weeks`).
2. A misspelled exclusion id: must fail loudly, not keep the asset. Pinned in Task 2 (`test_an_exclusion_not_in_the_catalog_is_refused`).
3. An empty or too-short universe (first run, fetch failed): the run must write refusal rows, not crash. Pinned in Task 5 (`test_an_empty_universe_refuses_every_test`).
4. Decile ties (many equal MVRVs): the assignment must be deterministic. Pinned in Task 4 (`test_ties_are_broken_by_asset_id`).
5. One +900% micro-cap in Fama-MacBeth: winsorizing must stop it setting the slope. Pinned in Task 4 (`test_fama_macbeth_winsorizes_an_outlier`).

---

### Task 1: Configuration

**Files:**
- Create: `config/factors.yml`, `cryptoval/factors/__init__.py`, `cryptoval/factors/settings.py`
- Test: `tests/test_factors_settings.py`

**Interfaces:**
- Produces: `FactorConfig(exclude: dict[str, str], min_market_cap_usd: float, min_assets: int, momentum_weeks: int, newey_west_lags: int, winsor_quantile: float, deciles: int, tail_share: float)`; `load_factor_config(path="config/factors.yml") -> FactorConfig`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from cryptoval.factors.settings import load_factor_config


def test_the_repository_config_loads():
    config = load_factor_config()
    assert config.exclude["wbtc"] == "wrapper"
    assert config.exclude["usdt"] == "pegged"
    assert (config.min_assets, config.deciles, config.newey_west_lags) == (50, 10, 4)


def test_an_id_excluded_twice_is_refused(tmp_path):
    path = tmp_path / "f.yml"
    path.write_text("exclude:\n  pegged: [usdt]\n  wrapper: [usdt]\n")
    with pytest.raises(ValueError, match="usdt"):
        load_factor_config(str(path))


def test_an_unknown_key_is_refused(tmp_path):
    path = tmp_path / "f.yml"
    path.write_text("exclude: {}\nmin_asets: 3\n")
    with pytest.raises(TypeError):
        load_factor_config(str(path))
```

- [ ] **Step 2: Run, expect FAIL** — `.venv/bin/python -m pytest tests/test_factors_settings.py -q` → `ModuleNotFoundError: cryptoval.factors`.

- [ ] **Step 3: Implement**

`config/factors.yml`:

```yaml
# Spec 11.5 on the Coin Metrics community universe (design 2026-10-21-factor-models).
# Exclusions depend on what a listing IS, never on how it performed.
min_market_cap_usd: 1000000   # Liu-Tsyvinski-Wu's floor
min_assets: 50                # first met 2017-12-03, about 5 per decile
momentum_weeks: 3
newey_west_lags: 4
winsor_quantile: 0.05         # Fama-MacBeth only
deciles: 10
tail_share: 0.3               # LTW 30/40/30 legs
exclude:
  pegged: [usdt, usdt_omni, usdt_eth, usdt_trx, usdt_avaxc, tusd, tusd_eth, tusd_trx, gusd,
           usdc, usdc_eth, usdc_trx, usdc_avaxc, pax, busd, husd, dai, sai, usdk, frax_eth,
           lusd_eth, usdd_eth, eurc_eth, fdusd_eth, pyusd_eth, crvusd_eth, usde_eth, susde_eth,
           usdm_eth, sdai_eth, buidl_eth, xaut, paxg]
  wrapper: [wbtc, weth, renbtc, hbtc, wnxm]
  # the same token on another chain: one listing kept (avaxp, leo_eth, bnb, flow_native)
  other_chain: [avaxc, avaxx, leo_eos, bnb_eth, flow]
```

`cryptoval/factors/__init__.py`: `"""Cross-sectional asset pricing on the Coin Metrics universe (spec 11.5)."""`

`cryptoval/factors/settings.py`:

```python
"""The factor chain's configuration: floors, lags and the exclusion list are data."""

from __future__ import annotations

import dataclasses
import pathlib

import yaml


@dataclasses.dataclass(frozen=True)
class FactorConfig:
    exclude: dict[str, str]  # listing id -> why it is not a risk asset of its own
    min_market_cap_usd: float = 1_000_000.0
    min_assets: int = 50
    momentum_weeks: int = 3
    newey_west_lags: int = 4
    winsor_quantile: float = 0.05
    deciles: int = 10
    tail_share: float = 0.3


def load_factor_config(path: str = "config/factors.yml") -> FactorConfig:
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    exclude: dict[str, str] = {}
    for reason, ids in (raw.pop("exclude", None) or {}).items():
        for listing in ids:
            if listing in exclude:
                raise ValueError(f"{listing!r} is excluded twice ({exclude[listing]}, {reason})")
            exclude[listing] = reason
    return FactorConfig(exclude=exclude, **raw)
```

- [ ] **Step 4: Run, expect PASS** (3 tests).
- [ ] **Step 5: Commit** `feat: factor chain configuration and its exclusion list`.

---

### Task 2: The universe fetch

**Files:**
- Create: `cryptoval/factors/universe.py`
- Test: `tests/test_factors_universe.py`

**Interfaces:**
- Consumes: `get_with_retry`, `parse_timeseries`, `BASE_URL`, `MAX_PAGES` from `cryptoval.sources`; `FactorConfig`.
- Produces: `UNIVERSE_COLUMNS = ["asset", "metric_date", "price_usd", "market_cap_usd", "active_addresses", "mvrv"]`; `catalog_ids(client) -> list[str]`; `check_exclusions(config, ids) -> None`; `fetch_universe(client, ids, max_retries=5) -> pd.DataFrame` (UNIVERSE_COLUMNS, tz-naive `metric_date`); `to_universe(raw) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
import httpx
import pandas as pd
import pytest

from cryptoval.factors import universe
from cryptoval.factors.settings import FactorConfig


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_catalog_ids_keeps_daily_market_cap_listings_only():
    payload = {
        "data": [
            {"asset": "btc", "metrics": [{"metric": "CapMrktCurUSD", "frequencies": [{"frequency": "1d"}]}]},
            {"asset": "xyz", "metrics": [{"metric": "CapMrktCurUSD", "frequencies": [{"frequency": "1h"}]}]},
            {"asset": "abc", "metrics": [{"metric": "PriceUSD", "frequencies": [{"frequency": "1d"}]}]},
        ]
    }
    with _client(lambda request: httpx.Response(200, json=payload)) as client:
        assert universe.catalog_ids(client) == ["btc"]


def test_an_exclusion_not_in_the_catalog_is_refused():
    config = FactorConfig(exclude={"wbtc": "wrapper", "wbtcc": "wrapper"})
    with pytest.raises(ValueError, match="wbtcc"):
        universe.check_exclusions(config, ["btc", "wbtc"])


def test_fetch_follows_pages_and_asks_to_ignore_unsupported_metrics():
    seen = []

    def handler(request):
        seen.append(request.url)
        if "page=2" in str(request.url):
            return httpx.Response(200, json={"data": [
                {"asset": "eth", "time": "2020-01-05T00:00:00.000000000Z", "PriceUSD": "130"}]})
        return httpx.Response(200, json={
            "data": [{"asset": "btc", "time": "2020-01-05T00:00:00.000000000Z",
                      "PriceUSD": "7400", "CapMrktCurUSD": "1.3e11"}],
            "next_page_url": "https://community-api.coinmetrics.io/v4/x?page=2"})

    with _client(handler) as client:
        out = universe.fetch_universe(client, ["btc", "eth"])
    assert seen[0].params["ignore_unsupported_errors"] == "true"
    assert list(out.columns) == universe.UNIVERSE_COLUMNS
    assert list(out["asset"]) == ["btc", "eth"]
    assert out["metric_date"].dt.tz is None


def test_to_universe_nulls_non_positive_values_and_refuses_duplicates():
    raw = pd.DataFrame({
        "asset": ["a", "b"],
        "metric_date": pd.to_datetime(["2020-01-05", "2020-01-05"], utc=True),
        "PriceUSD": [0.0, 2.0],
        "CapMrktCurUSD": [5.0, -1.0],
        "CapMVRVCur": [0.0, 1.5],
    })
    out = universe.to_universe(raw)
    assert out["price_usd"].isna().tolist() == [True, False]
    assert out["market_cap_usd"].isna().tolist() == [False, True]
    assert out["mvrv"].isna().tolist() == [True, False]
    assert out["active_addresses"].isna().all()  # a metric nobody served is a null column
    with pytest.raises(ValueError, match="duplicate"):
        universe.to_universe(pd.concat([raw, raw]))
```

- [ ] **Step 2: Run, expect FAIL** — `ModuleNotFoundError: cryptoval.factors.universe`.

- [ ] **Step 3: Implement** `cryptoval/factors/universe.py`:

```python
"""The Coin Metrics community universe (design 1-3): every listing serving a daily market cap.

Membership is point-in-time INSIDE the catalog (no asset enters before its data does), but
the catalog itself is chosen ex post by Coin Metrics. That selection bias is declared in the
chapter, not corrected here.
"""

from __future__ import annotations

import httpx
import pandas as pd

from cryptoval.factors.settings import FactorConfig
from cryptoval.sources.base import get_with_retry
from cryptoval.sources.coinmetrics import BASE_URL, MAX_PAGES, parse_timeseries

METRICS = {
    "PriceUSD": "price_usd",
    "CapMrktCurUSD": "market_cap_usd",
    "AdrActCnt": "active_addresses",
    "CapMVRVCur": "mvrv",
}
UNIVERSE_COLUMNS = ["asset", "metric_date", *METRICS.values()]
_POSITIVE = ("price_usd", "market_cap_usd", "mvrv")


def catalog_ids(client: httpx.Client, max_retries: int = 5) -> list[str]:
    payload = get_with_retry(
        client,
        f"{BASE_URL}/catalog-v2/asset-metrics",
        {"metrics": "CapMrktCurUSD", "page_size": 10_000},
        max_retries=max_retries,
    )
    if payload.get("next_page_url"):
        raise RuntimeError("the catalog no longer fits one page; follow next_page_url")
    return sorted(
        entry["asset"]
        for entry in payload["data"]
        if any(
            m["metric"] == "CapMrktCurUSD" and any(f["frequency"] == "1d" for f in m["frequencies"])
            for m in entry["metrics"]
        )
    )


def check_exclusions(config: FactorConfig, ids: list[str]) -> None:
    unknown = sorted(set(config.exclude) - set(ids))
    if unknown:
        # A typo would silently keep the asset it meant to remove.
        raise ValueError(f"excluded ids not in the Coin Metrics catalog: {unknown}")


def to_universe(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.rename(columns=METRICS).reindex(columns=UNIVERSE_COLUMNS)
    out["metric_date"] = pd.to_datetime(out["metric_date"], utc=True).dt.tz_localize(None)
    for col in METRICS.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in _POSITIVE:
        out[col] = out[col].where(out[col] > 0)
    duplicated = out.duplicated(["asset", "metric_date"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicate (asset, metric_date) rows")
    return out.sort_values(["asset", "metric_date"], ignore_index=True)


def fetch_universe(client: httpx.Client, ids: list[str], max_retries: int = 5) -> pd.DataFrame:
    params: dict | None = {
        "assets": ",".join(ids),
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "page_size": 10_000,
        "start_time": "2009-01-01",
        # most listings lack AdrActCnt or CapMVRVCur; without this one gap fails the request
        "ignore_unsupported_errors": "true",
    }
    next_url: str | None = f"{BASE_URL}/timeseries/asset-metrics"
    frames: list[pd.DataFrame] = []
    while next_url:
        if len(frames) >= MAX_PAGES:
            raise RuntimeError(f"pagination exceeded {MAX_PAGES} pages; last url: {next_url}")
        payload = get_with_retry(client, next_url, params, max_retries=max_retries)
        frames.append(parse_timeseries(payload))
        next_url = payload.get("next_page_url")
        params = None  # next_page_url carries the query string
    return to_universe(pd.concat(frames, ignore_index=True))
```

Note: `utc=True` then `tz_localize(None)` drops the zone; `parse_timeseries` already returns UTC-normalised dates.

- [ ] **Step 4: Run, expect PASS** (4 tests).
- [ ] **Step 5: Commit** `feat: fetch the Coin Metrics market-cap universe`.

---

### Task 3: The weekly panel

**Files:**
- Create: `cryptoval/factors/panel.py`
- Test: `tests/test_factors_panel.py`

**Interfaces:**
- Consumes: `UNIVERSE_COLUMNS` frame; `FactorConfig`.
- Produces: `SIGNALS = ("size", "momentum", "value", "mvrv")`; `weekly_panel(daily, config) -> pd.DataFrame` with columns `week, asset, market_cap_usd, ret, size, momentum, value, mvrv`, only eligible rows, only weeks with ≥ `min_assets` rows.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd

from cryptoval.factors.panel import weekly_panel
from cryptoval.factors.settings import FactorConfig

SUNDAYS = pd.date_range("2020-01-05", periods=8, freq="W-SUN")


def _daily(n_assets=4, days=None, price=None):
    days = days if days is not None else pd.date_range("2020-01-05", SUNDAYS[-1], freq="D")
    rows = []
    for i in range(n_assets):
        for k, day in enumerate(days):
            rows.append({
                "asset": f"a{i}", "metric_date": day,
                "price_usd": price(i, k) if price else 10.0 * (1 + i) * 1.01 ** k,
                "market_cap_usd": 1e8 * (1 + i), "active_addresses": 1000.0 * (1 + i), "mvrv": 1.0 + i,
            })
    return pd.DataFrame(rows)


CONFIG = FactorConfig(exclude={}, min_assets=1)


def test_only_sundays_are_used_and_the_return_is_to_the_next_sunday():
    panel = weekly_panel(_daily(), CONFIG)
    assert set(panel["week"]) == set(SUNDAYS[:-1])  # the last Sunday has no t+1
    row = panel[(panel["asset"] == "a0") & (panel["week"] == SUNDAYS[0])].iloc[0]
    assert np.isclose(row["ret"], 1.01**7 - 1)
    assert np.isclose(row["value"], 1000.0 / 1e8)
    assert np.isclose(row["size"], np.log(1e8))


def test_momentum_looks_back_three_weeks():
    panel = weekly_panel(_daily(), CONFIG)
    a0 = panel[panel["asset"] == "a0"].set_index("week")
    assert a0["momentum"].iloc[:3].isna().all()
    assert np.isclose(a0.loc[SUNDAYS[3], "momentum"], 1.01**21 - 1)


def test_a_missing_sunday_never_spans_two_weeks():
    # momentum_weeks=1, so SUNDAYS[5]'s lookback is the absent SUNDAYS[4]: NaN, not the
    # two-week change; SUNDAYS[3]'s t+1 is the absent Sunday, so it has no return.
    days = pd.date_range("2020-01-05", SUNDAYS[-1], freq="D").drop(SUNDAYS[4])
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
```

- [ ] **Step 2: Run, expect FAIL** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `cryptoval/factors/panel.py`:

```python
"""Week-t signals against the week t -> t+1 return (design 4-5).

A week is the observation dated Sunday, on a complete Sunday grid: an absent Sunday is a
NaN cell, so a shift never reaches across it into the week before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.factors.settings import FactorConfig

SIGNALS = ("size", "momentum", "value", "mvrv")
PANEL_COLUMNS = ["week", "asset", "market_cap_usd", "ret", *SIGNALS]


def weekly_panel(daily: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    dates = pd.to_datetime(daily["metric_date"])
    sundays = daily[(dates.dt.dayofweek == 6) & ~daily["asset"].isin(set(config.exclude))]
    if sundays.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    sundays = sundays.assign(metric_date=pd.to_datetime(sundays["metric_date"]))
    grid = pd.date_range(sundays["metric_date"].min(), sundays["metric_date"].max(), freq="W-SUN")

    def wide(column: str) -> pd.DataFrame:
        return sundays.pivot(index="metric_date", columns="asset", values=column).reindex(grid)

    price, cap = wide("price_usd"), wide("market_cap_usd")
    columns = {
        "market_cap_usd": cap,
        "ret": price.shift(-1) / price - 1,
        "size": np.log(cap),
        "momentum": price / price.shift(config.momentum_weeks) - 1,
        "value": wide("active_addresses") / cap,
        "mvrv": wide("mvrv"),
    }
    long = pd.concat({k: v.stack() for k, v in columns.items()}, axis=1)
    long.index = long.index.set_names(["week", "asset"])
    long = long.reset_index()
    eligible = long[(long["market_cap_usd"] >= config.min_market_cap_usd) & long["ret"].notna()]
    counts = eligible.groupby("week")["asset"].transform("size")
    return eligible[counts >= config.min_assets].reset_index(drop=True)[PANEL_COLUMNS]
```

`price.shift(-1)/price` is NaN when either Sunday is absent, which also enforces "price present at t".

- [ ] **Step 4: Run, expect PASS** (4 tests).
- [ ] **Step 5: Commit** `feat: the weekly factor panel on a complete Sunday grid`.

---

### Task 4: Portfolios and inference

**Files:**
- Create: `cryptoval/factors/portfolios.py`, `cryptoval/factors/inference.py`
- Test: `tests/test_factors_portfolios.py`

**Interfaces:**
- Consumes: `weekly_panel` output; `FactorConfig`; `Unusable` from `cryptoval.econometrics.base`.
- Produces:
  - `decile_returns(panel, signal, config) -> DataFrame` index week, columns `D1..D10, HL, n_assets`.
  - `ltw_factors(panel, config) -> DataFrame` index week, columns `CMKT, CSMB, CMOM`.
  - `MIN_WEEKS = 52`; `nw_mean(series, lags) -> {"mean_return", "t_stat", "p_value", "n_weeks"}`; `nw_alpha(portfolio, factors, lags) -> {"alpha", "t_stat", "p_value", "n_weeks", "beta_cmkt", "beta_csmb", "beta_cmom"}`; `fama_macbeth(panel, signals, config) -> DataFrame` index week, one slope column per signal. The `nw_*` functions raise `Unusable` below `MIN_WEEKS`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.factors.inference import fama_macbeth, nw_alpha, nw_mean
from cryptoval.factors.portfolios import decile_returns, ltw_factors
from cryptoval.factors.settings import FactorConfig

CONFIG = FactorConfig(exclude={}, min_assets=50)


def _week(n=50, week="2020-01-05", seed=0):
    rng = np.random.default_rng(seed)
    signal = np.arange(n, dtype=float)
    return pd.DataFrame({
        "week": pd.Timestamp(week), "asset": [f"a{i:02d}" for i in range(n)],
        "market_cap_usd": rng.uniform(1e6, 1e9, n), "ret": signal / 100.0,
        "size": signal, "momentum": signal, "value": signal, "mvrv": signal,
    })


def test_deciles_are_equal_count_value_weighted_and_hl_is_d10_minus_d1():
    week = _week()
    out = decile_returns(week, "value", CONFIG).iloc[0]
    top = week.tail(5)
    assert np.isclose(out["D10"], (top["ret"] * top["market_cap_usd"]).sum() / top["market_cap_usd"].sum())
    assert np.isclose(out["HL"], out["D10"] - out["D1"])
    assert out["n_assets"] == 50


def test_a_week_below_min_assets_is_skipped():
    assert decile_returns(_week(n=49), "value", CONFIG).empty


def test_ties_are_broken_by_asset_id():
    week = _week().assign(mvrv=1.0)
    a = decile_returns(week, "mvrv", CONFIG)
    b = decile_returns(week.sample(frac=1.0, random_state=3), "mvrv", CONFIG)
    pd.testing.assert_frame_equal(a, b)


def test_ltw_signs():
    week = _week()
    out = ltw_factors(week, CONFIG).iloc[0]
    assert out["CMOM"] > 0  # high momentum earned more
    assert out["CSMB"] < 0  # small minus big; small earned less here
    w = week["market_cap_usd"]
    assert np.isclose(out["CMKT"], (week["ret"] * w).sum() / w.sum())


def test_nw_mean_detects_a_premium_and_not_noise():
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.normal(0, 0.1, 500))
    assert nw_mean(noise, 4)["p_value"] > 0.05
    assert nw_mean(noise + 0.05, 4)["p_value"] < 1e-6
    with pytest.raises(Unusable, match="51 weeks"):
        nw_mean(noise.iloc[:51], 4)


def test_nw_alpha_recovers_alpha_and_beta():
    rng = np.random.default_rng(1)
    factors = pd.DataFrame(rng.normal(0, 0.1, (500, 3)), columns=["CMKT", "CSMB", "CMOM"])
    portfolio = 0.02 + 1.5 * factors["CMKT"] + rng.normal(0, 0.01, 500)
    fit = nw_alpha(portfolio, factors, 4)
    assert fit["alpha"] == pytest.approx(0.02, abs=0.003)
    assert fit["beta_cmkt"] == pytest.approx(1.5, abs=0.02)
    assert fit["p_value"] < 1e-6


def test_fama_macbeth_slope_is_the_rank_spread():
    week = _week().assign(size=0.0, momentum=0.0, mvrv=0.0)  # only value varies
    week[["size", "momentum", "mvrv"]] = np.random.default_rng(2).normal(size=(50, 3))
    week["ret"] = week["value"].rank(pct=True)  # winsorizing at 5/95 trims the ends a little
    slopes = fama_macbeth(week, ("value",), CONFIG)
    assert slopes["value"].iloc[0] == pytest.approx(0.9, abs=0.05)


def test_fama_macbeth_winsorizes_an_outlier():
    week = _week().assign(ret=0.0)
    week.loc[0, "ret"] = 9.0  # one +900% micro-cap at the bottom of the value rank
    slope = fama_macbeth(week, ("value",), CONFIG)["value"].iloc[0]
    assert abs(slope) < 0.01
```

- [ ] **Step 2: Run, expect FAIL** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`cryptoval/factors/portfolios.py`:

```python
"""Value-weighted decile sorts and the Liu-Tsyvinski-Wu factors (design 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.factors.settings import FactorConfig


def _vw(group: pd.DataFrame) -> float:
    weight = group["market_cap_usd"]
    return float((group["ret"] * weight).sum() / weight.sum())


def _sorted(week: pd.DataFrame, signal: str) -> pd.DataFrame:
    # asset id breaks ties, so the sort never depends on row order
    return week.dropna(subset=[signal]).sort_values([signal, "asset"], kind="stable")


def decile_returns(panel: pd.DataFrame, signal: str, config: FactorConfig) -> pd.DataFrame:
    names = [f"D{b + 1}" for b in range(config.deciles)]
    rows = {}
    for week, group in panel.groupby("week"):
        ranked = _sorted(group, signal)
        n = len(ranked)
        if n < config.min_assets:
            continue
        bucket = np.arange(n) * config.deciles // n
        rows[week] = {name: _vw(ranked[bucket == b]) for b, name in enumerate(names)}
        rows[week]["n_assets"] = float(n)
    out = pd.DataFrame.from_dict(rows, orient="index", columns=[*names, "n_assets"])
    out.insert(config.deciles, "HL", out[names[-1]] - out[names[0]])
    return out


def ltw_factors(panel: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    """Single 30/40/30 sorts, not LTW's 2x3 (design 6: 50-110 assets make 2x3 legs too thin)."""

    def spread(group: pd.DataFrame, signal: str) -> float:
        ranked = _sorted(group, signal)
        if len(ranked) < config.min_assets:
            return np.nan
        k = int(len(ranked) * config.tail_share)
        return _vw(ranked.tail(k)) - _vw(ranked.head(k))

    rows = {
        week: {
            "CMKT": _vw(group),
            "CSMB": -spread(group, "size"),
            "CMOM": spread(group, "momentum"),
        }
        for week, group in panel.groupby("week")
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["CMKT", "CSMB", "CMOM"])
```

`cryptoval/factors/inference.py`:

```python
"""Newey-West means, factor alphas and Fama-MacBeth slopes (design 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cryptoval.econometrics.base import Unusable
from cryptoval.factors.settings import FactorConfig

MIN_WEEKS = 52


def _hac(y: np.ndarray, x: np.ndarray, lags: int):
    if len(y) < MIN_WEEKS:
        raise Unusable(f"{len(y)} weeks, {MIN_WEEKS} needed")
    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def nw_mean(series: pd.Series, lags: int) -> dict[str, float]:
    y = series.dropna().to_numpy(dtype="float64")
    fit = _hac(y, np.ones((len(y), 1)), lags)
    return {
        "mean_return": float(fit.params[0]),
        "t_stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
        "n_weeks": float(len(y)),
    }


def nw_alpha(portfolio: pd.Series, factors: pd.DataFrame, lags: int) -> dict[str, float]:
    data = pd.concat([portfolio.rename("_y"), factors], axis=1).dropna()
    x = sm.add_constant(data[factors.columns].to_numpy(dtype="float64"), has_constant="add")
    fit = _hac(data["_y"].to_numpy(dtype="float64"), x, lags)
    out = {
        "alpha": float(fit.params[0]),
        "t_stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
        "n_weeks": float(len(data)),
    }
    for i, name in enumerate(factors.columns, start=1):
        out[f"beta_{name.lower()}"] = float(fit.params[i])
    return out


def fama_macbeth(panel: pd.DataFrame, signals: tuple[str, ...], config: FactorConfig) -> pd.DataFrame:
    """Weekly OLS of winsorized returns on percentile ranks in [0, 1]: a slope is the
    top-minus-bottom return spread. The winsorizing stops one micro-cap setting the slope."""
    rows = {}
    for week, group in panel.dropna(subset=list(signals)).groupby("week"):
        n = len(group)
        if n < config.min_assets:
            continue
        lo, hi = group["ret"].quantile([config.winsor_quantile, 1 - config.winsor_quantile])
        y = group["ret"].clip(lo, hi).to_numpy(dtype="float64")
        ranks = (group[list(signals)].rank() - 1) / (n - 1)
        x = np.column_stack([np.ones(n), ranks.to_numpy(dtype="float64")])
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        rows[week] = dict(zip(signals, coef[1:], strict=True))
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(signals))
```

- [ ] **Step 4: Run, expect PASS** (9 tests).
- [ ] **Step 5: Commit** `feat: decile sorts, LTW factors, Newey-West alphas and Fama-MacBeth`.

---

### Task 5: Runner, CLI, dbt, Makefile

**Files:**
- Create: `cryptoval/factors/runner.py`, `dbt/models/marts/fct_factor_test.sql`, `dbt/models/marts/fct_factor_returns.sql`, `tests/test_factors_runner.py`
- Modify: `cryptoval/cli.py` (commands), `dbt/models/marts/_marts.yml`, `tests/test_marts_contract.py`, `tests/test_cli.py`, `Makefile`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: `FACTOR_COLUMNS`; `run_factors(daily, config, panel_vintage_ts=None) -> (table, returns)`; `cli.run_factors_fetch(settings) -> dict`; `cli.run_factors_cmd(settings) -> dict`; silver `cm_universe_daily` (UNIVERSE_COLUMNS + `fetched_at`); gold `factor_test`, `factor_returns` (`week, series, value`).

- [ ] **Step 1: Write the failing tests** `tests/test_factors_runner.py`:

```python
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
        frames.append(pd.DataFrame({
            "asset": f"a{i:02d}", "metric_date": days, "price_usd": price,
            "market_cap_usd": price * 1e7 * (1 + i), "active_addresses": 1e4 * rng.uniform(1, 2),
            "mvrv": rng.uniform(0.5, 3.0, weeks),
        }))
    return pd.concat(frames, ignore_index=True)


def test_thirteen_hypotheses_one_holm_family():
    table, returns = run_factors(_universe(), CONFIG)
    assert list(table.columns) == FACTOR_COLUMNS
    assert (table["metric"] == "refused").sum() == 0
    count = table[table["metric"] == "hypotheses_tested"]["value"].item()
    assert count == 13
    assert (table["metric"] == "p_value_holm").sum() == 13
    assert not table.duplicated(["model_id", "signal", "portfolio", "metric"]).any()
    assert {"HL_value", "CMKT"} <= set(returns["series"])


def test_an_empty_universe_refuses_every_test():
    empty = _universe().iloc[0:0]
    table, returns = run_factors(empty, CONFIG)
    assert (table["metric"] == "refused").sum() == 13
    assert table[table["metric"] == "hypotheses_tested"]["value"].item() == 0
    assert returns.empty
```

Add to `tests/test_marts_contract.py`:

```python
from cryptoval.factors.runner import FACTOR_COLUMNS


def test_the_factor_mart_selects_exactly_factor_columns():
    assert _selected_columns("dbt/models/marts/fct_factor_test.sql") == FACTOR_COLUMNS
```

Add to `tests/test_cli.py`:

```python
def test_factors_fetch_then_run_writes_both_gold_tables(tmp_path, monkeypatch):
    from test_factors_runner import _universe

    settings = Settings(data_root=str(tmp_path))
    monkeypatch.setattr(cli, "catalog_ids", lambda client, max_retries=5: ["a00", "usdt"])
    monkeypatch.setattr(cli, "fetch_universe", lambda client, ids, max_retries=5: _universe())
    monkeypatch.setattr(cli, "load_factor_config",
                        lambda: FactorConfig(exclude={"usdt": "pegged"}, min_assets=50))
    assert cli.run_factors_fetch(settings)["listings"] == 2
    report = cli.run_factors_cmd(settings)
    assert report["fama_macbeth"]["rows"] > 0
    gold = pathlib.Path(settings.resolver.table("gold", "factor_test")) / "part-0.parquet"
    assert pd.read_parquet(gold)["panel_vintage_ts"].notna().all()


def test_factors_run_without_a_fetch_says_so(tmp_path):
    assert "factors fetch" in cli.run_factors_cmd(Settings(data_root=str(tmp_path)))["status"]
```

(Import `FactorConfig` at the top of `tests/test_cli.py` alongside the existing imports; `test_factors_runner` resolves under bare pytest the same way `dashboard_fixture` does.)

- [ ] **Step 2: Run, expect FAIL** — `ModuleNotFoundError: cryptoval.factors.runner`.

- [ ] **Step 3: Implement**

`cryptoval/factors/runner.py`:

```python
"""One long table for spec 11.5, at grain (model_id, signal, portfolio, metric).

ONE Holm family of 13: four decile HL means, three LTW factor means, two factor alphas, four
Fama-MacBeth slopes. A test that cannot run is a refused ROW carrying the reason, never an
absent one (the fct_volatility_test contract).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pandas as pd

from cryptoval.econometrics.base import Unusable
from cryptoval.factors.inference import fama_macbeth, nw_alpha, nw_mean
from cryptoval.factors.panel import SIGNALS, weekly_panel
from cryptoval.factors.portfolios import decile_returns, ltw_factors
from cryptoval.factors.settings import FactorConfig
from cryptoval.hashing import stable_hash
from cryptoval.scoring.multiplicity import correct_family_wise

FACTOR_COLUMNS = [
    "model_id",
    "signal",
    "portfolio",
    "metric",
    "value",
    "n_weeks",
    "reason",
    "params_hash",
    "panel_vintage_ts",
]
COUNT_KEYS = {"model_id": "*", "signal": "*"}
ALPHA_SIGNALS = ("value", "mvrv")  # size and momentum ARE factors, so they get no alpha


def run_factors(
    daily: pd.DataFrame, config: FactorConfig, panel_vintage_ts: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = {"params_hash": stable_hash(dataclasses.asdict(config)), "panel_vintage_ts": panel_vintage_ts}
    rows: list[dict[str, Any]] = []
    series: dict[str, pd.Series] = {}

    def row(**values: Any) -> None:
        out = dict.fromkeys(FACTOR_COLUMNS)
        out.update(common, **values)
        rows.append(out)

    def test(model_id: str, signal: str, portfolio: str | None, compute: Callable[[], dict]) -> None:
        try:
            result = compute()
        except Unusable as exc:
            row(model_id=model_id, signal=signal, portfolio=portfolio, metric="refused", reason=str(exc))
            return
        n = result.pop("n_weeks")
        for metric, value in result.items():
            row(model_id=model_id, signal=signal, portfolio=portfolio, metric=metric, value=value, n_weeks=n)

    lags = config.newey_west_lags
    panel = weekly_panel(daily, config)
    deciles = {s: decile_returns(panel, s, config) for s in SIGNALS}
    for signal, table in deciles.items():
        n = float(len(table))
        for name in table.columns.drop(["HL", "n_assets"]):
            row(model_id="decile", signal=signal, portfolio=name, metric="mean_return",
                value=float(table[name].mean()) if n else None, n_weeks=n)
        row(model_id="decile", signal=signal, metric="mean_assets",
            value=float(table["n_assets"].mean()) if n else None, n_weeks=n)
        test("decile", signal, "HL", lambda t=table: nw_mean(t["HL"], lags))
        series[f"HL_{signal}"] = table["HL"]

    factors = ltw_factors(panel, config)
    for name in factors.columns:
        test("factor", name, None, lambda f=factors[name]: nw_mean(f, lags))
        series[name] = factors[name]
    for signal in ALPHA_SIGNALS:
        test("alpha", signal, "HL", lambda s=signal: nw_alpha(deciles[s]["HL"], factors, lags))

    slopes = fama_macbeth(panel, SIGNALS, config)
    for signal in SIGNALS:
        test("fama_macbeth", signal, None, lambda s=signal: nw_mean(slopes[s], lags))

    table = correct_family_wise(
        pd.DataFrame(rows, columns=FACTOR_COLUMNS),
        p_metric="p_value",
        columns=FACTOR_COLUMNS,
        count_keys={**COUNT_KEYS, **common},
    )
    returns = pd.concat(
        [s.dropna().rename("value").rename_axis("week").reset_index().assign(series=k) for k, s in series.items()],
        ignore_index=True,
    )[["week", "series", "value"]]
    return table, returns
```

(When `series` values are all empty, `pd.concat` of empty frames still yields the three columns.)

`cryptoval/cli.py` — imports at the top with the other package imports:

```python
from cryptoval.factors.runner import run_factors
from cryptoval.factors.settings import load_factor_config
from cryptoval.factors.universe import catalog_ids, check_exclusions, fetch_universe
```

and beside `run_vol`:

```python
def run_factors_fetch(settings: Settings) -> dict:
    """The Coin Metrics market-cap universe to silver, replaced whole (design 3)."""
    config = load_factor_config()
    with httpx.Client(timeout=settings.request_timeout_s) as client:
        ids = catalog_ids(client, max_retries=settings.max_retries)
        check_exclusions(config, ids)
        daily = fetch_universe(client, ids, max_retries=settings.max_retries)
    daily["fetched_at"] = pd.Timestamp.now(tz="UTC")
    write_silver(settings.resolver, "cm_universe_daily", daily)
    return {"listings": len(ids), "rows": len(daily), "through": str(daily["metric_date"].max())}


def run_factors_cmd(settings: Settings) -> dict:
    """Spec 11.5 over the fetched universe, full sample, written to gold."""
    daily = _read_silver(settings, "cm_universe_daily")
    if daily.empty:
        return {"status": "no universe in silver; run `cryptoval factors fetch` first"}
    table, returns = run_factors(daily, load_factor_config(), daily["fetched_at"].max())
    write_gold(settings.resolver, "factor_test", table)
    write_gold(settings.resolver, "factor_returns", returns)
    report: dict[str, dict] = {}
    for model_id, group in table[table["model_id"] != "*"].groupby("model_id"):
        holm = group[(group["metric"] == "p_value_holm") & (group["value"] < 0.05)]
        report[str(model_id)] = {
            "rows": int(len(group)),
            "refusals": int((group["metric"] == "refused").sum()),
            "significant_after_holm": int(len(holm)),
        }
    return report
```

In `main`: `fc = sub.add_parser("factors"); fc.add_argument("action", choices=["fetch", "run"])`, and

```python
    if args.command == "factors":
        if args.action == "fetch":
            print(run_factors_fetch(settings))
        else:
            for key, outcome in sorted(run_factors_cmd(settings).items()):
                print(key, outcome)
        # No priced signal is a result, not a failed run.
        return 0
```

`dbt/models/marts/fct_factor_test.sql`:

```sql
-- Spec 11.5 at grain (model_id, signal, portfolio, metric), on the Coin Metrics community
-- universe (design 2026-10-21-factor-models). Weekly, full sample, descriptive: the signals
-- and floors were chosen in 2026 with the literature in hand, so no tradability claim.
--
-- portfolio is D1..D10 or HL on decile and alpha rows, NULL otherwise. model_id = '*' holds
-- the hypothesis count: the family is every 'p_value' row (13 when nothing is refused).
select
    model_id,
    signal,
    portfolio,
    metric,
    value,
    n_weeks,
    reason,
    params_hash,
    panel_vintage_ts
from read_parquet('{{ var("data_root") }}/gold/factor_test/*.parquet')
```

`dbt/models/marts/fct_factor_returns.sql`:

```sql
-- The weekly series behind fct_factor_test: HL_<signal> decile spreads and the LTW
-- factors CMKT, CSMB, CMOM, each the week-t sort's return over week t -> t+1.
select
    week,
    series,
    value
from read_parquet('{{ var("data_root") }}/gold/factor_returns/*.parquet')
```

`_marts.yml`, after `fct_volatility_test`:

```yaml
  - name: fct_factor_test
    description: >
      Spec 11.5 cross-sectional asset pricing on the Coin Metrics community universe:
      value-weighted decile sorts, the Liu-Tsyvinski-Wu factors, factor alphas and
      Fama-MacBeth slopes, one Holm family. Weekly, full sample, descriptive.
    tests:
      - dbt_utils.unique_combination_of_columns:
          arguments:
            combination_of_columns: [model_id, signal, portfolio, metric]
    columns:
      - name: model_id
        tests: [not_null]
      - name: metric
        tests: [not_null]
      - name: params_hash
        tests: [not_null]
  - name: fct_factor_returns
    description: The weekly decile spreads and LTW factor returns behind fct_factor_test.
    tests:
      - dbt_utils.unique_combination_of_columns:
          arguments:
            combination_of_columns: [week, series]
```

`Makefile`: add `factors` to `.PHONY`; target

```make
factors:
	$(PY) -m cryptoval.cli factors fetch
	$(PY) -m cryptoval.cli factors run
```

and in `all`, `$(MAKE) factors` after `$(MAKE) vol`.

- [ ] **Step 4: Run, expect PASS** — `.venv/bin/python -m pytest tests/test_factors_runner.py tests/test_marts_contract.py tests/test_cli.py -q`, then the whole suite and `make lint`.
- [ ] **Step 5: Commit** `feat: the factor runner, CLI, marts and make target`.

---

### Task 6: Live run and the chapter

**Files:**
- Create: `reports/do-valuation-signals-earn-returns.md`
- Modify: `reports/dissertation-comparison.md` (scorecard row and the remaining-work line), `README.md` (chapter list, if it lists chapters)

- [ ] **Step 1:** `make factors`. Expected: `listings 135`-ish, then a report per model_id with 0 refusals.
- [ ] **Step 2:** `cd dbt && ../.venv/bin/dbt build --profiles-dir . --select fct_factor_test fct_factor_returns` → PASS.
- [ ] **Step 3:** Query the marts for every number in the chapter (decile means, HL t and Holm p, factor premia, alphas and betas, FM slopes, sample weeks and mean cross-section).
- [ ] **Step 4:** Write the chapter: the three biases of design §1 first (selection, XRP's supply, delisting), then the tables, then the verdict against §11.5's question, then the disclaimer. Add the scorecard row.
- [ ] **Step 5: Commit** `docs: do valuation signals earn returns -- the 11.5 chapter`.
