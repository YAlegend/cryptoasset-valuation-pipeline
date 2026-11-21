# The Volatility Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GARCH(1,1)/EGARCH, Markov-switching and bivariate DCC-GARCH over the panel, one Holm family
over the hypotheses that test the dissertation's §6.3 and §8.4 claims, written to
`fct_volatility_test`, and a chapter reporting what the live run found.

**Architecture:** A new `cryptoval/volatility/` package layered like `cryptoval/econometrics/`:
pure estimators (`garch`, `exhaustion`, `regimes`, `dcc`) over a returns module, one runner producing
a long table with refusals as rows, `correct_family_wise` reused for Holm. `cryptoval.cli vol`
writes gold; a dbt mart reads it; `make all` runs it after `econ`.

**Tech Stack:** Python 3.14, pandas, numpy, `arch==8.0.0` (new), statsmodels 0.15 (`het_arch`,
`MarkovRegression`), scipy (`minimize`, `fisher_exact`, `chi2`), dbt-duckdb.

**Spec:** `docs/superpowers/specs/2026-10-18-volatility-chain-design.md`

## Global Constraints

- One new dependency only: `arch==8.0.0` pinned in `requirements.txt`. No DCC package.
- Daily log returns of `price_usd` **in percent** (×100).
- Student-t innovations for every GARCH/EGARCH fit.
- `MIN_RETURNS = 730`; `MIN_WINDOW_ROWS = 20`.
- Stablecoins (`is_stablecoin: true`) are refused, not modelled.
- Holm family = every row with `metric == "p_value"`: ARCH-LM, EGARCH γ, §6.3 exhaustion, Engle-Sheppard. Nothing else.
- `TEST_COLUMNS = [model_id, symbol, metric, value, n_rows, window_name, reason, params_hash, panel_vintage_ts]`, asserted against the mart SQL.
- Full sample, descriptive; the chapter states no tradability claim and no comparison to Plan 4's null.
- Run tests from the worktree root with `.venv/bin/python -m pytest` so the worktree's `cryptoval` is imported.

## Review Focus

1. **An asset whose price starts late** (ETH before 2015-07-30 has NULL `price_usd`) must be modelled from its first price, not refused for a "gap". → `test_leading_nulls_are_not_a_gap` (Task 1).
2. **A dissertation window outside an asset's history** (SOL in 2021) must produce a NaN row with the row count in `reason`, not an exception and not a missing row. → `test_a_window_outside_the_history_is_a_nan_row_with_its_count` (Task 6).
3. **One asset's optimizer failing** must become a refusal row and leave every other asset's rows intact. → `test_a_garch_that_fails_to_converge_is_a_refusal_and_the_run_continues` (Task 6).
4. **A crypto-traditional pair where the equity history predates the crypto** (AAPL from 2015, AAVE from 2020) must use the overlap, not refuse. → `test_trading_calendar_uses_only_the_overlap` (Task 1).
5. **A pair whose anchor is unusable** (BTC absent) must refuse every pair with the anchor named, not raise `KeyError`. → `test_a_missing_anchor_refuses_its_pairs_by_name` (Task 6).

---

### Task 1: Dependency, config, and returns

**Files:**
- Modify: `requirements.txt` (add `arch==8.0.0`)
- Create: `cryptoval/volatility/__init__.py`, `cryptoval/volatility/settings.py`, `cryptoval/volatility/returns.py`, `config/volatility.yml`
- Test: `tests/test_vol_returns.py`

**Interfaces:**
- Consumes: `cryptoval.econometrics.base.Unusable`; `cryptoval.scoring.targets._require_contiguous_days(panel: DataFrame) -> None` (raises `ValueError`).
- Produces: `VolConfig` (frozen dataclass: `windows: dict[str, tuple[date, date]]`, `peak_window: str`, `crypto_anchors: tuple[str, ...]`, `traditional_anchors: tuple[str, ...]`, `arch_lm_lags=7`, `exhaustion_block_days=30`, `engle_sheppard_lags=5`, `markov_search_reps=20`); `load_vol_config(path="config/volatility.yml") -> VolConfig`; `MIN_RETURNS = 730`; `price_series(panel, symbol) -> pd.Series`; `crypto_returns(prices, symbol) -> pd.Series`; `pair_returns(left, right, label, *, trading_calendar: bool) -> pd.DataFrame` with columns `a`, `b`.

- [ ] **Step 1: Add the dependency** — append `arch==8.0.0` to `requirements.txt` after `statsmodels==0.15.0`; `.venv/bin/pip install arch==8.0.0`.

- [ ] **Step 2: Write the failing tests** — `tests/test_vol_returns.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.volatility.returns import MIN_RETURNS, crypto_returns, pair_returns, price_series
from cryptoval.volatility.settings import load_vol_config


def _panel(symbol, start, days, gap_at=None, leading_nulls=0):
    dates = pd.date_range(start, periods=days, freq="D")
    price = np.exp(np.cumsum(np.full(days, 0.001))) * 100.0
    frame = pd.DataFrame({"symbol": symbol, "metric_date": dates, "price_usd": price})
    if leading_nulls:
        frame.loc[: leading_nulls - 1, "price_usd"] = np.nan
    if gap_at is not None:
        frame.loc[gap_at, "price_usd"] = np.nan
    return frame


def test_returns_are_percent_log_returns():
    returns = crypto_returns(price_series(_panel("BTC", "2019-01-01", 800), "BTC"), "BTC")
    assert len(returns) == 799
    assert returns.iloc[0] == pytest.approx(0.1)


def test_leading_nulls_are_not_a_gap():
    returns = crypto_returns(
        price_series(_panel("ETH", "2015-01-01", 1000, leading_nulls=200), "ETH"), "ETH"
    )
    assert len(returns) == 799


def test_an_interior_gap_is_refused_with_its_date():
    prices = price_series(_panel("BTC", "2019-01-01", 1000, gap_at=500), "BTC")
    with pytest.raises(Unusable, match="2-day step"):
        crypto_returns(prices, "BTC")


def test_a_short_history_is_refused_with_its_count():
    prices = price_series(_panel("SOL", "2025-10-14", 363), "SOL")
    with pytest.raises(Unusable, match=f"362 returns, {MIN_RETURNS} needed"):
        crypto_returns(prices, "SOL")


def test_an_absent_symbol_is_refused():
    with pytest.raises(Unusable, match="XRP is not in the panel"):
        price_series(_panel("BTC", "2019-01-01", 10), "XRP")


def test_trading_calendar_returns_span_the_weekend():
    crypto = price_series(_panel("BTC", "2019-01-01", 1400), "BTC")
    days = pd.bdate_range("2019-01-01", "2022-10-31")
    equity = pd.Series(np.linspace(100.0, 200.0, len(days)), index=days)
    out = pair_returns(crypto, equity, "BTC|AAPL", trading_calendar=True)
    friday, monday = pd.Timestamp("2019-01-04"), pd.Timestamp("2019-01-07")
    assert out.loc[monday, "a"] == pytest.approx(100.0 * np.log(crypto[monday] / crypto[friday]))


def test_trading_calendar_uses_only_the_overlap():
    crypto = price_series(_panel("AAVE", "2020-10-01", 1500), "AAVE")
    days = pd.bdate_range("2015-01-01", "2026-10-16")
    equity = pd.Series(100.0 + np.arange(len(days)), index=days, dtype="float64")
    out = pair_returns(crypto, equity, "AAVE|AAPL", trading_calendar=True)
    assert out.index.min() > pd.Timestamp("2020-10-01")


def test_trading_calendar_refuses_a_missing_crypto_price():
    crypto = price_series(_panel("BTC", "2019-01-01", 1400, gap_at=700), "BTC")
    days = pd.bdate_range("2019-01-01", "2022-10-31")
    equity = pd.Series(100.0, index=days)
    with pytest.raises(Unusable, match="no crypto price on 1 trading date"):
        pair_returns(crypto, equity, "BTC|AAPL", trading_calendar=True)


def test_daily_pairs_use_the_intersection():
    btc = price_series(_panel("BTC", "2019-01-01", 1400), "BTC")
    eth = price_series(_panel("ETH", "2020-01-01", 1000), "ETH")
    out = pair_returns(btc, eth, "BTC|ETH", trading_calendar=False)
    assert list(out.columns) == ["a", "b"]
    assert len(out) == 999


def test_the_shipped_config_names_the_dissertation_windows():
    config = load_vol_config()
    assert config.windows["peak_oct21_nov21"] == (dt.date(2021, 10, 1), dt.date(2021, 11, 30))
    assert config.peak_window == "peak_oct21_nov21"
    assert len(config.windows) == 8


def test_a_window_that_ends_before_it_starts_is_refused(tmp_path):
    path = tmp_path / "v.yml"
    path.write_text(
        "windows: {w: [2022-01-02, 2022-01-01]}\npeak_window: w\n"
        "crypto_anchors: [BTC]\ntraditional_anchors: [BTC]\n"
    )
    with pytest.raises(ValueError, match="before start"):
        load_vol_config(str(path))
```

- [ ] **Step 3: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_vol_returns.py -v` → FAIL, `ModuleNotFoundError: cryptoval.volatility`.

- [ ] **Step 4: Implement.** `cryptoval/volatility/__init__.py`:

```python
"""The volatility chain (spec 11.4's second half): GARCH/EGARCH, Markov-switching, DCC.

Full sample and descriptive, like the anchoring chain: parameters are estimated over the
whole history, so nothing here is out of sample or supports a tradability claim.
"""
```

`config/volatility.yml`:

```yaml
# The dissertation's correlation windows (Tables 3-10), named as it names them, and the
# anchors every pair is built on. Design 7-8.
windows:
  full_2018_2022: [2018-01-01, 2022-11-30]       # Table 3
  jan21_jan22: [2021-01-01, 2022-01-31]          # Table 4
  feb22_may22: [2022-02-01, 2022-05-31]          # Table 5
  jun22_nov22: [2022-06-01, 2022-11-30]          # Table 6
  peak_oct21_nov21: [2021-10-01, 2021-11-30]     # Table 7, the bull-market peak
  equities_2020_2022: [2020-01-01, 2022-11-30]   # Table 8
  nft_2020_2021: [2020-01-01, 2021-06-30]        # Table 9
  nft_2021_2022: [2021-07-01, 2022-11-30]        # Table 10
peak_window: peak_oct21_nov21
crypto_anchors: [BTC]
traditional_anchors: [BTC, ETH]
arch_lm_lags: 7
exhaustion_block_days: 30
engle_sheppard_lags: 5
markov_search_reps: 20
```

`cryptoval/volatility/settings.py`:

```python
"""The volatility chain's configuration (design 8): windows and anchors are data."""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib

import yaml

_INTS = ("arch_lm_lags", "exhaustion_block_days", "engle_sheppard_lags", "markov_search_reps")


@dataclasses.dataclass(frozen=True)
class VolConfig:
    windows: dict[str, tuple[dt.date, dt.date]]
    peak_window: str
    crypto_anchors: tuple[str, ...]
    traditional_anchors: tuple[str, ...]
    arch_lm_lags: int = 7
    exhaustion_block_days: int = 30
    engle_sheppard_lags: int = 5
    markov_search_reps: int = 20


def load_vol_config(path: str = "config/volatility.yml") -> VolConfig:
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    windows: dict[str, tuple[dt.date, dt.date]] = {}
    for name, (start, end) in raw["windows"].items():
        start, end = dt.date.fromisoformat(str(start)), dt.date.fromisoformat(str(end))
        if end < start:
            raise ValueError(f"window {name}: end {end} is before start {start}")
        windows[name] = (start, end)
    if raw["peak_window"] not in windows:
        raise ValueError(f"peak_window {raw['peak_window']!r} is not one of {sorted(windows)}")
    return VolConfig(
        windows=windows,
        peak_window=raw["peak_window"],
        crypto_anchors=tuple(raw["crypto_anchors"]),
        traditional_anchors=tuple(raw["traditional_anchors"]),
        **{key: int(raw[key]) for key in _INTS if key in raw},
    )
```

`cryptoval/volatility/returns.py`:

```python
"""Percent log returns (design 3), and the calendar each pair is measured on.

Crypto trades every day, so a crypto series and a crypto-crypto pair are guarded for a
contiguous daily grid. A crypto-traditional pair is measured on the TRADITIONAL calendar:
the crypto price is sampled on the equity's trading dates, and both returns span the same
gaps (a Monday return covers Friday to Monday for both legs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.econometrics.base import Unusable
from cryptoval.scoring.targets import _require_contiguous_days

# Two years, the anchoring chain's floor. Measured on the 2026-10-18 vintage: SOL (362
# returns) and TRX (363) fall below it; every other non-stablecoin clears it.
MIN_RETURNS = 730


def price_series(panel: pd.DataFrame, symbol: str) -> pd.Series:
    """Positive prices indexed by date. NULL and non-positive prices are dropped, so an
    asset whose history starts late begins at its first price rather than at a gap."""
    rows = panel[panel["symbol"] == symbol]
    if rows.empty:
        raise Unusable(f"{symbol} is not in the panel")
    values = pd.to_numeric(rows["price_usd"], errors="coerce").astype("float64").to_numpy()
    series = pd.Series(values, index=pd.DatetimeIndex(pd.to_datetime(rows["metric_date"])))
    series = series.sort_index()
    series = series[series > 0.0]
    if series.empty:
        raise Unusable(f"{symbol} has no positive price")
    return series


def _contiguous(index: pd.DatetimeIndex, label: str) -> None:
    try:
        _require_contiguous_days(pd.DataFrame({"symbol": label, "metric_date": index}))
    except ValueError as exc:
        raise Unusable(str(exc)) from exc


def _enough(n: int, label: str) -> None:
    if n < MIN_RETURNS:
        raise Unusable(
            f"{label}: {n} returns, {MIN_RETURNS} needed for a volatility model on daily data"
        )


def crypto_returns(prices: pd.Series, symbol: str) -> pd.Series:
    _contiguous(prices.index, symbol)
    returns = 100.0 * np.log(prices).diff().dropna()
    _enough(len(returns), symbol)
    return returns


def pair_returns(
    left: pd.Series, right: pd.Series, label: str, *, trading_calendar: bool
) -> pd.DataFrame:
    """Aligned percent log returns, columns ``a`` (left) and ``b`` (right)."""
    if trading_calendar:
        inside = (right.index >= left.index.min()) & (right.index <= left.index.max())
        dates = right.index[inside]
        sampled = left.reindex(dates)
        missing = sampled.index[sampled.isna()]
        if len(missing):
            raise Unusable(
                f"{label}: no crypto price on {len(missing)} trading date(s), "
                f"first {missing[0]:%Y-%m-%d}"
            )
        prices = pd.DataFrame({"a": sampled, "b": right.reindex(dates)})
    else:
        prices = pd.concat({"a": left, "b": right}, axis=1, join="inner")
        _contiguous(prices.index, label)
    returns = 100.0 * np.log(prices).diff().dropna()
    _enough(len(returns), label)
    return returns
```

- [ ] **Step 5: Run to verify they pass** — `.venv/bin/python -m pytest tests/test_vol_returns.py -v` → 11 passed.

- [ ] **Step 6: Mutation proof** — temporarily replace `_contiguous(prices.index, symbol)` with `pass` in `crypto_returns`; `test_an_interior_gap_is_refused_with_its_date` must FAIL. Restore.

- [ ] **Step 7: Commit** — `git add requirements.txt cryptoval/volatility config/volatility.yml tests/test_vol_returns.py && git commit -m "feat: the volatility chain's returns, windows and the arch dependency"`

---

### Task 2: ARCH-LM, GARCH(1,1) and EGARCH(1,1,1)

**Files:**
- Create: `cryptoval/volatility/garch.py`
- Test: `tests/test_vol_garch.py`

**Interfaces:**
- Consumes: `Unusable`.
- Produces: `arch_lm(returns: pd.Series, lags: int = 7) -> dict` keys `statistic, p_value, n`; `fit_garch(returns) -> dict` keys `omega, alpha, beta, nu, persistence, half_life_days, n` (floats) plus `std_resid`, `cond_vol` (pd.Series on `returns.index`); `fit_egarch(returns) -> dict` keys `gamma, gamma_stderr, p_value, n`.

- [ ] **Step 1: Write the failing tests** — `tests/test_vol_garch.py`:

```python
import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.volatility import garch


def _simulate(n=3000, omega=0.1, alpha=0.1, beta=0.85, leverage=0.0, seed=0):
    """GJR-style: a negative shock adds `leverage` to the variance response."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n)
    out = np.empty(n)
    variance = omega / (1.0 - alpha - beta - leverage / 2.0)
    for t in range(n):
        out[t] = np.sqrt(variance) * shocks[t]
        variance = omega + (alpha + leverage * (out[t] < 0)) * out[t] ** 2 + beta * variance
    return pd.Series(out, index=pd.date_range("2015-01-01", periods=n, freq="D"))


def _iid(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n), index=pd.date_range("2015-01-01", periods=n))


def test_arch_lm_finds_clustering_in_a_garch_process():
    assert garch.arch_lm(_simulate())["p_value"] < 1e-6


def test_arch_lm_finds_none_in_iid_noise():
    assert garch.arch_lm(_iid())["p_value"] > 0.05


def test_garch_recovers_its_parameters():
    fit = garch.fit_garch(_simulate())
    assert fit["alpha"] == pytest.approx(0.1, abs=0.04)
    assert fit["beta"] == pytest.approx(0.85, abs=0.06)
    assert fit["half_life_days"] == pytest.approx(np.log(0.5) / np.log(fit["persistence"]))
    assert fit["std_resid"].index.equals(_simulate().index)


def test_an_integrated_garch_has_no_half_life(monkeypatch):
    class _Res:
        convergence_flag = 0
        nobs = 10
        params = pd.Series({"omega": 0.1, "alpha[1]": 0.2, "beta[1]": 0.8, "nu": 4.0})
        std_resid = conditional_volatility = pd.Series([1.0])

    class _Model:
        def fit(self, disp):
            return _Res()

    monkeypatch.setattr(garch, "arch_model", lambda *a, **k: _Model())
    assert np.isnan(garch.fit_garch(_simulate(100))["half_life_days"])


def test_egarch_recovers_the_leverage_sign():
    fit = garch.fit_egarch(_simulate(leverage=0.15, alpha=0.02))
    assert fit["gamma"] < 0.0
    assert fit["p_value"] < 1e-3


def test_egarch_finds_no_asymmetry_in_a_symmetric_process():
    assert garch.fit_egarch(_simulate())["p_value"] > 0.05


def test_a_flat_series_is_refused_with_its_variance():
    flat = pd.Series(np.zeros(1000), index=pd.date_range("2015-01-01", periods=1000))
    with pytest.raises(Unusable, match="variance 0 is below"):
        garch.fit_garch(flat)


def test_non_convergence_is_refused_with_the_flag(monkeypatch):
    class _Res:
        convergence_flag = 4

        class optimization_result:
            message = "Iteration limit reached"

    class _Model:
        def fit(self, disp):
            return _Res()

    monkeypatch.setattr(garch, "arch_model", lambda *a, **k: _Model())
    with pytest.raises(Unusable, match="flag 4: Iteration limit reached"):
        garch.fit_garch(_simulate(100))
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_vol_garch.py -v` → FAIL, no module `garch`.

- [ ] **Step 3: Implement** `cryptoval/volatility/garch.py`:

```python
"""Volatility clustering and asymmetry (spec 11.4): ARCH-LM, GARCH(1,1), EGARCH(1,1,1).

Student-t innovations throughout: crypto returns are fat-tailed, and a normal likelihood
would be misspecified on every asset. ARCH-LM and EGARCH's gamma are HYPOTHESES (in the
Holm family); the GARCH parameters are estimates.

EGARCH's gamma multiplies the signed shock in log variance: gamma < 0 is the equity
leverage effect (falls raise volatility more than rises), gamma > 0 the inverse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

from cryptoval.econometrics.base import Unusable

ARCH_LM_LAGS = 7

# A series flatter than this has no volatility to model and the likelihood is singular.
MIN_VARIANCE = 1e-8


def _check(returns: pd.Series, name: str) -> None:
    values = returns.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise Unusable(f"{name}: {int((~np.isfinite(values)).sum())} non-finite returns")
    variance = float(np.var(values))
    if variance < MIN_VARIANCE:
        raise Unusable(
            f"{name}: return variance {variance:.3g} is below {MIN_VARIANCE:.0e}; "
            f"there is no volatility to model"
        )


def arch_lm(returns: pd.Series, lags: int = ARCH_LM_LAGS) -> dict[str, float]:
    """Engle (1982). H0: no ARCH effects, i.e. volatility does not cluster."""
    _check(returns, "arch_lm")
    values = returns.to_numpy(dtype="float64")
    statistic, p_value, _, _ = het_arch(
        values - values.mean(), nlags=lags, result_object=False
    )
    return {"statistic": float(statistic), "p_value": float(p_value), "n": float(len(values))}


def _fit(returns: pd.Series, name: str, **spec: Any) -> Any:
    _check(returns, name)
    result = arch_model(returns, mean="Constant", dist="t", **spec).fit(disp="off")
    if result.convergence_flag != 0:
        raise Unusable(
            f"{name}: optimizer did not converge (flag {result.convergence_flag}: "
            f"{result.optimization_result.message})"
        )
    return result


def fit_garch(returns: pd.Series) -> dict[str, Any]:
    result = _fit(returns, "garch", vol="GARCH", p=1, q=1)
    params = result.params
    persistence = float(params["alpha[1]"] + params["beta[1]"])
    # A shock's effect on variance decays by `persistence` a day; at or above 1 the
    # process is integrated and never halves, so a number there would be read as a speed.
    half_life = (
        float(np.log(0.5) / np.log(persistence)) if 0.0 < persistence < 1.0 else float("nan")
    )
    return {
        "omega": float(params["omega"]),
        "alpha": float(params["alpha[1]"]),
        "beta": float(params["beta[1]"]),
        "nu": float(params["nu"]),
        "persistence": persistence,
        "half_life_days": half_life,
        "n": float(result.nobs),
        "std_resid": result.std_resid,
        "cond_vol": result.conditional_volatility,
    }


def fit_egarch(returns: pd.Series) -> dict[str, float]:
    """H0: gamma = 0, a symmetric volatility response. Robust standard errors (arch's default)."""
    result = _fit(returns, "egarch", vol="EGARCH", p=1, o=1, q=1)
    return {
        "gamma": float(result.params["gamma[1]"]),
        "gamma_stderr": float(result.std_err["gamma[1]"]),
        "p_value": float(result.pvalues["gamma[1]"]),
        "n": float(result.nobs),
    }
```

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/python -m pytest tests/test_vol_garch.py -v` → 8 passed. If `test_egarch_finds_no_asymmetry_in_a_symmetric_process` fails at its seed, that is a size draw, not a bug: change `_simulate`'s seed for that test only and record the p-value seen in the commit message.

- [ ] **Step 5: Mutation proof** — replace `values - values.mean()` with `np.abs(values)` in `arch_lm`; `test_arch_lm_finds_none_in_iid_noise` must FAIL. Restore.

- [ ] **Step 6: Commit** — `git commit -m "feat: ARCH-LM, GARCH(1,1)-t and EGARCH(1,1,1)-t, refusing a flat series and a failed optimizer"`

---

### Task 3: The §6.3 seller-exhaustion test

**Files:**
- Create: `cryptoval/volatility/exhaustion.py`
- Test: `tests/test_vol_exhaustion.py`

**Interfaces:**
- Consumes: `fit_garch(...)["cond_vol"]` (pd.Series on the returns' index).
- Produces: `exhaustion_test(returns: pd.Series, cond_vol: pd.Series, block_days: int = 30) -> dict` keys `p_value, continuation_high_vol, continuation_low_vol, block_pairs`; `MIN_BLOCK_PAIRS = 20`.

- [ ] **Step 1: Write the failing tests** — `tests/test_vol_exhaustion.py`:

```python
import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.volatility.exhaustion import exhaustion_test


def _series(blocks=200, block_days=30, low_vol_continues=True, seed=0):
    """Low-volatility blocks are followed by a block of the same sign (when
    low_vol_continues); high-volatility blocks by a coin flip."""
    rng = np.random.default_rng(seed)
    high = rng.random(blocks) < 0.5
    signs = np.empty(blocks)
    signs[0] = 1.0
    for k in range(1, blocks):
        keep = low_vol_continues and not high[k - 1]
        signs[k] = signs[k - 1] if keep else rng.choice([-1.0, 1.0])
    days = blocks * block_days
    returns = np.repeat(signs * 0.5, block_days) + rng.normal(0.0, 0.1, days)
    vol = np.repeat(np.where(high, 5.0, 1.0), block_days)
    index = pd.date_range("2015-01-01", periods=days, freq="D")
    return pd.Series(returns, index=index), pd.Series(vol, index=index)


def test_it_finds_continuation_that_depends_on_volatility():
    result = exhaustion_test(*_series())
    assert result["p_value"] < 1e-6
    assert result["continuation_low_vol"] == pytest.approx(1.0)
    assert result["continuation_high_vol"] == pytest.approx(0.5, abs=0.15)
    assert result["block_pairs"] == 199


def test_it_finds_nothing_when_continuation_is_a_coin_flip():
    assert exhaustion_test(*_series(low_vol_continues=False))["p_value"] > 0.05


def test_too_few_blocks_are_refused_with_the_count():
    returns, vol = _series(blocks=15)
    with pytest.raises(Unusable, match="14 block pairs, 20 needed"):
        exhaustion_test(returns, vol)


def test_misaligned_volatility_is_refused():
    returns, vol = _series()
    with pytest.raises(Unusable, match="not on the returns' dates"):
        exhaustion_test(returns, vol.iloc[1:])
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_vol_exhaustion.py -v` → FAIL, no module.

- [ ] **Step 3: Implement** `cryptoval/volatility/exhaustion.py`:

```python
"""The dissertation's 6.3 claim, as a hypothesis (design 5).

"Elevated volatility bespeaks an uncertain and unpredictable market in which the
direction of price resists forecasting; subdued volatility bespeaks a steadier market
whose price may be expected to continue upon its established course."

NON-OVERLAPPING blocks: overlapping windows share days, so their outcomes are
mechanically correlated and a count test over them is anti-conservative. Each pair of
consecutive blocks is one observation: did block k+1's summed return keep block k's
sign, classified by the GARCH conditional volatility on block k's last day, split at the
median. Fisher's exact test, two-sided; the two continuation rates give the direction.

The conditional volatility uses full-sample GARCH parameters (design 2), and the "profit
metrics" half of 6.3 is not tested: those are UTXO series deferred with the BigQuery work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from cryptoval.econometrics.base import Unusable

MIN_BLOCK_PAIRS = 20


def exhaustion_test(
    returns: pd.Series, cond_vol: pd.Series, block_days: int = 30
) -> dict[str, float]:
    if not cond_vol.index.equals(returns.index):
        raise Unusable("exhaustion: the conditional volatility is not on the returns' dates")
    blocks = len(returns) // block_days
    if blocks - 1 < MIN_BLOCK_PAIRS:
        raise Unusable(
            f"exhaustion: {max(blocks - 1, 0)} block pairs, {MIN_BLOCK_PAIRS} needed "
            f"({len(returns)} returns in {block_days}-day blocks)"
        )
    used = blocks * block_days
    sums = returns.to_numpy(dtype="float64")[:used].reshape(blocks, block_days).sum(axis=1)
    vol = cond_vol.to_numpy(dtype="float64")[:used].reshape(blocks, block_days)[:, -1]

    signs = np.sign(sums)
    continued = signs[1:] == signs[:-1]
    high = vol[:-1] > np.median(vol[:-1])
    table = [
        [int((high & continued).sum()), int((high & ~continued).sum())],
        [int((~high & continued).sum()), int((~high & ~continued).sum())],
    ]
    _, p_value = fisher_exact(table)
    return {
        "p_value": float(p_value),
        "continuation_high_vol": float(continued[high].mean()) if high.any() else float("nan"),
        "continuation_low_vol": float(continued[~high].mean()) if (~high).any() else float("nan"),
        "block_pairs": float(blocks - 1),
    }
```

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/python -m pytest tests/test_vol_exhaustion.py -v` → 4 passed.

- [ ] **Step 5: Mutation proof** — change `[:, -1]` to `[:, 0]`... is not a guaranteed failure in this synthetic (vol is constant within a block). Use instead: replace `high = vol[:-1] > ...` with `high = vol[1:] > np.median(vol[1:])` (classifying by the *outcome* block); `test_it_finds_continuation_that_depends_on_volatility` must FAIL. Restore.

- [ ] **Step 6: Commit** — `git commit -m "feat: the dissertation's seller-exhaustion claim as a block-level Fisher test"`

---

### Task 4: Markov-switching regimes

**Files:**
- Create: `cryptoval/volatility/regimes.py`
- Test: `tests/test_vol_regimes.py`

**Interfaces:**
- Produces: `fit_regimes(returns: pd.Series, search_reps: int = 20, seed: int = 0) -> dict` with float keys `calm_mean, calm_variance, calm_p_stay, calm_duration_days, turbulent_mean, turbulent_variance, turbulent_p_stay, turbulent_duration_days, turbulent_share, n` and `turbulent_probability` (pd.Series on `returns.index`); `REGIME_SCALARS: tuple[str, ...]` (those float keys minus `n`).

- [ ] **Step 1: Write the failing tests** — `tests/test_vol_regimes.py`:

```python
import numpy as np
import pandas as pd
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.volatility.regimes import fit_regimes


def _switching(n=3000, p_stay=0.98, variances=(1.0, 9.0), seed=0):
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < p_stay else 1 - state[t - 1]
    values = rng.standard_normal(n) * np.sqrt(np.asarray(variances)[state])
    index = pd.date_range("2015-01-01", periods=n, freq="D")
    return pd.Series(values, index=index), state


def test_it_recovers_the_variances_in_order_and_the_persistence():
    returns, state = _switching()
    fit = fit_regimes(returns)
    assert fit["calm_variance"] == pytest.approx(1.0, rel=0.2)
    assert fit["turbulent_variance"] == pytest.approx(9.0, rel=0.2)
    assert fit["calm_duration_days"] > 20
    agreement = ((fit["turbulent_probability"] > 0.5).to_numpy() == (state == 1)).mean()
    assert agreement > 0.9


def test_labels_follow_variance_not_the_regime_index():
    returns, _ = _switching(variances=(9.0, 1.0))
    fit = fit_regimes(returns)
    assert fit["calm_variance"] < fit["turbulent_variance"]


def test_it_is_deterministic():
    returns, _ = _switching(n=1500)
    assert fit_regimes(returns)["turbulent_share"] == fit_regimes(returns)["turbulent_share"]


def test_a_short_series_is_refused():
    with pytest.raises(Unusable, match="regimes: 50 returns"):
        fit_regimes(pd.Series(np.ones(50), index=pd.date_range("2015-01-01", periods=50)))
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_vol_regimes.py -v` → FAIL, no module.

- [ ] **Step 3: Implement** `cryptoval/volatility/regimes.py`:

```python
"""Two-regime Markov-switching mean and variance (spec 11.4, design 6).

Regimes are labelled by VARIANCE: statsmodels' regime index is arbitrary and changes
between fits, so 'regime 1' means nothing on its own. 'calm' is the lower-variance one.

ESTIMATES ONLY. The likelihood-ratio test of two regimes against one has a non-standard
distribution (a nuisance parameter is unidentified under the null; Hansen 1992), so a
chi-squared p-value from it would be wrong, and none enters the Holm family.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from cryptoval.econometrics.base import Unusable

MIN_REGIME_ROWS = 365

REGIME_SCALARS = (
    "calm_mean",
    "calm_variance",
    "calm_p_stay",
    "calm_duration_days",
    "turbulent_mean",
    "turbulent_variance",
    "turbulent_p_stay",
    "turbulent_duration_days",
    "turbulent_share",
)


def fit_regimes(returns: pd.Series, search_reps: int = 20, seed: int = 0) -> dict[str, Any]:
    if len(returns) < MIN_REGIME_ROWS:
        raise Unusable(f"regimes: {len(returns)} returns, {MIN_REGIME_ROWS} needed")
    values = returns.to_numpy(dtype="float64")
    # The start-value search draws from numpy's global state; seeding it makes the fit
    # reproducible, which a published table needs.
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = MarkovRegression(
            values, k_regimes=2, trend="c", switching_variance=True
        ).fit(search_reps=search_reps)
    if not result.mle_retvals.get("converged", False):
        raise Unusable(f"regimes: the likelihood did not converge ({result.mle_retvals})")

    params = dict(zip(result.model.param_names, np.asarray(result.params), strict=True))
    variances = [params["sigma2[0]"], params["sigma2[1]"]]
    calm = int(np.argmin(variances))
    turbulent = 1 - calm
    # statsmodels reports p[i->0] only; p[1->1] = 1 - p[1->0].
    stay = [params["p[0->0]"], 1.0 - params["p[1->0]"]]
    durations = np.asarray(result.expected_durations, dtype="float64")
    probability = pd.Series(
        np.asarray(result.smoothed_marginal_probabilities)[:, turbulent], index=returns.index
    )

    out: dict[str, Any] = {"n": float(len(values)), "turbulent_probability": probability}
    for label, i in (("calm", calm), ("turbulent", turbulent)):
        out[f"{label}_mean"] = float(params[f"const[{i}]"])
        out[f"{label}_variance"] = float(variances[i])
        out[f"{label}_p_stay"] = float(stay[i])
        out[f"{label}_duration_days"] = float(durations[i])
    out["turbulent_share"] = float((probability > 0.5).mean())
    return out
```

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/python -m pytest tests/test_vol_regimes.py -v` → 4 passed.

- [ ] **Step 5: Mutation proof** — replace `calm = int(np.argmin(variances))` with `calm = 0`; `test_labels_follow_variance_not_the_regime_index` must FAIL (if statsmodels happens to put the low variance at index 0 for that input, the test is vacuous: swap the simulation's variances until the mutation fails, and keep that input). Restore.

- [ ] **Step 6: Commit** — `git commit -m "feat: two-regime Markov-switching, labelled by variance and seeded"`

---

### Task 5: Bivariate DCC(1,1) and the Engle-Sheppard test

**Files:**
- Create: `cryptoval/volatility/dcc.py`
- Test: `tests/test_vol_dcc.py`

**Interfaces:**
- Produces: `fit_dcc(z: np.ndarray) -> dict` keys `a, b, persistence` (float), `rho` (np.ndarray, len n); `engle_sheppard(z: np.ndarray, lags: int = 5) -> dict` keys `statistic, p_value, n`; `dcc_path(z, a, b) -> np.ndarray`; `MIN_DCC_ROWS = 250`.

- [ ] **Step 1: Write the failing tests** — `tests/test_vol_dcc.py`:

```python
import numpy as np
import pytest

from cryptoval.econometrics.base import Unusable
from cryptoval.volatility.dcc import engle_sheppard, fit_dcc


def _simulate(n, a, b, rho_bar=0.5, seed=0):
    rng = np.random.default_rng(seed)
    qbar = np.array([[1.0, rho_bar], [rho_bar, 1.0]])
    q = qbar.copy()
    z = np.empty((n, 2))
    for t in range(n):
        if t:
            q = (1 - a - b) * qbar + a * np.outer(z[t - 1], z[t - 1]) + b * q
        r = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
        z[t] = np.linalg.cholesky(r) @ rng.standard_normal(2)
    return z


def test_it_recovers_a_known_dcc():
    fit = fit_dcc(_simulate(3000, 0.05, 0.93))
    assert fit["a"] == pytest.approx(0.05, abs=0.03)
    assert fit["b"] == pytest.approx(0.93, abs=0.05)
    assert len(fit["rho"]) == 3000


def test_engle_sheppard_rejects_a_time_varying_correlation():
    assert engle_sheppard(_simulate(3000, 0.05, 0.93))["p_value"] < 1e-6


def test_engle_sheppard_does_not_reject_a_constant_correlation():
    assert engle_sheppard(_simulate(3000, 0.0, 0.0, seed=3))["p_value"] > 0.05


def test_engle_sheppard_size_is_at_most_nominal():
    rejections = sum(
        engle_sheppard(_simulate(1000, 0.0, 0.0, seed=s))["p_value"] < 0.05 for s in range(60)
    )
    assert rejections <= 6  # 5% of 60 is 3; allow sampling noise, not a broken test


def test_a_constant_correlation_fits_a_near_zero():
    assert fit_dcc(_simulate(3000, 0.0, 0.0, seed=3))["a"] < 0.03


def test_a_short_or_non_finite_input_is_refused():
    with pytest.raises(Unusable, match="100 rows, 250 needed"):
        fit_dcc(np.zeros((100, 2)))
    z = _simulate(500, 0.0, 0.0)
    z[10, 0] = np.nan
    with pytest.raises(Unusable, match="1 non-finite"):
        engle_sheppard(z)
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_vol_dcc.py -v` → FAIL, no module.

- [ ] **Step 3: Implement** `cryptoval/volatility/dcc.py`:

```python
"""Bivariate DCC(1,1) with correlation targeting (Engle 2002), and the Engle-Sheppard
(2001) test of constant conditional correlation. Hand-rolled: no free package supplies it.

Input is the two series' GARCH standardized residuals z (n x 2). Stage 2 of Engle's
two-step: Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}, with Qbar the sample
covariance of z, and (a, b) maximise the correlation part of the Gaussian
quasi-likelihood, -1/2 sum(log|R_t| + z_t' R_t^-1 z_t - z_t' z_t).

Measured on simulation (2026-10-18): a = 0.05, b = 0.93 is recovered as 0.060, 0.906 on
3,000 rows; Engle-Sheppard rejects that at p ~ 1e-21 and rejects 1 of 100 constant-
correlation simulations at the 5% level, so it is if anything conservative.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from cryptoval.econometrics.base import Unusable

MIN_DCC_ROWS = 250

# Nelder-Mead from three starts; the likelihood has a flat ridge along a + b near 1.
STARTS = ((0.02, 0.95), (0.05, 0.90), (0.10, 0.80))


def _check(z: np.ndarray, name: str) -> np.ndarray:
    z = np.asarray(z, dtype="float64")
    bad = int((~np.isfinite(z)).sum())
    if bad:
        raise Unusable(f"{name}: {bad} non-finite standardized residual(s)")
    if len(z) < MIN_DCC_ROWS:
        raise Unusable(f"{name}: {len(z)} rows, {MIN_DCC_ROWS} needed")
    return z


def dcc_path(z: np.ndarray, a: float, b: float) -> np.ndarray:
    qbar = np.cov(z, rowvar=False, bias=True)
    q = qbar.copy()
    rho = np.empty(len(z))
    for t in range(len(z)):
        if t:
            q = (1.0 - a - b) * qbar + a * np.outer(z[t - 1], z[t - 1]) + b * q
        rho[t] = q[0, 1] / np.sqrt(q[0, 0] * q[1, 1])
    return rho


def _negative_loglik(theta: np.ndarray, z: np.ndarray) -> float:
    a, b = theta
    if a < 0.0 or b < 0.0 or a + b >= 0.999:
        return 1e10
    rho = dcc_path(z, a, b)
    one = 1.0 - rho**2
    quad = (z[:, 0] ** 2 - 2.0 * rho * z[:, 0] * z[:, 1] + z[:, 1] ** 2) / one
    return float(0.5 * np.sum(np.log(one) + quad - (z[:, 0] ** 2 + z[:, 1] ** 2)))


def fit_dcc(z: np.ndarray) -> dict:
    z = _check(z, "dcc")
    best = None
    for start in STARTS:
        result = minimize(
            _negative_loglik,
            np.asarray(start),
            args=(z,),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
        )
        if best is None or result.fun < best.fun:
            best = result
    if not best.success:
        raise Unusable(f"dcc: the optimizer did not converge ({best.message})")
    a, b = (float(v) for v in best.x)
    return {"a": a, "b": b, "persistence": a + b, "rho": dcc_path(z, a, b)}


def engle_sheppard(z: np.ndarray, lags: int = 5) -> dict[str, float]:
    """H0: constant conditional correlation. Whiten z by the sample correlation; under H0
    the cross-product u1*u2 is unpredictable, so regressing it on a constant and its own
    lags gives coefficients of zero. Statistic d'X'Xd / s^2 ~ chi2(lags + 1)."""
    z = _check(z, "engle_sheppard")
    values, vectors = np.linalg.eigh(np.corrcoef(z, rowvar=False))
    u = z @ (vectors @ np.diag(values**-0.5) @ vectors.T)
    y = u[:, 0] * u[:, 1]
    target = y[lags:]
    design = np.column_stack(
        [np.ones(len(target))] + [y[lags - k : -k] for k in range(1, lags + 1)]
    )
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    s2 = float(resid @ resid / len(target))
    statistic = float(coef @ design.T @ design @ coef / s2)
    return {
        "statistic": statistic,
        "p_value": float(chi2.sf(statistic, lags + 1)),
        "n": float(len(target)),
    }
```

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/python -m pytest tests/test_vol_dcc.py -v` → 6 passed (the recovery and size tests take ~20 s).

- [ ] **Step 5: Mutation proof** — in `dcc_path` replace `a * np.outer(z[t - 1], z[t - 1])` with `a * np.outer(z[t], z[t])` (a same-day leak); `test_it_recovers_a_known_dcc` must FAIL. Restore.

- [ ] **Step 6: Commit** — `git commit -m "feat: hand-rolled bivariate DCC(1,1) and the Engle-Sheppard constant-correlation test"`

---

### Task 6: The runner, one Holm family, refusals as rows

**Files:**
- Create: `cryptoval/volatility/runner.py`
- Test: `tests/test_vol_runner.py`

**Interfaces:**
- Consumes: everything above; `correct_family_wise(scores, *, p_metric, columns, count_keys)`; `stable_hash`.
- Produces: `TEST_COLUMNS`; `run_volatility(crypto: DataFrame, traditional: DataFrame, config: VolConfig, stablecoins: frozenset[str] = frozenset(), panel_vintage_ts=None) -> DataFrame`. `crypto` and `traditional` both have columns `symbol, metric_date, price_usd`.

- [ ] **Step 1: Write the failing tests** — `tests/test_vol_runner.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.volatility import runner
from cryptoval.volatility.runner import TEST_COLUMNS, run_volatility
from cryptoval.volatility.settings import VolConfig

CONFIG = VolConfig(
    windows={
        "early": (dt.date(2019, 6, 1), dt.date(2019, 12, 31)),
        "peak": (dt.date(2020, 3, 1), dt.date(2020, 4, 30)),
        "before": (dt.date(2010, 1, 1), dt.date(2010, 12, 31)),
    },
    peak_window="peak",
    crypto_anchors=("BTC",),
    traditional_anchors=("BTC",),
    markov_search_reps=5,
)


def _garch_prices(n, seed):
    rng = np.random.default_rng(seed)
    out, variance = np.empty(n), 1.0
    for t in range(n):
        out[t] = np.sqrt(variance) * rng.standard_normal()
        variance = 0.1 + 0.1 * out[t] ** 2 + 0.85 * variance
    return 100.0 * np.exp(np.cumsum(out) / 100.0)


def _crypto(symbols=("BTC", "ETH", "USDC"), days=1200, start="2019-01-01"):
    frames = []
    for i, symbol in enumerate(symbols):
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "metric_date": pd.date_range(start, periods=days, freq="D"),
                    "price_usd": _garch_prices(days, seed=i),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _traditional():
    days = pd.bdate_range("2018-01-01", "2023-06-30")
    return pd.DataFrame(
        {"symbol": "AAPL", "metric_date": days, "price_usd": _garch_prices(len(days), seed=9)}
    )


@pytest.fixture(scope="module")
def table():
    return run_volatility(_crypto(), _traditional(), CONFIG, frozenset({"USDC"}))


def test_columns_and_the_hypothesis_count(table):
    assert list(table.columns) == TEST_COLUMNS
    count = table[table["metric"] == "hypotheses_tested"]["value"].item()
    assert count == ((table["metric"] == "p_value") & table["value"].notna()).sum()
    # BTC and ETH: arch_lm + egarch + exhaustion each; BTC|ETH and BTC|AAPL: engle-sheppard.
    assert count == 8
    assert (table["metric"] == "p_value_holm").sum() == 8


def test_a_stablecoin_is_refused_by_name(table):
    usdc = table[table["symbol"] == "USDC"]
    assert usdc["metric"].tolist() == ["refused"]
    assert "stablecoin" in usdc["reason"].item()


def test_both_pairs_are_fitted_with_windows_and_a_peak_rank(table):
    for pair in ("BTC|ETH", "BTC|AAPL"):
        rows = table[(table["symbol"] == pair) & (table["model_id"] == "dcc")]
        assert {"a", "b", "p_value"} <= set(rows["metric"])
        peak = rows[rows["metric"] == "peak_percentile"]["value"].item()
        assert 0.0 <= peak <= 1.0
        early = rows[(rows["metric"] == "pearson_correlation") & (rows["window_name"] == "early")]
        assert early["value"].notna().item()


def test_a_window_outside_the_history_is_a_nan_row_with_its_count(table):
    rows = table[(table["window_name"] == "before") & (table["symbol"] == "BTC")]
    assert rows["value"].isna().all()
    assert rows["reason"].str.contains("0 rows in window before").all()


def test_markov_rows_carry_each_window(table):
    rows = table[(table["model_id"] == "markov") & (table["symbol"] == "BTC")]
    assert set(rows["window_name"].dropna()) == {"early", "peak", "before"}


def test_a_garch_that_fails_to_converge_is_a_refusal_and_the_run_continues(monkeypatch):
    from cryptoval.econometrics.base import Unusable

    real = runner.fit_garch

    def flaky(returns):
        if len(returns) == 1199 and abs(returns.iloc[0] - _first_eth_return()) < 1e-9:
            raise Unusable("garch: optimizer did not converge (flag 4: test)")
        return real(returns)

    monkeypatch.setattr(runner, "fit_garch", flaky)
    out = run_volatility(_crypto(("BTC", "ETH")), _traditional(), CONFIG)
    eth = out[(out["symbol"] == "ETH")]
    assert eth[eth["model_id"] == "garch"]["metric"].tolist() == ["refused"]
    assert "needs the garch fit" in eth[eth["model_id"] == "exhaustion"]["reason"].item()
    assert (out[(out["symbol"] == "BTC") & (out["model_id"] == "garch")]["metric"] == "alpha").any()


def _first_eth_return():
    crypto = _crypto(("BTC", "ETH"))
    prices = crypto[crypto["symbol"] == "ETH"]["price_usd"].to_numpy()
    return 100.0 * np.log(prices[1] / prices[0])


def test_a_missing_anchor_refuses_its_pairs_by_name():
    out = run_volatility(_crypto(("ETH",)), _traditional(), CONFIG)
    refused = out[(out["model_id"] == "dcc") & (out["metric"] == "refused")]
    assert set(refused["symbol"]) == {"BTC|ETH", "BTC|AAPL"}
    assert refused["reason"].str.contains("BTC is not in the panel").all()
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_vol_runner.py -v` → FAIL, no module `runner`.

- [ ] **Step 3: Implement** `cryptoval/volatility/runner.py`:

```python
"""One long table for the volatility chain, at grain (model_id, symbol, metric, window).

The same contract as fct_econometric_test: a refusal is a ROW carrying the measurement,
not a silently absent asset. ONE Holm family over every row whose metric is 'p_value':
ARCH-LM, EGARCH gamma, the 6.3 exhaustion test, Engle-Sheppard. Everything else is an
estimate or a description (design 11).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from cryptoval.econometrics.base import Unusable
from cryptoval.hashing import stable_hash
from cryptoval.scoring.multiplicity import correct_family_wise
from cryptoval.volatility.dcc import engle_sheppard, fit_dcc
from cryptoval.volatility.exhaustion import exhaustion_test
from cryptoval.volatility.garch import arch_lm, fit_egarch, fit_garch
from cryptoval.volatility.regimes import REGIME_SCALARS, fit_regimes
from cryptoval.volatility.returns import crypto_returns, pair_returns, price_series
from cryptoval.volatility.settings import VolConfig

TEST_COLUMNS = [
    "model_id",
    "symbol",
    "metric",
    "value",
    "n_rows",
    "window_name",
    "reason",
    "params_hash",
    "panel_vintage_ts",
]

COUNT_KEYS = {"model_id": "*", "symbol": "*"}

# Below this a window's mean is a handful of days, and the reason names the count.
MIN_WINDOW_ROWS = 20

# What a singular or degenerate fit raises from numpy/scipy/arch; each becomes a refusal.
FIT_ERRORS = (Unusable, np.linalg.LinAlgError, ValueError)

GARCH_KEYS = ("omega", "alpha", "beta", "nu", "persistence", "half_life_days")
DCC_KEYS = ("a", "b", "persistence")


def _row(**values: Any) -> dict[str, Any]:
    row = dict.fromkeys(TEST_COLUMNS)
    row.update(values)
    return row


def _in_window(index: pd.DatetimeIndex, window: tuple) -> np.ndarray:
    start, end = window
    return np.asarray((index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end)))


def run_volatility(
    crypto: pd.DataFrame,
    traditional: pd.DataFrame,
    config: VolConfig,
    stablecoins: frozenset[str] = frozenset(),
    panel_vintage_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    common = {
        "params_hash": stable_hash(dataclasses.asdict(config)),
        "panel_vintage_ts": panel_vintage_ts,
    }
    rows: list[dict[str, Any]] = []

    def refuse(model_id: str, symbol: str, reason: Any, n: float | None = None) -> None:
        rows.append(
            _row(model_id=model_id, symbol=symbol, metric="refused", reason=str(reason),
                 n_rows=n, **common)
        )

    def emit(model_id: str, symbol: str, values: dict, keys: tuple, n: float) -> None:
        for key in keys:
            value = float(values[key])
            reason = None
            if key == "half_life_days" and np.isnan(value):
                reason = f"persistence {values['persistence']:.4f} >= 1: no half life"
            rows.append(
                _row(model_id=model_id, symbol=symbol, metric=key, value=value, n_rows=n,
                     reason=reason, **common)
            )

    def per_window(model_id, symbol, metric, index, compute) -> None:
        for name, window in sorted(config.windows.items()):
            mask = _in_window(index, window)
            k = int(mask.sum())
            if k < MIN_WINDOW_ROWS:
                rows.append(
                    _row(model_id=model_id, symbol=symbol, metric=metric, window_name=name,
                         n_rows=float(k), reason=f"{k} rows in window {name}, "
                         f"{MIN_WINDOW_ROWS} needed", **common)
                )
                continue
            rows.append(
                _row(model_id=model_id, symbol=symbol, metric=metric, window_name=name,
                     value=float(compute(mask)), n_rows=float(k), **common)
            )

    prices: dict[str, pd.Series] = {}
    usable: list[str] = []
    for symbol in sorted(str(s) for s in crypto["symbol"].unique()):
        if symbol in stablecoins:
            refuse("returns", symbol, f"{symbol} is a stablecoin: a pegged price's variance "
                   f"is not a market quantity")
            continue
        try:
            prices[symbol] = price_series(crypto, symbol)
            returns = crypto_returns(prices[symbol], symbol)
        except Unusable as exc:
            refuse("returns", symbol, exc)
            continue
        usable.append(symbol)
        n = float(len(returns))

        try:
            emit("arch_lm", symbol, arch_lm(returns, config.arch_lm_lags),
                 ("statistic", "p_value"), n)
        except FIT_ERRORS as exc:
            refuse("arch_lm", symbol, exc, n)

        garch = None
        try:
            garch = fit_garch(returns)
            emit("garch", symbol, garch, GARCH_KEYS, n)
        except FIT_ERRORS as exc:
            refuse("garch", symbol, exc, n)

        try:
            emit("egarch", symbol, fit_egarch(returns), ("gamma", "gamma_stderr", "p_value"), n)
        except FIT_ERRORS as exc:
            refuse("egarch", symbol, exc, n)

        if garch is None:
            refuse("exhaustion", symbol, "exhaustion: needs the garch fit, which was refused", n)
        else:
            try:
                result = exhaustion_test(returns, garch["cond_vol"], config.exhaustion_block_days)
                emit("exhaustion", symbol, result, ("p_value", "continuation_high_vol",
                     "continuation_low_vol", "block_pairs"), n)
            except FIT_ERRORS as exc:
                refuse("exhaustion", symbol, exc, n)

        try:
            regimes = fit_regimes(returns, search_reps=config.markov_search_reps)
        except FIT_ERRORS as exc:
            refuse("markov", symbol, exc, n)
        else:
            emit("markov", symbol, regimes, REGIME_SCALARS, n)
            probability = regimes["turbulent_probability"]
            per_window("markov", symbol, "turbulent_probability", probability.index,
                       lambda mask, p=probability: p[mask].mean())

    pairs: list[tuple[str, str, bool]] = []
    for anchor in config.crypto_anchors:
        pairs += [(anchor, other, False) for other in usable if other != anchor]
    for anchor in config.traditional_anchors:
        pairs += [(anchor, s, True) for s in sorted(str(s) for s in traditional["symbol"].unique())]

    for left, right, trading in pairs:
        label = f"{left}|{right}"
        try:
            left_prices = prices[left] if left in prices else price_series(crypto, left)
            right_prices = (
                price_series(traditional, right) if trading else
                prices[right] if right in prices else price_series(crypto, right)
            )
            pair = pair_returns(left_prices, right_prices, label, trading_calendar=trading)
            z = np.column_stack([
                fit_garch(pair["a"])["std_resid"].to_numpy(),
                fit_garch(pair["b"])["std_resid"].to_numpy(),
            ])
            dcc = fit_dcc(z)
            es = engle_sheppard(z, config.engle_sheppard_lags)
        except FIT_ERRORS as exc:
            refuse("dcc", label, exc)
            continue

        n = float(len(pair))
        rho = pd.Series(dcc["rho"], index=pair.index)
        emit("dcc", label, dcc, DCC_KEYS, n)
        emit("dcc", label, {"correlation_mean": rho.mean(), "correlation_min": rho.min(),
             "correlation_max": rho.max()}, ("correlation_mean", "correlation_min",
             "correlation_max"), n)
        emit("dcc", label, {"engle_sheppard_statistic": es["statistic"],
             "p_value": es["p_value"]}, ("engle_sheppard_statistic", "p_value"), n)
        per_window("dcc", label, "dcc_correlation", pair.index, lambda m, r=rho: r[m].mean())
        per_window("dcc", label, "pearson_correlation", pair.index,
                   lambda m, p=pair: p["a"][m].corr(p["b"][m]))

        # Descriptive, not a test: rolling means overlap, so no rank has a p-value.
        peak = _in_window(pair.index, config.windows[config.peak_window])
        k = int(peak.sum())
        if k < MIN_WINDOW_ROWS:
            rows.append(_row(model_id="dcc", symbol=label, metric="peak_percentile",
                             window_name=config.peak_window, n_rows=float(k),
                             reason=f"{k} rows in window {config.peak_window}, "
                             f"{MIN_WINDOW_ROWS} needed", **common))
        else:
            rolling = rho.rolling(k).mean().dropna()
            rank = float((rolling < rho[peak].mean()).mean())
            rows.append(_row(model_id="dcc", symbol=label, metric="peak_percentile",
                             window_name=config.peak_window, value=rank,
                             n_rows=float(len(rolling)), **common))

    return correct_family_wise(
        pd.DataFrame(rows, columns=TEST_COLUMNS),
        p_metric="p_value",
        columns=TEST_COLUMNS,
        count_keys={**COUNT_KEYS, **common},
    )
```

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/python -m pytest tests/test_vol_runner.py -v` → 7 passed. Format with `.venv/bin/ruff format cryptoval/volatility` and lint with `.venv/bin/ruff check cryptoval tests`.

- [ ] **Step 5: Mutation proof** — change `p_metric="p_value"` to `p_metric="statistic"`; `test_columns_and_the_hypothesis_count` must FAIL. Restore.

- [ ] **Step 6: Commit** — `git commit -m "feat: the volatility runner — one Holm family, refusals as rows, windows as a column"`

---

### Task 7: CLI, mart, Makefile

**Files:**
- Modify: `cryptoval/cli.py` (imports; `_read_vol_inputs`, `run_vol`; `vol` subcommand), `Makefile` (`vol` target, `.PHONY`, `all`), `dbt/models/marts/_marts.yml`
- Create: `dbt/models/marts/fct_volatility_test.sql`
- Test: `tests/test_marts_contract.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `run_volatility`, `TEST_COLUMNS`, `load_vol_config`.
- Produces: `cli.run_vol(settings, assets) -> dict[str, dict]` (per `model_id`: `rows, refusals, symbols, significant_after_holm`); `cli._read_vol_inputs(settings) -> tuple[DataFrame, DataFrame]`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_marts_contract.py` (and add `from cryptoval.volatility.runner import TEST_COLUMNS as VOL_COLUMNS` to its imports):

```python
def test_the_volatility_mart_selects_exactly_test_columns():
    assert _selected_columns("dbt/models/marts/fct_volatility_test.sql") == VOL_COLUMNS
```

Append to `tests/test_cli.py`:

```python
def test_vol_writes_the_documented_columns_to_gold(tmp_path, monkeypatch):
    from cryptoval.volatility.runner import TEST_COLUMNS as VOL_COLUMNS

    rng = np.random.default_rng(7)
    days = pd.date_range("2019-01-01", periods=1000, freq="D")
    crypto = pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": days,
            "price_usd": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, len(days)))),
            "panel_vintage_ts": pd.Timestamp("2026-10-18 08:30", tz="UTC"),
        }
    )
    trading = pd.bdate_range("2019-01-01", "2021-12-31")
    traditional = pd.DataFrame(
        {
            "symbol": "AAPL",
            "metric_date": trading,
            "price_usd": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, len(trading)))),
        }
    )
    settings = Settings(data_root=str(tmp_path))
    monkeypatch.setattr(cli, "_read_vol_inputs", lambda _settings: (crypto, traditional))
    report = cli.run_vol(settings, [])

    written = pd.read_parquet(
        pathlib.Path(settings.resolver.table("gold", "volatility_test")) / "part-0.parquet"
    )
    assert list(written.columns) == VOL_COLUMNS
    assert (written["metric"] == "hypotheses_tested").any()
    assert report["garch"]["rows"] > 0
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_marts_contract.py tests/test_cli.py -k "volatility or vol_" -v` → FAIL (no SQL file; no `run_vol`).

- [ ] **Step 3: Implement.** `dbt/models/marts/fct_volatility_test.sql`:

```sql
-- The volatility chain at grain (model_id, symbol, metric, window), spec 11.4.
--
-- FULL SAMPLE, by design: parameters are estimated over the whole history, so every
-- conditional volatility and correlation uses future data. No forecasting claim; not
-- comparable to fct_model_score.
--
-- symbol is an asset ('BTC') or a pair ('BTC|ETH'). window_name is NULL except on per-window
-- rows (the dissertation's correlation windows, config/volatility.yml).
-- model_id = '*' holds the hypothesis count: the family is every row whose metric is
-- 'p_value' (ARCH-LM, EGARCH gamma, the 6.3 exhaustion test, Engle-Sheppard).
-- A 'refused' row carries a NULL value and the measurement in reason.
select
    model_id,
    symbol,
    metric,
    value,
    n_rows,
    window_name,
    reason,
    params_hash,
    panel_vintage_ts
from read_parquet('{{ var("data_root") }}/gold/volatility_test/*.parquet')
```

Append to `dbt/models/marts/_marts.yml` after `fct_econometric_test`:

```yaml
  - name: fct_volatility_test
    description: >
      The volatility chain (GARCH/EGARCH, Markov-switching, DCC) at grain (model_id,
      symbol, metric, window), spec 11.4. Full sample and therefore descriptive. symbol is
      an asset or an 'A|B' pair; window_name is NULL except on per-window rows. model_id = '*'
      holds the hypothesis count over every 'p_value' row.
    tests:
      - dbt_utils.unique_combination_of_columns:
          arguments:
            combination_of_columns: [model_id, symbol, metric, window_name]
    columns:
      - name: model_id
        tests: [not_null]
      - name: metric
        tests: [not_null]
      - name: params_hash
        tests: [not_null]
```

In `cryptoval/cli.py`, beside the econometrics imports:

```python
from cryptoval.volatility.runner import run_volatility
from cryptoval.volatility.settings import load_vol_config
```

After `run_econ`:

```python
def _read_vol_inputs(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crypto prices and traditional prices, each (symbol, metric_date, price_usd).
    Separate frames: a shared date spine would manufacture weekend nulls (spec 8.1)."""
    import duckdb

    con = duckdb.connect(str(pathlib.Path(settings.data_root) / "cryptoval.duckdb"), read_only=True)
    try:
        crypto = con.execute(
            "select symbol, metric_date, price_usd, panel_vintage_ts from fct_asset_metrics_daily"
        ).fetchdf()
        traditional = con.execute(
            "select symbol, metric_date, price_usd from fct_equity_prices_daily"
        ).fetchdf()
        return crypto, traditional
    finally:
        con.close()


def run_vol(settings: Settings, assets: list[Asset]) -> dict[str, dict]:
    """The volatility chain over every asset and pair, full sample, written to gold."""
    crypto, traditional = _read_vol_inputs(settings)
    stablecoins = frozenset(a.symbol for a in assets if a.is_stablecoin)
    vintage = crypto["panel_vintage_ts"].max() if "panel_vintage_ts" in crypto.columns else None
    table = run_volatility(crypto, traditional, load_vol_config(), stablecoins, vintage)
    write_gold(settings.resolver, "volatility_test", table)

    report: dict[str, dict] = {}
    for model_id, group in table[table["model_id"] != "*"].groupby("model_id"):
        holm = group[(group["metric"] == "p_value_holm") & (group["value"] < 0.05)]
        report[str(model_id)] = {
            "rows": int(len(group)),
            "refusals": int((group["metric"] == "refused").sum()),
            "symbols": int(group["symbol"].nunique()),
            "significant_after_holm": int(len(holm)),
        }
    return report
```

In `main`, after the `econ` subparser: `sub.add_parser("vol")`; and after the `econ` branch:

```python
    if args.command == "vol":
        for model_id, outcome in sorted(run_vol(settings, assets).items()):
            print(model_id, outcome)
        # No clustering or no time-varying correlation is a result, not a failed run.
        return 0
```

`Makefile`: add `vol` to `.PHONY`; add

```make
vol:
	$(PY) -m cryptoval.cli vol
```

and in `all`, after `$(MAKE) econ`, add `$(MAKE) vol`.

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/python -m pytest -q` → all pass (756 + the new tests); `.venv/bin/ruff check cryptoval tests` clean.

- [ ] **Step 5: Commit** — `git commit -m "feat: cryptoval vol writes gold volatility_test; fct_volatility_test mart; make vol in make all"`

---

### Task 8: The live run and the chapter

**Files:**
- Create: `reports/does-volatility-cluster.md`
- Modify: `reports/does-price-anchor-to-fundamentals.md` (its out-of-scope row for GARCH/Markov/DCC points at the new chapter)

- [ ] **Step 1: Run live** — from the worktree, with the main checkout's data: `CRYPTOVAL_DATA_ROOT="/Users/yas/Documents/Crypto/Crypto Valuation/data" .venv/bin/python -m cryptoval.cli vol`, then `cd dbt && ../.venv/bin/dbt build --profiles-dir . --vars '{data_root: "<that path>"}' -s fct_volatility_test`. Record the per-model report and the runtime.

- [ ] **Step 2: Measure** — query `fct_volatility_test` for: hypotheses tested; Holm survivors per model; per-asset GARCH persistence and half life; EGARCH γ signs; exhaustion continuation rates high vs low; Markov turbulent share and the peak window's turbulent probability for BTC/ETH; BTC|X DCC a, b, Engle-Sheppard survivors; per-window DCC vs Pearson for BTC against ETH, DOT, SOL, LINK, ADA beside the dissertation's Tables 3-7 figures (BTC-ETH 0.82, 0.84, 0.93, 0.87, 0.73); peak percentile ranks; BTC|equity and ETH|equity window means beside Table 8 (ETH-META 0.58, ETH-AAPL 0.42, BTC ≤ 0.14).

- [ ] **Step 3: Write the chapter** in the house style of `reports/does-price-anchor-to-fundamentals.md`: the questions (design 1); what was tested and on which universe, with every refusal and its count; results per family with the numbers from Step 2; the dissertation's claims one by one with a verdict (holds / does not hold / not testable here); the multiple-testing section (family size, what is outside it and why); declared gaps (design 12) and the scope table. Every number comes from the Step 2 queries; no number is typed from memory.

- [ ] **Step 4: Commit** — `git commit -m "docs: the volatility chapter, from a live run"`
