# Cryptoasset Valuation — The Modern Layer on Free Data Implementation Plan (Plan 3 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build spec §13's Phase 3a — Family C (the DCF chapter and its counter-model), the remaining
Family B, D and E(a) methods, and the Family F falsified-benchmark shelf — on free data, after
ingesting the three series those methods need and the panel does not yet carry.

**Architecture:** Three stages. **3a-1 (Tasks 1-3)** adds the nine methods that need no data the
panel does not already have, so the registry gains a falsified shelf and its remaining multiples
before any ingestion work. **3a-2 (Tasks 4-7)** ingests what Family C requires: per-protocol fee and
revenue *history* (the existing DefiLlama adapter stores a daily snapshot, so
`fct_protocol_revenue_daily` holds three days where a DCF needs years), supply metadata for FDV, and
protocol TVL; then widens the frame `run_estimate` hands the registry, so a method reaches the new
columns by declaring them in `requires` and the method contract itself does not change. **3a-3
(Tasks 8-12)** adds the data-dependent methods — the remaining multiples, sector-cohorted
cross-sectional z-scores (the one shape the per-symbol run loop cannot express), the five Family C
methods and the supply-effect counter-model, plus the Kampakis log-linear EoE this plan's own
self-review found missing from Plan 2 — then the cards, the estimate run and the report chapter.

**Tech Stack:** Python 3.14.5, pandas 3.0.6, DuckDB 1.5.6, dbt-core 1.12.5 with dbt-duckdb 1.11.0,
pyarrow 26.0.0, httpx 0.28.1, statsmodels 0.15.0, scipy (already present as a statsmodels
dependency at 1.18.1, pinned explicitly in this plan for the LPPLS optimiser), pytest 9.1.1,
ruff 0.17.0.

**Spec:** `docs/superpowers/specs/2026-10-11-cryptoasset-valuation-pipeline-design.md` (revised
2026-10-12). This plan implements §10.3 remainder, §10.4 in full, §10.5 remainder, §10.6(a)
remainder and §10.7, under the Phase 3a row of §13.

**Scope boundary — read this before starting.** Spec §13 splits Phase 3 in two, and so does the
plan sequence: **Family E(b) (Cointime, HODL waves, SLRV/STHPL/RPV) and the BigQuery sandbox
extraction are NOT in this plan.** They are an independent subsystem — a different source, a
different cost-control mechanism (dry-run byte gate plus monthly byte ledger, §6.3), and a timeline
the spec itself measures in months — and §6.3 requires the family to flip from `applicable=False`
to live in one step, which only a completed backfill can do. Nothing in this plan depends on it, and
it depends on nothing here, so it belongs in its own plan (Plan 3b) written against §6.3 alone.
Plan 2's spec-coverage appendix assigned E(b) to "Plan 3"; this is a deliberate narrowing of that,
recorded here so the next planner does not read it as an omission.

## Global Constraints

Every task's requirements implicitly include this section.

- **Free tiers only, no exceptions.** No paid API calls, no API keys. BigQuery stays untouched in
  this plan (it is Plan 3b).
- **Python 3.14.5**, dependencies pinned to exact versions in `requirements.txt`. One addition in
  this plan: `scipy==1.18.1`, which is already installed as a statsmodels dependency — pin it, do
  not upgrade it, and add nothing else.
- **All timestamps UTC.** Every datetime parsed with `utc=True`; naive datetimes are a bug.
  DefiLlama's history endpoints return Unix *seconds*: parse with `pd.to_datetime(..., unit="s",
  utc=True)`.
- **Parquet is the system of record.** DuckDB files are derived and disposable.
- **Bronze is immutable.** Never delete or overwrite a bronze partition; `write_bronze` takes the
  run's `run_id` and appends.
- **No network calls in unit tests.** Unit tests read recorded fixtures. Network tests are marked
  `@pytest.mark.integration` and excluded from the default run.
- **No `train_test_split`, no shuffling** of time-ordered data anywhere in the repository, enforced
  by `tests/test_repo_hygiene.py`'s AST detector. Every fit in this plan uses a trailing window or
  the full prior series; none resamples.
- **Nulls stay null.** A missing vendor value is never coerced to `0.0`. This bites hardest in this
  plan: DefiLlama's history arrays carry a literal `0` for days before a protocol existed and for
  days its adapter was down, and a mean over those zeros is a wrong number that looks right. Task 4
  separates the two cases.
- **No method invents a number.** A method whose inputs are unavailable returns `applicable=False`
  with a reason carrying the measurement. The registry enforces this; methods do not restate it.
- **No multiple is published without naming its denominator** (spec §10.4). Gross fees, net protocol
  revenue, and value reaching holders after dilution are three quantities; every estimate row names
  which one it used in `inputs_used`.
- **Every commit ends with** `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Review Focus

Failure modes the spec implies that no obvious task test would exercise. Each has its test assigned
to the task owning the code.

1. **A DCF whose answer is entirely its terminal value.** A three-stage model with a terminal
   multiple contributes most of its present value in the terminal term; at a low discount rate and a
   high multiple the explicit forecast is decoration. Expected: every DCF row reports
   `terminal_share` in diagnostics, and a grid cell whose terminal share exceeds a declared ceiling
   is still returned but flagged, so the reader sees that the number is a terminal-multiple
   assumption rather than a cash-flow analysis. → Task 10.
2. **A revenue zero that is not a zero.** DefiLlama's `totalDataChart` begins at the dataset's
   start, not the protocol's: Ethereum's series opens with `[1438905600, 0]` in 2015 and aave-v3's
   with `[..., 6]`-scale noise before launch. Leading zeros are pre-existence, interior zeros may be
   a genuine zero-fee day or an adapter outage. Expected: leading zeros are trimmed rather than
   stored, interior zeros are stored as zero, and the trim boundary is recorded, so no DCF averages
   a protocol's pre-history. → Task 4.
3. **An FDV on an asset that has no maximum supply.** CoinGecko returns `max_supply: null` for ETH
   and a real figure for AAVE (16,000,000). A silent fallback to `total_supply` would publish an
   FDV/revenue multiple for ETH that is really a market-cap/revenue multiple. Expected: a missing
   max supply refuses the FDV-based methods for that asset with the reason naming the absent field,
   and never substitutes another supply series. → Task 8.
4. **A cohort z-score over a cohort of one.** `dim_asset.sector` has five single-member sectors
   (`metaverse`, `scaling`, `oracle`, `exchange`, `stablecoin`) and seven assets with a null sector.
   A z-score needs a cross-sectional standard deviation, which is zero or undefined there, and
   `(x - mean) / 0` is `inf` or `NaN`, not a score. Expected: a declared minimum cohort size, and a
   refusal naming the measured cohort size below it. → Task 9.
5. **An LPPLS fit on a window containing no bubble.** The optimiser will converge on *something* for
   any monotone series, and a critical time `tc` fitted a decade away is a fit, not a signal.
   Expected: `tc` bounded to a declared horizon beyond the window, a fit-quality floor, and a
   refusal when the bound binds or the floor fails — the flag reports "no bubble regime detected"
   rather than a date. → Task 3.

---

## File Structure

**Stage 3a-1 — methods on the existing panel**

| File | Responsibility |
|---|---|
| `cryptoval/methods/benchmarks.py` *(create)* | Family F: `StockToFlow`, `PowerLaw`. The shelf is a separate module because its methods are published *as failed*, and a reader of `multiples.py` should not have to know which of its classes are discredited |
| `cryptoval/methods/cost.py` *(modify)* | Family E(a) remainder: `PuellMultiple`, `MayerMultiple`, `MvrvZScore`, `EnergyValue` — all four are cost-side, and `EnergyValue` reuses `hayes_cost_per_coin` directly |
| `cryptoval/methods/network.py` *(modify)* | Family B remainder: `PetersonMetcalfe`, `NvmRatio`, `LpplsBubbleFlag`; `fit_lppls()` as a module function so its optimiser is testable without the registry |
| `config/methods/*.yml` *(create ×9)* | One config per new method, carrying `requires`, density floor, params, citation, and the card prose |

**Stage 3a-2 — the data Family C needs**

| File | Responsibility |
|---|---|
| `cryptoval/sources/defillama.py` *(modify)* | `parse_fee_history()` and `DefiLlamaHistorySource` — one overview call for the name→slug map, then `/summary/fees/<slug>` per configured protocol for the daily series |
| `config/contracts/defillama_history.yml` *(create)* | Bounds for the history series: non-negative gross fees, net revenue unbounded below (Plan 2 Task 4's ruling), a maximum guarding a vendor rescale |
| `cryptoval/sources/prices.py` *(modify)* | `parse_supply_metadata()` — circulating, total and max supply plus FDV from CoinGecko `/coins/markets`, one call for the whole universe |
| `cryptoval/sources/defillama.py` *(modify)* | `parse_tvl_history()` and `DefiLlamaTvlSource` — `/protocol/<slug>`'s `tvl` array, behind a payload-size guard |
| `dbt/models/staging/stg_protocol_fee_history.sql` *(create)* | The history series resolved to `symbol` and rolled up over slug lists |
| `dbt/models/staging/stg_protocol_tvl.sql` *(create)* | TVL resolved to `symbol`, rolled up the same way |
| `dbt/models/marts/fct_protocol_revenue_daily.sql` *(modify)* | Carries the history series, not only the snapshot, with `series_source` naming which produced each row |
| `dbt/models/marts/fct_protocol_tvl_daily.sql` *(create)* | The TVL fact |
| `dbt/models/marts/dim_asset.sql` *(modify)* | Gains `total_supply`, `max_supply`, `supply_as_of` |
| `cryptoval/cli.py` *(modify)* | Register the two new sources; `_read_panel` left-joins the revenue, TVL and supply columns onto the asset panel |

**Stage 3a-3 — the data-dependent methods and the deliverables**

| File | Responsibility |
|---|---|
| `cryptoval/methods/multiples.py` *(modify)* | Family D remainder: `PriceToFees`, `PriceToSales`, `FdvToRevenue`, `DfdvToRevenue`, `PriceToTvl` |
| `cryptoval/methods/cohort.py` *(create)* | `CohortZScore` plus `run_cross_sectional()` — the one method shape `run_methods`' per-symbol loop cannot express |
| `cryptoval/methods/registry.py` *(modify)* | `run_methods` dispatches a method declaring `cross_sectional: true` to `run_cross_sectional` |
| `cryptoval/methods/cashflow.py` *(create)* | Family C: `EthThreeStageDcf`, `DefiNetRevenueDcf`, `BondTargetYield`, `EmissionsAdjustedRealYield`, `SupplyEffectCounterModel` |
| `cryptoval/methods/monetary.py` *(modify)* | `fit_log_linear_eoe` and `KampakisLogLinearEoe` — spec §10.2's fourth specification, which Plan 2 omitted |
| `config/methods/*.yml` *(create ×12)* | The data-dependent methods' configs, including `kampakis_log_linear_eoe.yml` |
| `docs/methods/*.md` *(regenerate)* | `make cards` rewrites all 14 existing plus the 21 new cards |
| `reports/modern-layer.md` *(create)* | The Phase 3a chapter: what the modern layer adds, where the DCF and its counter-model disagree, and the declared-gap record for the §10.5 multiples no free source supports |

---

# Stage 3a-1 — Methods on the panel we already have

## Task 1: Family F — the falsified-benchmark shelf

Spec §10.7. Both methods are published **as broken**: stock-to-flow because its post-2021 residuals
falsify it, the power law because it is the most-cited and most-contested long-horizon model. They go
first because they need no data the panel lacks, and because a shelf of deliberately weak benchmarks
is what Plan 4's out-of-sample tests need something to beat.

**Files:**
- Create: `cryptoval/methods/benchmarks.py`
- Create: `config/methods/stock_to_flow.yml`, `config/methods/power_law.yml`
- Create: `tests/test_methods_benchmarks.py`
- Modify: `requirements.txt` (pin `scipy==1.18.1`)

**Interfaces:**
- Consumes: `MethodSpec`, `Estimate`, `register`, `param_set_hash`, `load_method_specs` from
  `cryptoval.methods.{base,registry}`; `_finite` from `cryptoval.methods.cost`; `_refuse` from
  `cryptoval.methods.monetary`. All exist unchanged from Plan 2.
- Produces: `fit_log_log(x, y, min_rows)` → `dict | None` with keys `alpha`, `beta`, `beta_ci_low`,
  `beta_ci_high`, `r_squared`, `durbin_watson`, `n`; classes `StockToFlow`, `PowerLaw`. Task 3's
  `PetersonMetcalfe` and `NvmRatio` reuse `fit_log_log` rather than refitting an OLS of their own.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_methods_benchmarks.py
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.benchmarks import PowerLaw, StockToFlow, fit_log_log


def _frame(days: int = 800, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=days, freq="D", tz="UTC")
    # A deterministic exponential path: both methods have a defined answer on it.
    supply = np.linspace(1.0e7, 1.9e7, days)
    issuance_coins = np.gradient(supply)
    price = start_price * np.exp(np.linspace(0.0, 1.5, days))
    return pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": [d.date() for d in dates],
            "price_usd": price,
            "supply": supply,
            "market_cap_usd": price * supply,
            "issuance_usd": issuance_coins * price,
            "genesis_date": dt.date(2009, 1, 3),
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_fit_log_log_recovers_a_known_exponent():
    x = pd.Series(np.linspace(1.0, 100.0, 400))
    y = pd.Series(3.0 * x**1.5)
    fit = fit_log_log(x, y, min_rows=100)
    assert fit is not None
    assert fit["beta"] == pytest.approx(1.5, abs=1e-6)
    assert fit["beta_ci_low"] < fit["beta"] < fit["beta_ci_high"]
    assert fit["n"] == 400


def test_fit_log_log_refuses_below_min_rows():
    x = pd.Series(np.linspace(1.0, 10.0, 20))
    assert fit_log_log(x, 2.0 * x, min_rows=100) is None


def test_stock_to_flow_reports_beta_and_the_post_2021_label():
    estimates = StockToFlow().estimate(_frame(), dict(StockToFlow.spec.params))
    applicable = [e for e in estimates if e.applicable]
    assert applicable, "stock-to-flow should be applicable on a full synthetic series"
    last = applicable[-1]
    assert last.fair_value > 0
    assert "beta" in last.diagnostics and "r_squared" in last.diagnostics
    # The shelf's whole point: the row says it is a falsified model, on every row.
    assert last.diagnostics["status"] == "falsified_post_2021"
    assert last.inputs_used["stock_to_flow"] > 0


def test_power_law_uses_days_since_genesis_and_refuses_without_one():
    frame = _frame()
    estimates = PowerLaw().estimate(frame, dict(PowerLaw.spec.params))
    last = [e for e in estimates if e.applicable][-1]
    assert last.inputs_used["days_since_genesis"] > 4000
    assert last.diagnostics["status"] == "contested"

    no_genesis = frame.assign(genesis_date=None)
    refusals = PowerLaw().estimate(no_genesis, dict(PowerLaw.spec.params))
    assert all(not e.applicable for e in refusals)
    assert "genesis" in refusals[-1].reason


def test_neither_shelf_method_is_flagged_circular():
    from cryptoval.methods.base import circularity

    assert circularity(StockToFlow.spec) == ()
    assert circularity(PowerLaw.spec) == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_methods_benchmarks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.methods.benchmarks'`

- [ ] **Step 3: Write the configs**

```yaml
# config/methods/stock_to_flow.yml
method_id: stock_to_flow
family: F_shelf
# supply and issuance_usd are Coin Metrics columns, dense from 2015 for every asset that
# has them: the density floor is high because this method's own claim is a tight fit.
requires: [market_cap_usd, supply, issuance_usd, price_usd]
min_density: 0.9
min_rows: 730
params: {flow_window_rows: 365, min_fit_rows: 365}
citation: "PlanB (2019), Modeling Bitcoin Value with Scarcity; falsified post-2021"

failure_modes:
- 'Published here AS A FALSIFIED MODEL (spec 10.7): the 2021-2022 drawdown put price far below the
  model and it has not recovered the fit. A row from this method is a benchmark to beat, never a
  fair value to act on.'
- The regression is of one trending series on another, so a high R-squared is weak evidence; the
  Durbin-Watson statistic travels on every row for exactly that reason.
- 'Flow is a trailing 365-row sum of issuance converted to coins at each day''s price, so a price
  spike inside the window perturbs the stock-to-flow ratio through the denominator.'
- A halving is a step change in flow, so the ratio jumps discontinuously and the fit straddles
  regimes rather than describing one.
assumptions:
- Scarcity, measured as stock over annual flow, prices the asset.
- issuance_usd divided by price_usd recovers issuance in coins.

# config/methods/power_law.yml
method_id: power_law
family: F_shelf
requires: [price_usd, genesis_date]
min_density: 0.9
min_rows: 730
params: {min_fit_rows: 365}
citation: "Santostasi power law; contested - see the 2026 nine-objections paper (spec 10.7)"

failure_modes:
- 'Published here AS CONTESTED (spec 10.7), not as a model: the claimed mechanistic derivation has
  nine documented methodological objections against it, and the fit is to time alone.'
- 'Regressing log price on log days since genesis fits ANY long-lived appreciating asset: the
  explanatory variable is the calendar, so the fit cannot fail for a reason that is about the asset.'
- An asset without a configured genesis date is refused rather than given the panel's first
  observation, which would be the vendor's coverage start, not the asset's birth.
- The residuals are strongly autocorrelated by construction, so the confidence interval on beta is
  optimistic.
assumptions:
- A configured genesis date exists for the asset and is its real launch.
- Time since genesis is the explanatory variable, which is the objection as much as the model.
```

- [ ] **Step 4: Write the implementation**

```python
# cryptoval/methods/benchmarks.py
"""Family F -- the falsified and contested benchmark shelf (spec 10.7).

These methods are published AS BROKEN. Stock-to-flow's residuals falsify it after 2021;
the power law regresses price on the calendar and carries nine documented objections.
Including them visibly is more honest than omitting them, and Plan 4's out-of-sample
tests need a deliberately weak benchmark to beat -- a model that beats nothing has not
been shown to be a model.

Every row from this module carries diagnostics["status"], so a number cannot travel out
of here without its own obituary attached.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.cost import _finite
from cryptoval.methods.monetary import _refuse
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def fit_log_log(x: pd.Series, y: pd.Series, min_rows: int) -> dict[str, Any] | None:
    """OLS of log(y) on log(x), returning beta with its confidence interval.

    None rather than an exception below ``min_rows`` or when either series has too few
    strictly positive values: a caller's job is to turn that into a refusal naming the
    measurement, and an exception here would become a generic "method raised" row.

    Durbin-Watson travels with every fit because a log-log regression of two trending
    series reports a high R-squared almost regardless of whether one explains the other
    (spec 10.3's caveat). The statistic is how a reader sees that.
    """
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    frame = frame[(frame["x"] > 0) & (frame["y"] > 0)].dropna()
    if len(frame) < min_rows:
        return None

    design = sm.add_constant(np.log(frame["x"].to_numpy()))
    result = sm.OLS(np.log(frame["y"].to_numpy()), design).fit()
    low, high = result.conf_int()[1]
    return {
        "alpha": float(result.params[0]),
        "beta": float(result.params[1]),
        "beta_ci_low": float(low),
        "beta_ci_high": float(high),
        "r_squared": float(result.rsquared),
        "durbin_watson": float(sm.stats.stattools.durbin_watson(result.resid)),
        "n": int(len(frame)),
    }


@register
class StockToFlow:
    spec = _SPECS["stock_to_flow"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("flow_window_rows", 365))
        min_fit = int(params.get("min_fit_rows", 365))
        phash = param_set_hash(params)

        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        issuance_coins = pd.to_numeric(frame["issuance_usd"], errors="coerce") / price.where(price > 0)
        flow = issuance_coins.rolling(window, min_periods=window).sum()
        supply = pd.to_numeric(frame["supply"], errors="coerce")
        ratio = supply / flow.where(flow > 0)
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            # shift(1): this row's own ratio never enters the fit that values it.
            fit = fit_log_log(ratio.iloc[:position], market_cap.iloc[:position], min_fit)
            today = _finite(ratio.iloc[position])
            today_supply = _finite(row.supply)
            if fit is None:
                out.append(_refuse(self.spec, row, params, f"fewer than {min_fit} prior stock-to-flow observations"))
                continue
            if today is None or today <= 0 or today_supply is None or today_supply <= 0:
                out.append(_refuse(self.spec, row, params, "no stock-to-flow ratio or supply on this day"))
                continue
            implied_cap = float(np.exp(fit["alpha"] + fit["beta"] * np.log(today)))
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=implied_cap / today_supply,
                    band_low=float(np.exp(fit["alpha"] + fit["beta_ci_low"] * np.log(today))) / today_supply,
                    band_high=float(np.exp(fit["alpha"] + fit["beta_ci_high"] * np.log(today))) / today_supply,
                    inputs_used={"stock_to_flow": today, "supply": today_supply, "flow_window_rows": window},
                    diagnostics={**fit, "status": "falsified_post_2021"},
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class PowerLaw:
    spec = _SPECS["power_law"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        min_fit = int(params.get("min_fit_rows", 365))
        phash = param_set_hash(params)
        price = pd.to_numeric(frame["price_usd"], errors="coerce")

        genesis = frame["genesis_date"].dropna()
        if genesis.empty:
            # One refusal per row, not one per asset: the registry's grain is the row, and a
            # single refusal would silently drop the rest of the series.
            return [
                _refuse(self.spec, row, params, "no configured genesis date for this asset")
                for row in frame.itertuples(index=False)
            ]
        origin = pd.Timestamp(genesis.iloc[0])
        days = (pd.to_datetime(frame["metric_date"]) - origin).dt.days.astype("float64")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            fit = fit_log_log(days.iloc[:position], price.iloc[:position], min_fit)
            today = _finite(days.iloc[position])
            if fit is None:
                out.append(_refuse(self.spec, row, params, f"fewer than {min_fit} prior observations since genesis"))
                continue
            if today is None or today <= 0:
                out.append(_refuse(self.spec, row, params, "metric_date is not after the genesis date"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=float(np.exp(fit["alpha"] + fit["beta"] * np.log(today))),
                    band_low=float(np.exp(fit["alpha"] + fit["beta_ci_low"] * np.log(today))),
                    band_high=float(np.exp(fit["alpha"] + fit["beta_ci_high"] * np.log(today))),
                    inputs_used={"days_since_genesis": today, "genesis_date": origin.date()},
                    diagnostics={**fit, "status": "contested"},
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
```

`genesis_date` is not in the panel yet. Add it to `config/assets.yml` for the assets whose launch
date is a matter of record, and surface it through the seed and `dim_asset`, so `PowerLaw` refuses
exactly the assets whose genesis is genuinely unknown rather than inventing one from vendor coverage:

```yaml
# config/assets.yml -- add genesis_date to each asset entry, e.g.
  - {symbol: BTC,  cm_id: btc,  gecko_id: bitcoin,  sector: pow_monetary, consensus: pow,
     genesis_date: 2009-01-03}
  - {symbol: ETH,  cm_id: eth,  gecko_id: ethereum, sector: smart_contract_l1, consensus: pos,
     genesis_date: 2015-07-30, defillama_slugs: [Ethereum]}
```

```python
# cryptoval/config.py -- Asset gains the field; a malformed date fails config load, not a method.
    genesis_date: dt.date | None = None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_methods_benchmarks.py -v && .venv/bin/python -m pytest -q`
Expected: the five new tests PASS and the existing suite stays green. `make cards` is not run yet —
Task 11 regenerates every card once.

- [ ] **Step 6: Verify on live data that the shelf is actually falsified**

This step is the task's real deliverable: a shelf that quietly fits well would mean the
implementation is wrong, not that stock-to-flow works.

```bash
cd "/Users/yas/Documents/Crypto/Crypto Valuation"
.venv/bin/python -m cryptoval.cli estimate --methods stock_to_flow,power_law
.venv/bin/python - <<'PY'
import duckdb, json
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
rows = con.execute("""
  select method_id, symbol, metric_date, fair_value,
         json_extract_string(inputs_used, '$.stock_to_flow') as s2f,
         json_extract_string(diagnostics, '$.beta') as beta,
         json_extract_string(diagnostics, '$.r_squared') as r2,
         json_extract_string(diagnostics, '$.durbin_watson') as dw
  from fct_valuation_estimate
  where method_id in ('stock_to_flow','power_law') and applicable
    and metric_date = (select max(metric_date) from fct_valuation_estimate)
  order by method_id, symbol
""").fetchdf()
print(rows.to_string())
PY
```

Record in the task report, for BTC on the latest date: each method's β, R², Durbin-Watson, its fair
value, and the ratio of that fair value to the live price. Expected shape, which the report must
state as measured rather than assumed: both fits report a high R² *and* a Durbin-Watson far below
the ~1.69 no-autocorrelation threshold, and stock-to-flow's fair value sits well above price — that
divergence is the falsification, and it is the finding.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/methods/benchmarks.py config/methods/stock_to_flow.yml \
        config/methods/power_law.yml tests/test_methods_benchmarks.py \
        config/assets.yml cryptoval/config.py requirements.txt
git commit -m "feat: Family F, the falsified and contested benchmark shelf

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Family E(a) remainder — Puell, Mayer, MVRV Z-score, energy value

Spec §10.6(a) lists eight cost-approach methods. Plan 2 Task 12 built six (realized price, thermocap,
MVTV, investor cap, MVRV reversion, Hayes); the Puell multiple, the Mayer multiple, the MVRV Z-score
and Capriole's energy value remain. All four are arithmetic on columns the panel already carries, so
this is one task.

**The finding to carry into the card, not to hide:** energy value as Capriole defines it is
`2 x annual energy cost / annual coins issued`, which on a daily basis is exactly twice
`hayes_cost_per_coin` smoothed over a year. It is therefore **not independent evidence** from Hayes —
it is Hayes times two with a trailing mean. Implement it by calling `hayes_cost_per_coin`, and say so
on the card. Reimplementing the same arithmetic under a second name would manufacture a false
agreement between two methods in the §10.8 composite.

**Files:**
- Modify: `cryptoval/methods/cost.py`
- Create: `config/methods/puell_multiple.yml`, `config/methods/mayer_multiple.yml`,
  `config/methods/mvrv_z_score.yml`, `config/methods/energy_value.yml`
- Modify: `tests/test_methods_cost.py`

**Interfaces:**
- Consumes: `hayes_cost_per_coin`, `_finite` (both already in `cryptoval/methods/cost.py`),
  `_refuse`, `register`, `param_set_hash`.
- Produces: classes `PuellMultiple`, `MayerMultiple`, `MvrvZScore`, `EnergyValue`. Task 11's report
  reads their `diagnostics["ratio"]` for the dispersion table.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_methods_cost.py -- append
import numpy as np
import pandas as pd

from cryptoval.methods.cost import EnergyValue, MayerMultiple, MvrvZScore, PuellMultiple, hayes_cost_per_coin


def _cost_frame(days: int = 900) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D", tz="UTC")
    price = 100.0 + np.arange(days) * 0.5
    supply = 1.0e7 + np.arange(days) * 50.0
    return pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": [d.date() for d in dates],
            "price_usd": price,
            "supply": supply,
            "market_cap_usd": price * supply,
            "realized_cap": price * supply * 0.7,
            "issuance_usd": np.full(days, 1.0e6),
            "hash_rate": np.full(days, 1.0e9),
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_mayer_multiple_is_price_over_its_own_trailing_mean():
    frame = _cost_frame()
    estimates = MayerMultiple().estimate(frame, {"window_rows": 200, "target_multiple": 1.0})
    last = [e for e in estimates if e.applicable][-1]
    expected_ma = frame["price_usd"].iloc[-200:].mean()
    assert last.inputs_used["moving_average_price"] == pytest.approx(expected_ma)
    assert last.diagnostics["mayer_multiple"] == pytest.approx(frame["price_usd"].iloc[-1] / expected_ma)
    # target_multiple 1.0 means the fair value IS the moving average.
    assert last.fair_value == pytest.approx(expected_ma)


def test_puell_multiple_refuses_before_a_full_365_row_window():
    estimates = PuellMultiple().estimate(_cost_frame(days=300), dict(PuellMultiple.spec.params))
    assert all(not e.applicable for e in estimates)
    assert "365" in estimates[-1].reason


def test_mvrv_z_score_is_flagged_circular_because_realized_cap_is_derived():
    from cryptoval.methods.base import circularity

    assert "realized_cap" in circularity(MvrvZScore.spec)
    last = [e for e in MvrvZScore().estimate(_cost_frame(), dict(MvrvZScore.spec.params)) if e.applicable][-1]
    assert "z_score" in last.diagnostics


def test_energy_value_is_exactly_twice_smoothed_hayes():
    frame = _cost_frame()
    params = {"window_rows": 365, "joules_per_hash": 2.0e-11,
              "electricity_usd_per_kwh": 0.05, "hashes_per_unit": 1.0e12}
    last = [e for e in EnergyValue().estimate(frame, params) if e.applicable][-1]

    coins_per_day = frame["issuance_usd"] / frame["price_usd"]
    daily = [
        hayes_cost_per_coin(1.0e9, 1.0e12, 2.0e-11, 0.05, c)
        for c in coins_per_day.iloc[-365:]
    ]
    assert last.fair_value == pytest.approx(2.0 * float(np.mean(daily)))
    # The card must not claim independence from Hayes; the diagnostics say so on every row.
    assert last.diagnostics["derived_from"] == "hayes_cost_of_production"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_cost.py -k "mayer or puell or z_score or energy" -v`
Expected: FAIL — `ImportError: cannot import name 'EnergyValue' from 'cryptoval.methods.cost'`

- [ ] **Step 3: Write the configs**

```yaml
# config/methods/puell_multiple.yml
method_id: puell_multiple
family: E_cost
requires: [issuance_usd, supply, price_usd]
min_density: 0.9
min_rows: 730
params: {window_rows: 365, target_multiple: 1.0}
citation: "David Puell (2019), Puell multiple; spec 6.2 derives it from free inputs"
failure_modes:
- 'A miner-revenue oscillator, not a valuation: fair value here means "the price at which daily
  issuance revenue equals its own trailing-year average", which is a cycle indicator wearing a
  price''s clothes.'
- A halving halves issuance overnight, so the multiple steps by construction and the step is not
  information about price.
- Post-Merge ETH has no mining issuance in the miner sense, so the series describes staking issuance
  instead and the analogy to Puell's own construction breaks.
assumptions:
- Issuance revenue mean-reverts to its trailing-year average.

# config/methods/mayer_multiple.yml
method_id: mayer_multiple
family: E_cost
requires: [price_usd]
min_density: 0.9
min_rows: 400
params: {window_rows: 200, target_multiple: 1.0}
citation: "Trace Mayer, Mayer multiple (200-day moving average)"
failure_modes:
- 'The purest form of the circularity this project exists to flag: fair value is a moving average OF
  PRICE, so the method cannot disagree with price for long and its residual is a momentum signal,
  not a valuation gap. Reported as a diagnostic.'
- The 200-row window is a convention with no derivation behind it; target_multiple 1.0 is likewise
  a choice, and the estimate scales linearly in it.
- A 200-row window on a gapped series spans more than 200 calendar days, so the average is over a
  longer stretch than its name claims.
assumptions:
- Price reverts to its own 200-row moving average.

# config/methods/mvrv_z_score.yml
method_id: mvrv_z_score
family: E_cost
# realized_cap is market-cap-derived (spec 6.2), so the registry flags this method circular.
requires: [market_cap_usd, realized_cap, supply]
min_density: 0.8
min_rows: 730
params: {window_rows: 365, target_z: 0.0}
citation: "Awe and Wonder / Murad Mahmudov, MVRV Z-score; dissertation section 3.5"
failure_modes:
- 'Circular by construction: realized cap is derived from market cap through the MVRV identity
  (spec 6.2), so this scores market cap against a transform of itself.'
- The standard deviation is of market cap in dollars, so the Z-score is not comparable across assets
  of different size, nor across eras of one asset.
- A trailing-window standard deviation on a trending series understates dispersion early and
  overstates it after a crash.
assumptions:
- Market cap reverts toward realized cap, and the trailing window measures the normal spread.

# config/methods/energy_value.yml
method_id: energy_value
family: E_cost
requires: [hash_rate, issuance_usd, price_usd]
min_density: 0.9
min_rows: 400
params:
  window_rows: 365
  joules_per_hash: 2.0e-11
  electricity_usd_per_kwh: 0.05
  hashes_per_unit: 1.0e+12
citation: "Charles Edwards / Capriole (2019), energy value"
failure_modes:
- 'NOT INDEPENDENT OF HAYES. Energy value is 2 x the trailing-year mean of exactly the Hayes cost
  per coin this repo already computes, and it is implemented by calling that function. Treat the
  two as one piece of evidence; counting them twice in a composite manufactures agreement.'
- The factor of two is Capriole''s assumption about the ratio of value to energy input, not a
  measurement, and the estimate scales linearly in it.
- It inherits every Hayes failure mode: the efficiency and electricity-price parameters, the
  calibrated hash-rate unit, and proof-of-work only.
assumptions:
- Value is twice the energy cost of issuance, averaged over a trailing year.
- The same fleet efficiency and electricity price as hayes_cost_of_production.
```

- [ ] **Step 4: Write the implementation**

```python
# cryptoval/methods/cost.py -- append

@register
class MayerMultiple:
    spec = _SPECS["mayer_multiple"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 200))
        target = float(params.get("target_multiple", 1.0))
        phash = param_set_hash(params)
        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        moving = price.rolling(window, min_periods=window).mean()

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            average = _finite(moving.iloc[position])
            today = _finite(row.price_usd)
            if average is None or average <= 0:
                out.append(_refuse(self.spec, row, params, f"fewer than {window} prior price observations"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash, fair_value=average * target, band_low=None, band_high=None,
                    inputs_used={"moving_average_price": average, "window_rows": window,
                                 "target_multiple": target},
                    diagnostics={"mayer_multiple": (today / average) if today else None,
                                 "status": "diagnostic_not_fair_value"},
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class PuellMultiple:
    spec = _SPECS["puell_multiple"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        target = float(params.get("target_multiple", 1.0))
        phash = param_set_hash(params)
        issuance = pd.to_numeric(frame["issuance_usd"], errors="coerce")
        average = issuance.rolling(window, min_periods=window).mean()

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            baseline = _finite(average.iloc[position])
            today_issuance = _finite(row.issuance_usd)
            price, supply = _finite(row.price_usd), _finite(row.supply)
            if baseline is None or baseline <= 0:
                out.append(_refuse(self.spec, row, params,
                                   f"fewer than {window} prior issuance observations"))
                continue
            if today_issuance is None or today_issuance <= 0 or price is None or price <= 0:
                out.append(_refuse(self.spec, row, params, "no issuance or price on this day"))
                continue
            # Fair value = the price at which today's issuance revenue would equal its
            # trailing average, holding coins issued fixed: price x (baseline / today).
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash, fair_value=price * (baseline * target / today_issuance),
                    band_low=None, band_high=None,
                    inputs_used={"issuance_usd": today_issuance, "trailing_mean_issuance_usd": baseline,
                                 "window_rows": window, "supply": supply},
                    diagnostics={"puell_multiple": today_issuance / baseline,
                                 "status": "diagnostic_not_fair_value"},
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class MvrvZScore:
    spec = _SPECS["mvrv_z_score"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        target_z = float(params.get("target_z", 0.0))
        phash = param_set_hash(params)
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
        realized = pd.to_numeric(frame["realized_cap"], errors="coerce")
        deviation = market_cap.rolling(window, min_periods=window).std()

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            sigma = _finite(deviation.iloc[position])
            cap, real, supply = _finite(row.market_cap_usd), _finite(row.realized_cap), _finite(row.supply)
            if sigma is None or sigma <= 0:
                out.append(_refuse(self.spec, row, params,
                                   f"no market-cap standard deviation over {window} rows"))
                continue
            if cap is None or real is None or supply is None or supply <= 0:
                out.append(_refuse(self.spec, row, params, "no market cap, realized cap or supply on this day"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash, fair_value=(real + target_z * sigma) / supply,
                    band_low=None, band_high=None,
                    inputs_used={"market_cap_usd": cap, "realized_cap": real, "stdev_market_cap": sigma,
                                 "window_rows": window, "supply": supply},
                    diagnostics={"z_score": (cap - real) / sigma, "status": "circular_realized_cap"},
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class EnergyValue:
    spec = _SPECS["energy_value"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        phash = param_set_hash(params)
        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        coins_per_day = pd.to_numeric(frame["issuance_usd"], errors="coerce") / price.where(price > 0)
        hash_rate = pd.to_numeric(frame["hash_rate"], errors="coerce")

        # Capriole's energy value is 2 x annual energy cost / annual coins issued, which on a
        # daily basis is 2 x the Hayes cost per coin. Calling hayes_cost_per_coin rather than
        # restating the arithmetic is the point: the two methods are one piece of evidence.
        daily_cost = pd.Series(
            [
                hayes_cost_per_coin(
                    hash_rate.iloc[i], float(params["hashes_per_unit"]), float(params["joules_per_hash"]),
                    float(params["electricity_usd_per_kwh"]), coins_per_day.iloc[i],
                )
                for i in range(len(frame))
            ],
            dtype="float64",
        )
        smoothed = daily_cost.rolling(window, min_periods=window).mean()

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            mean_cost = _finite(smoothed.iloc[position])
            if mean_cost is None or mean_cost <= 0:
                out.append(_refuse(self.spec, row, params,
                                   f"fewer than {window} prior days with a computable production cost"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash, fair_value=2.0 * mean_cost, band_low=None, band_high=None,
                    inputs_used={"trailing_mean_production_cost": mean_cost, "window_rows": window,
                                 "joules_per_hash": params["joules_per_hash"],
                                 "electricity_usd_per_kwh": params["electricity_usd_per_kwh"]},
                    diagnostics={"derived_from": "hayes_cost_of_production", "multiplier": 2.0,
                                 "status": "not_independent_of_hayes"},
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_methods_cost.py -v && .venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 6: Measure the Hayes/energy-value relationship on live data**

```bash
.venv/bin/python -m cryptoval.cli estimate --methods hayes_cost_of_production,energy_value
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
print(con.execute("""
  select symbol,
         max(case when method_id='hayes_cost_of_production' then fair_value end) as hayes,
         max(case when method_id='energy_value' then fair_value end) as energy
  from fct_valuation_estimate
  where applicable and metric_date=(select max(metric_date) from fct_valuation_estimate)
    and method_id in ('hayes_cost_of_production','energy_value')
  group by symbol order by symbol
""").fetchdf().to_string())
PY
```

Record the energy/Hayes ratio per asset in the task report. It will not be exactly 2.0 — energy
value smooths over 365 rows and Hayes does not — and the *size* of that gap is what the card's
"not independent" claim is quantified by.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/methods/cost.py config/methods/puell_multiple.yml config/methods/mayer_multiple.yml \
        config/methods/mvrv_z_score.yml config/methods/energy_value.yml tests/test_methods_cost.py
git commit -m "feat: Family E(a) remainder - Puell, Mayer, MVRV Z-score and energy value

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Family B remainder — Peterson, NVM ratio, LPPLS bubble flag

Spec §10.3. Plan 2 Task 14 built Metcalfe with β estimated. Three remain: Peterson's generalised
Metcalfe, the NVM ratio (network value over Metcalfe-implied value — the *deviation series* is the
signal, not the level), and the LPPLS bubble flag.

**LPPLS is the one genuinely new piece of numerics in this plan.** The log-periodic power-law
singularity has a critical time `tc` that the optimiser will happily place anywhere. Review Focus
item 5 is this task's: `tc` is bounded to a declared horizon beyond the window, there is a fit-quality
floor, and when either binds the method reports "no bubble regime detected" rather than a date.

**Files:**
- Modify: `cryptoval/methods/network.py`
- Create: `config/methods/peterson_metcalfe.yml`, `config/methods/nvm_ratio.yml`,
  `config/methods/lppls_bubble_flag.yml`
- Modify: `tests/test_methods_network.py`

**Interfaces:**
- Consumes: `fit_log_log` from Task 1's `cryptoval.methods.benchmarks`; `fit_metcalfe` (already in
  `network.py` from Plan 2 Task 14); `_finite`, `_refuse`, `register`, `param_set_hash`.
- Produces: `fit_lppls(log_price, params)` → `dict | None` with keys `tc`, `m`, `omega`,
  `rmse_ratio`, `window_rows`; classes `PetersonMetcalfe`, `NvmRatio`, `LpplsBubbleFlag`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_methods_network.py -- append
import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.network import LpplsBubbleFlag, NvmRatio, PetersonMetcalfe, fit_lppls


def _network_frame(days: int = 800) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D", tz="UTC")
    addresses = np.linspace(2.0e5, 1.1e6, days)
    price = 50.0 + np.arange(days) * 0.2
    supply = np.full(days, 1.0e8)
    return pd.DataFrame(
        {
            "symbol": "ETH", "metric_date": [d.date() for d in dates],
            "active_addresses": addresses, "price_usd": price, "supply": supply,
            "market_cap_usd": price * supply, "tx_count": addresses * 3.0,
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_peterson_values_the_network_from_n_squared_and_bitcoin_days_destroyed():
    frame = _network_frame()
    last = [e for e in PetersonMetcalfe().estimate(frame, dict(PetersonMetcalfe.spec.params))
            if e.applicable][-1]
    n = frame["active_addresses"].iloc[-1]
    assert last.inputs_used["pairs"] == pytest.approx(n * (n - 1) / 2.0)
    assert last.fair_value > 0


def test_nvm_ratio_reports_the_deviation_and_is_not_a_fair_value():
    last = [e for e in NvmRatio().estimate(_network_frame(), dict(NvmRatio.spec.params))
            if e.applicable][-1]
    assert last.fair_value is None, "NVM is a deviation series, not a price"
    assert "nvm" in last.diagnostics
    assert NvmRatio.spec.returns_fair_value is False


def test_lppls_refuses_when_the_critical_time_hits_its_bound():
    # A straight line has no singularity: any tc the optimiser finds is an artefact.
    straight = pd.Series(np.log(np.linspace(100.0, 200.0, 400)))
    fit = fit_lppls(straight, {"max_horizon_rows": 180, "max_rmse_ratio": 0.02, "window_rows": 400})
    assert fit is None


def test_lppls_flag_says_no_bubble_rather_than_a_date_when_the_fit_is_poor():
    estimates = LpplsBubbleFlag().estimate(_network_frame(), dict(LpplsBubbleFlag.spec.params))
    applicable = [e for e in estimates if e.applicable]
    assert applicable, "the flag is applicable even when it detects nothing"
    assert applicable[-1].fair_value is None
    assert applicable[-1].diagnostics["bubble_regime"] is False
    assert applicable[-1].diagnostics["critical_time"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_network.py -k "peterson or nvm or lppls" -v`
Expected: FAIL — `ImportError: cannot import name 'fit_lppls'`

- [ ] **Step 3: Write the configs**

```yaml
# config/methods/peterson_metcalfe.yml
method_id: peterson_metcalfe
family: B_network
requires: [market_cap_usd, active_addresses, supply]
min_density: 0.8
min_rows: 365
params: {fit_window_rows: 365, min_fit_rows: 180}
citation: "Timothy Peterson (2018), Metcalfe's law as a model for Bitcoin's value; dissertation 3.3"
failure_modes:
- 'The dissertation specifies V = a x (n(n-1)/2 x 1/b_t) where b_t is bitcoin-days-destroyed, which
  the free tier does not serve. The constant a is fitted on the trailing window instead, so what is
  implemented is Peterson''s FUNCTIONAL FORM with an estimated scale, not his published model, and
  the card says so.'
- An active address is not a user: exchange and contract addresses inflate n, and n enters squared.
- A fitted scale constant absorbs any level error, so the fit can look good while the mechanism is
  wrong.
assumptions:
- Network value scales with the number of potential connections among active addresses.
- The scale constant is stable across the trailing window.

# config/methods/nvm_ratio.yml
method_id: nvm_ratio
family: B_network
requires: [market_cap_usd, active_addresses]
min_density: 0.8
min_rows: 365
params: {fit_window_rows: 365, min_fit_rows: 180}
returns_fair_value: false
citation: "Network value to Metcalfe ratio (spec 10.3); the deviation series is the signal"
failure_modes:
- 'Returns NO fair value by declaration: NVM is network value divided by Metcalfe-implied value, so
  publishing it as a price would be publishing the Metcalfe fit twice.'
- It inherits the Metcalfe fit''s autocorrelated residuals wholesale, so a deviation is only as
  meaningful as that fit.
- The level of NVM is not comparable across assets; only its deviation from its own history is.
assumptions:
- The Metcalfe fit on the trailing window is the right reference for today''s network value.

# config/methods/lppls_bubble_flag.yml
method_id: lppls_bubble_flag
family: B_network
requires: [price_usd]
min_density: 0.9
min_rows: 500
params:
  window_rows: 365
  # tc may sit at most this far beyond the window's end. Beyond it the "critical time"
  # is an extrapolation the data cannot support, and the flag must say so.
  max_horizon_rows: 180
  # Fit quality floor: residual RMSE as a fraction of the log-price range.
  max_rmse_ratio: 0.02
  n_restarts: 8
returns_fair_value: false
citation: "Sornette LPPLS; Wheatley et al. (2018) applied to cryptoassets (spec 10.3)"
failure_modes:
- 'The optimiser converges on SOMETHING for any monotone series. The bound on tc and the RMSE floor
  exist to turn that into a refusal; without them this method emits a crash date for every asset
  every day.'
- 'A detected bubble regime is not a forecast: LPPLS identifies a faster-than-exponential pattern,
  and the literature''s own out-of-sample record on crypto is contested.'
- The fit is on a fixed trailing window, so a regime that began before the window opens is invisible.
- Multi-start optimisation makes the result depend on the restart count; n_restarts travels on every
  row for reproducibility.
assumptions:
- A bubble is a faster-than-exponential price path with log-periodic oscillations.
- 365 rows is long enough to contain such a regime and short enough to be one regime.
```

- [ ] **Step 4: Write the implementation**

```python
# cryptoval/methods/network.py -- append

import numpy as np
from scipy.optimize import least_squares

from cryptoval.methods.benchmarks import fit_log_log


def fit_lppls(log_price: pd.Series, params: dict) -> dict | None:
    """Fit the LPPLS form to a log-price window, or return None if it is not a bubble.

    None covers four distinct cases, all of which mean "no date to report": too few rows,
    the optimiser failing, tc landing beyond max_horizon_rows (an extrapolation the window
    cannot support), and the residual RMSE exceeding max_rmse_ratio of the log-price range.
    A date returned from any of those would be a number invented by an optimiser, which is
    precisely what this project refuses to publish.
    """
    values = pd.to_numeric(log_price, errors="coerce").dropna().to_numpy()
    window = int(params.get("window_rows", 365))
    if len(values) < window:
        return None
    values = values[-window:]
    t = np.arange(len(values), dtype="float64")
    horizon = float(params.get("max_horizon_rows", 180))
    spread = float(np.ptp(values))
    if spread <= 0:
        return None

    def residual(theta):
        tc, m, omega, a, b, c1, c2 = theta
        dt = np.clip(tc - t, 1e-8, None)
        power = dt**m
        return a + b * power + power * (c1 * np.cos(omega * np.log(dt)) + c2 * np.sin(omega * np.log(dt))) - values

    best = None
    rng = np.random.default_rng(0)  # Seeded: a bubble flag must not change between runs.
    for _ in range(int(params.get("n_restarts", 8))):
        guess = [
            len(values) + rng.uniform(1.0, horizon), rng.uniform(0.1, 0.9), rng.uniform(6.0, 13.0),
            float(values[-1]), -1.0, 0.0, 0.0,
        ]
        try:
            fit = least_squares(
                residual, guess, method="trf",
                bounds=(
                    [len(values) + 1.0, 0.1, 6.0, -np.inf, -np.inf, -np.inf, -np.inf],
                    [len(values) + horizon, 0.9, 13.0, np.inf, np.inf, np.inf, np.inf],
                ),
                max_nfev=2000,
            )
        except Exception:
            continue
        if best is None or fit.cost < best.cost:
            best = fit

    if best is None:
        return None
    rmse = float(np.sqrt(2.0 * best.cost / len(values)))
    tc = float(best.x[0])
    # tc pinned to its own bound is the optimiser saying "further away than you let me look".
    if tc >= len(values) + horizon - 1e-6:
        return None
    if rmse / spread > float(params.get("max_rmse_ratio", 0.02)):
        return None
    return {"tc": tc, "m": float(best.x[1]), "omega": float(best.x[2]),
            "rmse_ratio": rmse / spread, "window_rows": window}


@register
class PetersonMetcalfe:
    spec = _SPECS["peterson_metcalfe"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("fit_window_rows", 365))
        min_fit = int(params.get("min_fit_rows", 180))
        phash = param_set_hash(params)
        addresses = pd.to_numeric(frame["active_addresses"], errors="coerce")
        pairs = addresses * (addresses - 1.0) / 2.0
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            # The trailing window ENDS at position, exclusive: no row values itself.
            start = max(0, position - window)
            history_pairs, history_cap = pairs.iloc[start:position], market_cap.iloc[start:position]
            usable = (history_pairs > 0) & (history_cap > 0)
            today_pairs, supply = _finite(pairs.iloc[position]), _finite(row.supply)
            if int(usable.sum()) < min_fit:
                out.append(_refuse(self.spec, row, params,
                                   f"only {int(usable.sum())} usable rows in the trailing window; need {min_fit}"))
                continue
            if today_pairs is None or today_pairs <= 0 or supply is None or supply <= 0:
                out.append(_refuse(self.spec, row, params, "no active addresses or supply on this day"))
                continue
            # a is the mean ratio of network value to pairs over the window: Peterson's scale
            # constant, FITTED because bitcoin-days-destroyed is not on the free tier.
            scale = float((history_cap[usable] / history_pairs[usable]).mean())
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash, fair_value=scale * today_pairs / supply,
                    band_low=None, band_high=None,
                    inputs_used={"pairs": today_pairs, "active_addresses": _finite(row.active_addresses),
                                 "scale_constant": scale, "supply": supply},
                    diagnostics={"fit_rows": int(usable.sum()), "window_rows": window,
                                 "scale_constant_is_fitted": True},
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class NvmRatio:
    spec = _SPECS["nvm_ratio"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("fit_window_rows", 365))
        min_fit = int(params.get("min_fit_rows", 180))
        phash = param_set_hash(params)
        addresses = pd.to_numeric(frame["active_addresses"], errors="coerce")
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            start = max(0, position - window)
            fit = fit_log_log(addresses.iloc[start:position], market_cap.iloc[start:position], min_fit)
            today_addresses, today_cap = _finite(addresses.iloc[position]), _finite(row.market_cap_usd)
            if fit is None:
                out.append(_refuse(self.spec, row, params,
                                   f"fewer than {min_fit} usable rows in the trailing window"))
                continue
            if today_addresses is None or today_addresses <= 0 or today_cap is None or today_cap <= 0:
                out.append(_refuse(self.spec, row, params, "no active addresses or market cap on this day"))
                continue
            implied = float(np.exp(fit["alpha"] + fit["beta"] * np.log(today_addresses)))
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash,
                    # None, not a number: declared returns_fair_value false.
                    fair_value=None, band_low=None, band_high=None,
                    inputs_used={"active_addresses": today_addresses, "market_cap_usd": today_cap,
                                 "metcalfe_implied_cap": implied},
                    diagnostics={"nvm": today_cap / implied, **fit, "status": "deviation_series"},
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class LpplsBubbleFlag:
    spec = _SPECS["lppls_bubble_flag"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        phash = param_set_hash(params)
        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        log_price = np.log(price.where(price > 0))

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            if position < window:
                out.append(_refuse(self.spec, row, params,
                                   f"only {position} prior rows; the fit window is {window}"))
                continue
            fit = fit_lppls(log_price.iloc[:position], params)
            detected = fit is not None
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash, fair_value=None, band_low=None, band_high=None,
                    inputs_used={"window_rows": window, "n_restarts": params.get("n_restarts", 8)},
                    diagnostics={
                        "bubble_regime": detected,
                        # A date ONLY when the fit survived both gates; otherwise the honest
                        # answer is None, not the optimiser's best guess.
                        "critical_time": (fit["tc"] - position) if detected else None,
                        "m": fit["m"] if detected else None,
                        "omega": fit["omega"] if detected else None,
                        "rmse_ratio": fit["rmse_ratio"] if detected else None,
                    },
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_methods_network.py -v && .venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 6: Guard the runtime before running it on the whole panel**

LPPLS runs a multi-start optimiser per row. On ~4,300 BTC rows × 8 restarts that is not free. Measure
it on one asset first and record the figure:

```bash
time .venv/bin/python -m cryptoval.cli estimate --methods lppls_bubble_flag
```

If a full-panel run exceeds ten minutes, raise `min_rows`, lower `n_restarts`, or add a
`stride_rows` parameter that fits every Nth row and forward-fills the flag between fits — and record
which, with the measured timing, in the task report. Do **not** silently reduce the restart count
below 4: a single-start LPPLS fit is the failure mode this task exists to prevent.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/methods/network.py config/methods/peterson_metcalfe.yml \
        config/methods/nvm_ratio.yml config/methods/lppls_bubble_flag.yml \
        tests/test_methods_network.py
git commit -m "feat: Family B remainder - Peterson, NVM ratio and a bounded LPPLS bubble flag

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

# Stage 3a-2 — The data Family C needs

## Task 4: Per-protocol fee and revenue history, because three days is not a cash-flow series

**This is the task Family C is blocked on.** The existing `DefiLlamaSource` reads
`/overview/fees`, which is a *snapshot*: it carries `total24h` and `total30d` per protocol, stamps
them with the run's end date, and merges one row per protocol per run. Measured on 2026-10-13,
`fct_protocol_revenue_daily` therefore holds **27 rows over 3 days for 9 symbols**. A DCF discounts a
series; there is no series.

The free fix is the endpoint the spec already cites in §6.1. Verified live on 2026-10-13:

| Call | Result |
|---|---|
| `GET /overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true` | 200, 5.0 MB, 3,002 protocols, each carrying **both** `name` and `slug` (e.g. `Aave V3` → `aave-v3`, `Sky Lending` → `sky-lending`, `Ethereum` → `ethereum`) |
| `GET /summary/fees/ethereum?dataType=dailyRevenue` | 200, 253 KB, `totalDataChart` of **4,086** `[unix_seconds, usd]` pairs, opening `[1438905600, 0]` (2015-08-07) and ending `[1791849600, 96856]` |
| `GET /summary/fees/aave-v3?dataType=dailyRevenue` | 200, 575 KB, `totalDataChart` of **1,671** pairs |

`config/assets.yml` stores display *names* (`"Aave V3"`), which is what the overview payload matches
on, while the summary endpoint needs the *slug*. The overview carries both, so one overview call
builds the name→slug map and no new configuration is needed — do not hand-write slugs.

**Files:**
- Modify: `cryptoval/sources/defillama.py`
- Create: `config/contracts/defillama_history.yml`
- Create: `tests/fixtures/defillama_summary_fees.json`, `tests/fixtures/defillama_overview_slugs.json`
- Modify: `tests/test_defillama.py`
- Modify: `cryptoval/cli.py` (`SOURCES`, `SILVER_TABLE`, `KEYS`)
- Create: `dbt/models/staging/stg_protocol_fee_history.sql`
- Modify: `dbt/models/marts/fct_protocol_revenue_daily.sql`, `dbt/models/marts/_marts.yml`

**Interfaces:**
- Consumes: `get_with_retry`, `FetchResult` from `cryptoval.sources.base`; `rollup_protocols` from
  `cryptoval.rollup` (Plan 2 Task 5, used by the dbt staging model's logic, unchanged);
  `methodology_hash` from the same module.
- Produces: `slug_map(payload) -> dict[str, str]` (display name → slug);
  `parse_fee_history(payload, column) -> pd.DataFrame` with columns
  `[protocol, metric_date, <column>]`; `trim_leading_zeros(frame, column) -> tuple[pd.DataFrame, int]`;
  `DefiLlamaHistorySource` with `name = "defillama_history"`, `contract_path =
  "config/contracts/defillama_history.yml"`. Task 7 reads the mart column `net_revenue_usd` and
  `total_fees_usd` these produce; Task 10's DCF requires them by name.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_defillama.py -- append
import json
import pathlib

import pandas as pd
import pytest

from cryptoval.sources.defillama import (
    DefiLlamaHistorySource,
    parse_fee_history,
    slug_map,
    trim_leading_zeros,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_slug_map_reads_display_name_to_slug_from_the_overview():
    payload = json.loads((FIXTURES / "defillama_overview_slugs.json").read_text())
    mapping = slug_map(payload)
    assert mapping["Aave V3"] == "aave-v3"
    assert mapping["Sky Lending"] == "sky-lending"
    assert mapping["Ethereum"] == "ethereum"


def test_parse_fee_history_reads_unix_seconds_as_utc_dates():
    payload = json.loads((FIXTURES / "defillama_summary_fees.json").read_text())
    frame = parse_fee_history(payload, "net_revenue_usd")
    assert list(frame.columns) == ["protocol", "metric_date", "net_revenue_usd"]
    assert frame["metric_date"].dt.tz is not None, "timestamps must be tz-aware UTC"
    assert frame["metric_date"].iloc[0] == pd.Timestamp("2015-08-07", tz="UTC")
    assert frame["net_revenue_usd"].iloc[-1] == pytest.approx(96856.0)
    assert frame["protocol"].unique().tolist() == ["Ethereum"]


def test_trim_leading_zeros_drops_pre_existence_but_keeps_interior_zeros():
    frame = pd.DataFrame(
        {
            "protocol": "X",
            "metric_date": pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC"),
            "net_revenue_usd": [0.0, 0.0, 12.0, 0.0, 5.0, 0.0],
        }
    )
    trimmed, dropped = trim_leading_zeros(frame, "net_revenue_usd")
    assert dropped == 2
    assert trimmed["metric_date"].iloc[0] == pd.Timestamp("2024-01-03", tz="UTC")
    # The interior zeros are real zero-fee days and must survive.
    assert trimmed["net_revenue_usd"].tolist() == [12.0, 0.0, 5.0, 0.0]


def test_trim_leading_zeros_on_an_all_zero_series_returns_nothing_rather_than_everything():
    frame = pd.DataFrame(
        {
            "protocol": "X",
            "metric_date": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "net_revenue_usd": [0.0, 0.0, 0.0],
        }
    )
    trimmed, dropped = trim_leading_zeros(frame, "net_revenue_usd")
    assert trimmed.empty and dropped == 3


def test_history_source_fetches_only_configured_protocols(monkeypatch):
    from cryptoval.config import Asset

    overview = json.loads((FIXTURES / "defillama_overview_slugs.json").read_text())
    summary = json.loads((FIXTURES / "defillama_summary_fees.json").read_text())
    called: list[str] = []

    def fake_get(client, url, params=None, **kwargs):
        called.append(url)
        return overview if "/overview/" in url else summary

    monkeypatch.setattr("cryptoval.sources.defillama.get_with_retry", fake_get)
    assets = [Asset(symbol="ETH", cm_id="eth", gecko_id="ethereum", defillama_slugs=["Ethereum"])]
    result = DefiLlamaHistorySource().fetch(
        dt.date(2015, 1, 1), dt.date(2026, 10, 13), assets, client=None
    )

    # One overview call for the slug map, then two per protocol (fees and revenue dataTypes).
    assert sum("/overview/" in u for u in called) == 1
    assert sum("/summary/fees/ethereum" in u for u in called) == 2
    assert not result.frame.empty
    assert {"protocol", "metric_date", "total_fees_usd", "net_revenue_usd"} <= set(result.frame.columns)
    # Aave V3 is in the overview but not configured on any asset: it must not be fetched.
    assert not any("aave-v3" in u for u in called)


def test_history_source_names_a_protocol_whose_slug_is_unknown(monkeypatch):
    from cryptoval.config import Asset

    overview = json.loads((FIXTURES / "defillama_overview_slugs.json").read_text())
    monkeypatch.setattr(
        "cryptoval.sources.defillama.get_with_retry",
        lambda client, url, params=None, **kw: overview,
    )
    assets = [Asset(symbol="MKR", cm_id="mkr", gecko_id="maker", defillama_slugs=["Maker"])]
    result = DefiLlamaHistorySource().fetch(
        dt.date(2015, 1, 1), dt.date(2026, 10, 13), assets, client=None
    )
    assert any("Maker" in e for e in result.errors), "an unmatched protocol name must be reported"
```

Build the two fixtures from the live payloads, trimmed to what the tests need — do not commit a 5 MB
overview:

```bash
cd /tmp && curl -s "https://api.llama.fi/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true" -o ov.json
cd "/Users/yas/Documents/Crypto/Crypto Valuation"
.venv/bin/python - <<'PY'
import json, pathlib
ov = json.load(open("/tmp/ov.json"))
keep = {"Ethereum", "Aave V3", "Sky Lending"}
small = {"protocols": [p for p in ov["protocols"] if p.get("name") in keep]}
pathlib.Path("tests/fixtures/defillama_overview_slugs.json").write_text(json.dumps(small, indent=1))
PY
curl -s "https://api.llama.fi/summary/fees/ethereum?dataType=dailyRevenue" -o /tmp/eth.json
.venv/bin/python - <<'PY'
import json, pathlib
d = json.load(open("/tmp/eth.json"))
# Keep the head (the pre-existence zeros the trim test needs) and the tail, not 4,086 points.
chart = d["totalDataChart"]
pathlib.Path("tests/fixtures/defillama_summary_fees.json").write_text(json.dumps(
    {"name": d["name"], "slug": d["slug"], "total24h": d["total24h"],
     "methodology": d.get("methodology") or {},
     "totalDataChart": chart[:5] + chart[-5:]}, indent=1))
PY
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_defillama.py -k "slug or history or trim" -v`
Expected: FAIL — `ImportError: cannot import name 'slug_map'`

- [ ] **Step 3: Write the contract**

```yaml
# config/contracts/defillama_history.yml
table: protocol_fee_history
required_columns: [protocol, metric_date, total_fees_usd, net_revenue_usd]
unique: [protocol, metric_date]
not_null: [protocol, metric_date]
max_null_fraction:
  # Revenue is absent for protocols DefiLlama tracks fees but not revenue for. Measured
  # on the configured set in Step 6; raise this only with the measurement in the commit.
  net_revenue_usd: 0.60
  total_fees_usd: 0.20
ranges:
  # Gross fees cannot be negative. Net revenue CAN be (Plan 2 Task 4's ruling: a protocol
  # paying out more than it earns is a real event), so it has no floor.
  total_fees_usd: {minimum: 0.0, maximum: 1.0e+11}
  net_revenue_usd: {maximum: 1.0e+11}
freshness:
  column: metric_date
  max_age_hours: 72
```

- [ ] **Step 4: Write the implementation**

```python
# cryptoval/sources/defillama.py -- append

SUMMARY_PATH = "/summary/fees"
HISTORY_COLUMNS = ["protocol", "metric_date", "total_fees_usd", "net_revenue_usd"]


def slug_map(payload: dict) -> dict[str, str]:
    """Display name -> URL slug, from the overview payload.

    The configured identity in assets.yml is the display name, because that is what the
    overview matches on; the summary endpoint is addressed by slug. Both live in the same
    payload, so the mapping is read rather than hand-maintained -- a hand-written slug is
    one rebrand away from a silent 404.
    """
    return {
        record["name"]: record["slug"]
        for record in payload.get("protocols", []) or []
        if record.get("name") and record.get("slug")
    }


def parse_fee_history(payload: dict, column: str) -> pd.DataFrame:
    """One protocol's daily series from a /summary/fees response.

    DefiLlama returns [unix_seconds, usd] pairs. Parsed with unit="s", utc=True: a naive
    timestamp here would shift every cash flow by up to a day against the price panel.
    """
    chart = payload.get("totalDataChart") or []
    if not chart:
        return pd.DataFrame(columns=["protocol", "metric_date", column])
    frame = pd.DataFrame(chart, columns=["ts", column])
    frame["metric_date"] = pd.to_datetime(frame["ts"], unit="s", utc=True).dt.normalize()
    frame["protocol"] = payload.get("name")
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["protocol", "metric_date", column]]


def trim_leading_zeros(frame: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Drop the pre-existence zeros at the head of a series, keeping interior zeros.

    DefiLlama's chart starts at the DATASET's start, not the protocol's: Ethereum's
    revenue series opens in August 2015 with zeros. Those are "this protocol did not
    exist", and a DCF that averages them understates the protocol's economics for as long
    as the window reaches back. An INTERIOR zero is different -- a genuine zero-fee day,
    or an adapter outage -- and is kept, because dropping it would silently shorten the
    series that later rows average over.
    """
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    nonzero = values.to_numpy().nonzero()[0]
    if len(nonzero) == 0:
        return frame.iloc[0:0], len(frame)
    first = int(nonzero[0])
    return frame.iloc[first:].reset_index(drop=True), first


class DefiLlamaHistorySource:
    """Per-protocol daily fee and revenue HISTORY (spec 10.4's denominator discipline).

    Separate from DefiLlamaSource rather than folded into it: that source's row grain is
    one snapshot row per protocol per run, this one's is one row per protocol per DAY of
    history, and merging two grains into one silver table would make the SCD2 merge
    compare a 4,086-row backfill against a 1-row snapshot.
    """

    name = "defillama_history"
    contract_path = "config/contracts/defillama_history.yml"

    def __init__(self, max_retries: int = 5) -> None:
        self.max_retries = max_retries

    def fetch(self, start, end, assets, client) -> FetchResult:
        overview = get_with_retry(client, f"{BASE_URL}{OVERVIEW_PATH}", max_retries=self.max_retries)
        slugs = slug_map(overview)

        wanted: list[str] = []
        for asset in assets:
            wanted.extend(asset.defillama_slugs or [])

        frames: list[pd.DataFrame] = []
        errors: list[str] = []
        rows_in = 0
        for protocol in sorted(set(wanted)):
            slug = slugs.get(protocol)
            if slug is None:
                # A configured protocol the feed no longer lists: name it. Plan 2 Task 5
                # already proved this happens (Maker -> Sky Lending).
                errors.append(f"no DefiLlama slug for configured protocol {protocol!r}")
                continue
            series: dict[str, pd.DataFrame] = {}
            for column, data_type in (("total_fees_usd", None), ("net_revenue_usd", "dailyRevenue")):
                url = f"{BASE_URL}{SUMMARY_PATH}/{slug}"
                payload = get_with_retry(
                    client, url, {"dataType": data_type} if data_type else None,
                    max_retries=self.max_retries,
                )
                parsed = parse_fee_history(payload, column)
                rows_in += len(parsed)
                trimmed, dropped = trim_leading_zeros(parsed, column)
                if dropped:
                    errors.append(f"{protocol}/{column}: trimmed {dropped} leading zero rows (pre-existence)")
                if trimmed.empty:
                    errors.append(f"{protocol}/{column}: series is entirely zero or empty")
                    continue
                # The configured display name, not the payload's: the rollup joins on it.
                series[column] = trimmed.assign(protocol=protocol)

            if "total_fees_usd" not in series and "net_revenue_usd" not in series:
                continue
            merged = None
            for frame in series.values():
                merged = frame if merged is None else merged.merge(
                    frame, on=["protocol", "metric_date"], how="outer"
                )
            frames.append(merged)

        if not frames:
            return FetchResult(pd.DataFrame(columns=HISTORY_COLUMNS), rows_in, tuple(errors))
        combined = pd.concat(frames, ignore_index=True).reindex(columns=HISTORY_COLUMNS)
        return FetchResult(combined, rows_in, tuple(errors))
```

Wire it into the CLI beside the existing sources:

```python
# cryptoval/cli.py
SOURCES = {
    "coinmetrics": CoinMetricsSource(),
    "defillama": DefiLlamaSource(),
    "defillama_history": DefiLlamaHistorySource(),
    "coingecko": CoinGeckoSource(),
    "equities": EquitySource(),
}
SILVER_TABLE = {..., "defillama_history": "protocol_fee_history"}
KEYS = {..., "defillama_history": ["protocol", "metric_date"]}
```

- [ ] **Step 5: Write the dbt models**

```sql
-- dbt/models/staging/stg_protocol_fee_history.sql
-- The history series resolved to the conformed symbol. Same rollup rule as the snapshot
-- model: sum a symbol's configured protocols, which seeds/assets.csv has already been
-- checked for aggregate/component collisions (Plan 2 Task 5).
with history as (
    select protocol, cast(metric_date as date) as metric_date,
           total_fees_usd, net_revenue_usd
    from {{ source('silver', 'protocol_fee_history') }}
    where is_current
),
mapped as (
    select h.*, a.symbol
    from history h
    join {{ ref('protocol_symbol_map') }} a on a.protocol = h.protocol
)
select symbol, metric_date,
       sum(total_fees_usd)  as total_fees_usd,
       sum(net_revenue_usd) as net_revenue_usd,
       count(*)             as protocol_count
from mapped
group by symbol, metric_date
```

```sql
-- dbt/models/marts/fct_protocol_revenue_daily.sql -- modify
-- The history series is the spine; the snapshot supplies only today, where the history
-- endpoint has not yet published. series_source names which produced each row, because a
-- DCF built half on one and half on the other is not reproducible.
with history as (
    select symbol, metric_date, total_fees_usd, net_revenue_usd, protocol_count,
           'history' as series_source
    from {{ ref('stg_protocol_fee_history') }}
),
snapshot as (
    select symbol, metric_date, total_fees_24h as total_fees_usd,
           net_revenue_24h as net_revenue_usd, protocol_count,
           'snapshot' as series_source
    from {{ ref('stg_protocol_revenue') }}
    where (symbol, metric_date) not in (select symbol, metric_date from history)
)
select *, {{ revenue_regime() }} as revenue_regime from history
union all
select *, {{ revenue_regime() }} as revenue_regime from snapshot
```

Keep the existing `revenue_regime` logic exactly as Plan 2 Task 8 wrote it — extract it to a macro
if it is currently inline, so both branches classify identically rather than diverging.

- [ ] **Step 6: Backfill and measure what arrived**

```bash
.venv/bin/python -m cryptoval.cli ingest --sources defillama_history --start 2015-01-01 --end 2026-10-13
make dbt
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
print(con.execute("""
  select symbol, series_source, count(*) rows, min(metric_date) first, max(metric_date) last,
         round(100.0*sum(case when net_revenue_usd is null then 1 else 0 end)/count(*),1) as pct_rev_null
  from fct_protocol_revenue_daily group by 1,2 order by 1,2
""").fetchdf().to_string())
PY
.venv/bin/python -m cryptoval.cli validate; echo "exit=$?"
```

Record per symbol: row count, first and last date, and the net-revenue null percentage. Then set
`max_null_fraction.net_revenue_usd` in the contract to the measured worst case plus headroom, and
state both numbers in the commit message. Expected from the live probe: ETH ≈ 4,086 rows from
2015-08, AAVE ≈ 1,671 from 2022-03. If any configured protocol returns nothing, its name is already
in `result.errors` — carry it into the task report rather than leaving it to a later DCF refusal.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/sources/defillama.py config/contracts/defillama_history.yml \
        tests/test_defillama.py tests/fixtures/defillama_summary_fees.json \
        tests/fixtures/defillama_overview_slugs.json cryptoval/cli.py \
        dbt/models/staging/stg_protocol_fee_history.sql \
        dbt/models/marts/fct_protocol_revenue_daily.sql dbt/models/marts/_marts.yml
git commit -m "feat: per-protocol fee and revenue history, so a DCF has a series to discount

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: Supply metadata, so FDV is a measurement rather than an assumption

Spec §10.4's dFDV and §10.5's FDV/revenue need total and maximum supply. The panel carries only
`supply` (Coin Metrics `SplyCur`, the circulating series). CoinGecko's `/coins/markets` serves all
three plus FDV in one call — verified live 2026-10-13, 1.6 KB for two ids:

| id | circulating | total | max | FDV |
|---|---|---|---|---|
| ethereum | 122,130,709.7 | 122,130,709.7 | **null** | 305,812,878,878 |
| aave | 15,436,128.4 | 16,000,000.0 | 16,000,000.0 | 2,572,237,752 |

ETH's null `max_supply` is Review Focus item 3 and this task's load-bearing test: there is no maximum
supply for ETH, so an FDV-based multiple for ETH does not exist, and substituting `total_supply`
would publish a market-cap multiple under an FDV label.

This is **metadata, not a series**: CoinGecko's free tier serves current supply figures, not supply
history. It therefore lands on `dim_asset` with a `supply_as_of` date, and every method consuming it
records that date in `inputs_used` so a reader can see the figure is point-in-time.

**Files:**
- Modify: `cryptoval/sources/prices.py`, `cryptoval/cli.py`
- Create: `config/contracts/coingecko_supply.yml`, `tests/fixtures/coingecko_markets.json`
- Modify: `tests/test_prices.py`, `dbt/models/marts/dim_asset.sql`, `dbt/models/marts/_marts.yml`
- Create: `dbt/models/staging/stg_supply_metadata.sql`

**Interfaces:**
- Produces: `parse_supply_metadata(payload) -> pd.DataFrame` with columns
  `[gecko_id, circulating_supply, total_supply, max_supply, fully_diluted_valuation, supply_as_of]`;
  `CoinGeckoSupplySource` with `name = "coingecko_supply"`. Task 7 joins the mart columns
  `total_supply`, `max_supply`, `supply_as_of`; Task 8's `FdvToRevenue` and `DfdvToRevenue` require
  `max_supply`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prices.py -- append
import json
import pathlib

import pandas as pd
import pytest

from cryptoval.sources.prices import parse_supply_metadata

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_parse_supply_metadata_keeps_a_null_max_supply_null():
    payload = json.loads((FIXTURES / "coingecko_markets.json").read_text())
    frame = parse_supply_metadata(payload).set_index("gecko_id")

    assert frame.loc["aave", "max_supply"] == pytest.approx(16_000_000.0)
    # ETH has NO maximum supply. Not zero, not total_supply: null.
    assert pd.isna(frame.loc["ethereum", "max_supply"])
    assert frame.loc["ethereum", "total_supply"] == pytest.approx(122_130_709.73792061)
    assert frame["supply_as_of"].notna().all()


def test_parse_supply_metadata_never_falls_back_to_total_supply():
    payload = [{"id": "x", "circulating_supply": 10.0, "total_supply": 20.0, "max_supply": None,
                "fully_diluted_valuation": None, "market_cap": 100.0}]
    frame = parse_supply_metadata(payload)
    assert pd.isna(frame["max_supply"].iloc[0])
    assert frame["total_supply"].iloc[0] == pytest.approx(20.0)
```

Build the fixture:

```bash
curl -s "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=ethereum,aave&sparkline=false" \
  -o "tests/fixtures/coingecko_markets.json"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prices.py -k supply -v`
Expected: FAIL — `ImportError: cannot import name 'parse_supply_metadata'`

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/sources/prices.py -- append
SUPPLY_COLUMNS = [
    "gecko_id", "circulating_supply", "total_supply", "max_supply",
    "fully_diluted_valuation", "supply_as_of",
]


def parse_supply_metadata(payload: list[dict]) -> pd.DataFrame:
    """Supply metadata from /coins/markets.

    max_supply stays NULL when the asset has none (ETH). The tempting fallback to
    total_supply would turn an FDV multiple into a market-cap multiple wearing an FDV
    label -- the exact mislabelling spec 10.4's denominator discipline forbids.
    """
    rows = [
        {
            "gecko_id": record.get("id"),
            "circulating_supply": record.get("circulating_supply"),
            "total_supply": record.get("total_supply"),
            "max_supply": record.get("max_supply"),
            "fully_diluted_valuation": record.get("fully_diluted_valuation"),
        }
        for record in payload or []
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=SUPPLY_COLUMNS)
    for column in ("circulating_supply", "total_supply", "max_supply", "fully_diluted_valuation"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # A snapshot, and labelled as one: the free tier has no supply history.
    frame["supply_as_of"] = pd.Timestamp.now(tz="UTC").normalize()
    return frame.reindex(columns=SUPPLY_COLUMNS)


class CoinGeckoSupplySource:
    name = "coingecko_supply"
    contract_path = "config/contracts/coingecko_supply.yml"

    def __init__(self, max_retries: int = 5) -> None:
        self.max_retries = max_retries

    def fetch(self, start, end, assets, client) -> FetchResult:
        ids = sorted({a.gecko_id for a in assets if a.gecko_id})
        if not ids:
            return FetchResult(pd.DataFrame(columns=SUPPLY_COLUMNS), 0, ("no gecko_id configured",))
        # One call for the whole universe: /coins/markets takes a comma-separated id list,
        # which keeps this inside the keyless tier's rate limit without the per-asset sleep
        # the market_chart source needs.
        payload = get_with_retry(
            client, f"{BASE_URL}/coins/markets",
            {"vs_currency": "usd", "ids": ",".join(ids), "sparkline": "false"},
            max_retries=self.max_retries, initial_delay=2.0,
        )
        frame = parse_supply_metadata(payload)
        missing = sorted(set(ids) - set(frame["gecko_id"]))
        errors = (f"no market row returned for: {missing}",) if missing else ()
        return FetchResult(frame, len(payload or []), errors)
```

```yaml
# config/contracts/coingecko_supply.yml
table: supply_metadata
required_columns: [gecko_id, circulating_supply, total_supply, supply_as_of]
unique: [gecko_id]
not_null: [gecko_id, supply_as_of]
max_null_fraction:
  # Many assets legitimately have no maximum supply, so a high null rate here is correct,
  # not a defect. It is listed so a column going WHOLLY null still trips the gate.
  max_supply: 0.90
  total_supply: 0.20
ranges:
  circulating_supply: {minimum: 0.0}
  total_supply: {minimum: 0.0}
  max_supply: {minimum: 0.0}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prices.py -v && .venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 5: Wire into the CLI and dim_asset**

```python
# cryptoval/cli.py
SOURCES = {..., "coingecko_supply": CoinGeckoSupplySource()}
SILVER_TABLE = {..., "coingecko_supply": "supply_metadata"}
KEYS = {..., "coingecko_supply": ["gecko_id"]}
```

```sql
-- dbt/models/staging/stg_supply_metadata.sql
select a.symbol, s.circulating_supply, s.total_supply, s.max_supply,
       s.fully_diluted_valuation, cast(s.supply_as_of as date) as supply_as_of
from {{ source('silver', 'supply_metadata') }} s
join {{ ref('assets') }} a on a.gecko_id = s.gecko_id
where s.is_current
```

```sql
-- dbt/models/marts/dim_asset.sql -- add to the select list
       m.total_supply, m.max_supply, m.supply_as_of
-- ... left join {{ ref('stg_supply_metadata') }} m on m.symbol = a.symbol
```

A LEFT join: an asset with no CoinGecko row keeps its `dim_asset` row with null supply metadata, so
Plan 2's "one row per configured symbol including SOL" criterion still holds.

- [ ] **Step 6: Ingest and verify the ETH null survives the round trip**

```bash
.venv/bin/python -m cryptoval.cli ingest --sources coingecko_supply --start 2026-10-13 --end 2026-10-13
make dbt
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
print(con.execute("""
  select symbol, total_supply, max_supply, supply_as_of,
         case when max_supply is null then 'NO MAX (FDV undefined)' else 'has max' end as fdv_status
  from dim_asset order by symbol
""").fetchdf().to_string())
PY
```

Record in the task report how many of the 31 configured symbols have a max supply and which do not.
ETH must read `NO MAX (FDV undefined)`. If it reads a number, the fallback this task forbids has
crept in somewhere between the parser and the mart — find it before moving on.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/sources/prices.py cryptoval/cli.py config/contracts/coingecko_supply.yml \
        tests/test_prices.py tests/fixtures/coingecko_markets.json \
        dbt/models/staging/stg_supply_metadata.sql dbt/models/marts/dim_asset.sql \
        dbt/models/marts/_marts.yml
git commit -m "feat: supply metadata, with a null max supply kept null

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: Protocol TVL history, behind a payload-size guard

Spec §10.5 lists P/TVL. TVL history is free but **expensive to download**: verified live 2026-10-13,
`GET /protocol/aave-v3` returns **30.2 MB** (200 OK) for a `tvl` array of 1,676
`{date, totalLiquidityUSD}` points — the bulk of the payload is the per-chain and per-token
breakdowns we do not use, and DefiLlama offers no parameter to exclude them. Across ~20 configured
protocols that is roughly 600 MB for one backfill.

**Decide before implementing, and record the decision in the task report.** P/TVL is one multiple of
the six in §10.5. If 600 MB of one-time download is acceptable, implement this task as written: the
backfill runs once, lands in bronze, and daily runs top up from the small snapshot. If it is not,
**skip this task** and record P/TVL in Task 11's declared-gap list with this measurement as the
reason — a documented refusal with a number attached is a legitimate outcome here, and is cheaper
than a fact table serving one method. Do not compromise by sampling the series: a TVL series with
holes produces a P/TVL whose denominator jumps for reasons that are not about the protocol.

If implementing:

**Files:**
- Modify: `cryptoval/sources/defillama.py`, `cryptoval/cli.py`
- Create: `config/contracts/defillama_tvl.yml`, `tests/fixtures/defillama_protocol_tvl.json`
- Modify: `tests/test_defillama.py`
- Create: `dbt/models/staging/stg_protocol_tvl.sql`, `dbt/models/marts/fct_protocol_tvl_daily.sql`
- Modify: `dbt/models/marts/_marts.yml`

**Interfaces:**
- Produces: `parse_tvl_history(payload) -> pd.DataFrame` with columns
  `[protocol, metric_date, tvl_usd]`; `DefiLlamaTvlSource` with `name = "defillama_tvl"` and a
  `max_payload_mb` constructor argument. Task 8's `PriceToTvl` requires the mart column `tvl_usd`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_defillama.py -- append
import json
import pathlib

import pandas as pd
import pytest

from cryptoval.sources.defillama import DefiLlamaTvlSource, parse_tvl_history


def test_parse_tvl_history_reads_the_tvl_array():
    payload = json.loads((pathlib.Path(__file__).parent / "fixtures" / "defillama_protocol_tvl.json").read_text())
    frame = parse_tvl_history(payload)
    assert list(frame.columns) == ["protocol", "metric_date", "tvl_usd"]
    assert frame["metric_date"].dt.tz is not None
    assert (frame["tvl_usd"] > 0).all()
    assert frame["protocol"].unique().tolist() == ["Aave V3"]


def test_tvl_source_refuses_a_payload_above_its_size_guard(monkeypatch):
    from cryptoval.config import Asset

    big = {"name": "Aave V3", "tvl": [{"date": 1_700_000_000, "totalLiquidityUSD": 1.0}],
           "padding": "x" * 2_000_000}
    monkeypatch.setattr(
        "cryptoval.sources.defillama.get_with_retry",
        lambda client, url, params=None, **kw: {"protocols": [{"name": "Aave V3", "slug": "aave-v3"}]}
        if "/overview/" in url else big,
    )
    assets = [Asset(symbol="AAVE", cm_id="aave", gecko_id="aave", defillama_slugs=["Aave V3"])]
    result = DefiLlamaTvlSource(max_payload_mb=1.0).fetch(
        dt.date(2020, 1, 1), dt.date(2026, 10, 13), assets, client=None
    )
    assert result.frame.empty
    assert any("exceeds the" in e and "MB" in e for e in result.errors)
```

Build the fixture without committing 30 MB:

```bash
curl -s "https://api.llama.fi/protocol/aave-v3" -o /tmp/tvl.json
cd "/Users/yas/Documents/Crypto/Crypto Valuation"
.venv/bin/python - <<'PY'
import json, pathlib
d = json.load(open("/tmp/tvl.json"))
pathlib.Path("tests/fixtures/defillama_protocol_tvl.json").write_text(json.dumps(
    {"name": d["name"], "slug": d.get("slug"), "tvl": d["tvl"][-10:]}, indent=1))
PY
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_defillama.py -k tvl -v`
Expected: FAIL — `ImportError: cannot import name 'parse_tvl_history'`

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/sources/defillama.py -- append
PROTOCOL_PATH = "/protocol"
TVL_COLUMNS = ["protocol", "metric_date", "tvl_usd"]


def parse_tvl_history(payload: dict) -> pd.DataFrame:
    rows = payload.get("tvl") or []
    if not rows:
        return pd.DataFrame(columns=TVL_COLUMNS)
    frame = pd.DataFrame(rows)
    frame["metric_date"] = pd.to_datetime(frame["date"], unit="s", utc=True).dt.normalize()
    frame["tvl_usd"] = pd.to_numeric(frame["totalLiquidityUSD"], errors="coerce")
    frame["protocol"] = payload.get("name")
    return frame[TVL_COLUMNS]


class DefiLlamaTvlSource:
    """Protocol TVL history, with a payload-size guard.

    /protocol/<slug> carries per-chain and per-token breakdowns we do not use and offers
    no way to exclude them: aave-v3 measured 30.2 MB on 2026-10-13 for 1,676 usable
    points. The guard exists so a vendor payload that grows by an order of magnitude
    fails loudly with the measured size rather than quietly consuming the run.
    """

    name = "defillama_tvl"
    contract_path = "config/contracts/defillama_tvl.yml"

    def __init__(self, max_retries: int = 5, max_payload_mb: float = 64.0) -> None:
        self.max_retries = max_retries
        self.max_payload_mb = max_payload_mb

    def fetch(self, start, end, assets, client) -> FetchResult:
        overview = get_with_retry(client, f"{BASE_URL}{OVERVIEW_PATH}", max_retries=self.max_retries)
        slugs = slug_map(overview)
        wanted = sorted({p for a in assets for p in (a.defillama_slugs or [])})

        frames, errors, rows_in = [], [], 0
        for protocol in wanted:
            slug = slugs.get(protocol)
            if slug is None:
                errors.append(f"no DefiLlama slug for configured protocol {protocol!r}")
                continue
            payload = get_with_retry(client, f"{BASE_URL}{PROTOCOL_PATH}/{slug}", max_retries=self.max_retries)
            size_mb = len(json.dumps(payload)) / 1_048_576
            if size_mb > self.max_payload_mb:
                errors.append(f"{protocol}: payload {size_mb:.1f} MB exceeds the {self.max_payload_mb} MB guard")
                continue
            parsed = parse_tvl_history(payload)
            rows_in += len(parsed)
            if parsed.empty:
                errors.append(f"{protocol}: no tvl array in the payload")
                continue
            frames.append(parsed.assign(protocol=protocol))

        if not frames:
            return FetchResult(pd.DataFrame(columns=TVL_COLUMNS), rows_in, tuple(errors))
        return FetchResult(pd.concat(frames, ignore_index=True), rows_in, tuple(errors))
```

`import json` at the top of the module if it is not already there.

```yaml
# config/contracts/defillama_tvl.yml
table: protocol_tvl
required_columns: [protocol, metric_date, tvl_usd]
unique: [protocol, metric_date]
not_null: [protocol, metric_date]
max_null_fraction: {tvl_usd: 0.10}
ranges:
  tvl_usd: {minimum: 0.0, maximum: 1.0e+12}
freshness: {column: metric_date, max_age_hours: 72}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_defillama.py -v && .venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 5: The dbt models**

```sql
-- dbt/models/staging/stg_protocol_tvl.sql
with tvl as (
    select protocol, cast(metric_date as date) as metric_date, tvl_usd
    from {{ source('silver', 'protocol_tvl') }} where is_current
)
select a.symbol, t.metric_date, sum(t.tvl_usd) as tvl_usd, count(*) as protocol_count
from tvl t join {{ ref('protocol_symbol_map') }} a on a.protocol = t.protocol
group by a.symbol, t.metric_date
```

```sql
-- dbt/models/marts/fct_protocol_tvl_daily.sql
select symbol, metric_date, tvl_usd, protocol_count from {{ ref('stg_protocol_tvl') }}
```

- [ ] **Step 6: Backfill once and record the real cost**

```bash
time .venv/bin/python -m cryptoval.cli ingest --sources defillama_tvl --start 2015-01-01 --end 2026-10-13
make dbt && .venv/bin/python -m cryptoval.cli validate; echo "exit=$?"
```

Record the wall-clock time, the total bytes downloaded, and rows per symbol. State in the task
report whether the measured cost matched the 600 MB estimate above.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/sources/defillama.py cryptoval/cli.py config/contracts/defillama_tvl.yml \
        tests/test_defillama.py tests/fixtures/defillama_protocol_tvl.json \
        dbt/models/staging/stg_protocol_tvl.sql dbt/models/marts/fct_protocol_tvl_daily.sql \
        dbt/models/marts/_marts.yml
git commit -m "feat: protocol TVL history behind a payload-size guard

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: Widen the frame the registry sees, without touching the method contract

`run_estimate` builds its panel from one table:

```python
# cryptoval/cli.py:317-327 as it stands
return con.execute("select * from fct_asset_metrics_daily").fetchdf()
```

Every Family C and remaining-D method needs columns from three other places: revenue from
`fct_protocol_revenue_daily`, TVL from `fct_protocol_tvl_daily`, supply metadata and `sector` and
`genesis_date` from `dim_asset`. The lazy and correct change is to widen the frame here, in one
place, with LEFT joins on the conformed `(symbol, metric_date)` key. **Nothing about the method
contract changes**: a method reaches a new column by naming it in `requires`, and
`resolve_applicability` already refuses an absent or sparse column with a measurement. No new
abstraction, no per-family loaders.

The join must be LEFT and must not change the asset panel's grain. A protocol with two revenue rows
for one symbol-date would fan out the panel and silently double-count every Family A estimate, so
the test below pins the row count.

**Files:**
- Modify: `cryptoval/cli.py` (`_read_panel`)
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: the marts from Tasks 4-6.
- Produces: `_read_panel` returns the asset panel plus `total_fees_usd`, `net_revenue_usd`,
  `revenue_regime`, `protocol_count`, `tvl_usd`, `total_supply`, `max_supply`, `supply_as_of`,
  `sector`, `genesis_date`. Tasks 8, 9 and 10 declare these in `requires`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py -- append
import pandas as pd

from cryptoval.cli import _widen_panel


def test_widen_panel_left_joins_without_changing_the_row_count():
    panel = pd.DataFrame(
        {
            "symbol": ["ETH", "ETH", "BTC"],
            "metric_date": pd.to_datetime(["2026-10-11", "2026-10-12", "2026-10-12"]).date,
            "price_usd": [2500.0, 2550.0, 80000.0],
        }
    )
    revenue = pd.DataFrame(
        {
            "symbol": ["ETH"], "metric_date": pd.to_datetime(["2026-10-12"]).date,
            "total_fees_usd": [1.0e6], "net_revenue_usd": [9.0e5],
            "revenue_regime": ["retains_revenue"], "protocol_count": [1],
        }
    )
    tvl = pd.DataFrame({"symbol": ["ETH"], "metric_date": pd.to_datetime(["2026-10-12"]).date,
                        "tvl_usd": [5.0e10]})
    assets = pd.DataFrame({"symbol": ["ETH", "BTC"], "sector": ["smart_contract_l1", "pow_monetary"],
                           "total_supply": [1.2e8, 1.98e7], "max_supply": [None, 2.1e7],
                           "supply_as_of": pd.to_datetime(["2026-10-13", "2026-10-13"]).date,
                           "genesis_date": pd.to_datetime(["2015-07-30", "2009-01-03"]).date})

    wide = _widen_panel(panel, revenue, tvl, assets)

    # The asset panel's grain is sacred: three rows in, three rows out.
    assert len(wide) == 3
    assert wide.set_index(["symbol", "metric_date"]).loc[("ETH", panel["metric_date"].iloc[1]), "net_revenue_usd"] == 9.0e5
    # BTC has no protocol revenue: null, not zero.
    assert pd.isna(wide.loc[wide["symbol"] == "BTC", "net_revenue_usd"]).all()
    # BTC's max supply survives the join; ETH's null stays null.
    assert wide.loc[wide["symbol"] == "BTC", "max_supply"].iloc[0] == 2.1e7
    assert pd.isna(wide.loc[wide["symbol"] == "ETH", "max_supply"]).all()


def test_widen_panel_refuses_a_duplicated_join_key_rather_than_fanning_out():
    panel = pd.DataFrame({"symbol": ["ETH"], "metric_date": pd.to_datetime(["2026-10-12"]).date,
                          "price_usd": [2550.0]})
    doubled = pd.DataFrame(
        {"symbol": ["ETH", "ETH"], "metric_date": pd.to_datetime(["2026-10-12", "2026-10-12"]).date,
         "total_fees_usd": [1.0, 2.0], "net_revenue_usd": [1.0, 2.0],
         "revenue_regime": ["retains_revenue"] * 2, "protocol_count": [1, 1]}
    )
    assets = pd.DataFrame({"symbol": ["ETH"], "sector": ["smart_contract_l1"], "total_supply": [1.0],
                           "max_supply": [None], "supply_as_of": pd.to_datetime(["2026-10-13"]).date,
                           "genesis_date": pd.to_datetime(["2015-07-30"]).date})
    with pytest.raises(ValueError, match="duplicate"):
        _widen_panel(panel, doubled, pd.DataFrame(columns=["symbol", "metric_date", "tvl_usd"]), assets)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -k widen -v`
Expected: FAIL — `ImportError: cannot import name '_widen_panel'`

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/cli.py -- replace _read_panel and add _widen_panel

def _widen_panel(
    panel: pd.DataFrame,
    revenue: pd.DataFrame,
    tvl: pd.DataFrame,
    assets: pd.DataFrame,
) -> pd.DataFrame:
    """The asset panel plus the protocol, TVL and dimension columns, LEFT joined.

    One place, so a method reaches a new column by naming it in ``requires`` and the
    method contract does not grow a second shape. LEFT, so an asset with no protocol
    revenue keeps its row and its Family A estimates: a missing denominator is the
    registry's refusal to make, not this function's.

    A duplicated join key RAISES rather than fanning out. A silent fan-out would multiply
    every estimate for that symbol-date, and a doubled QTM estimate looks exactly like a
    real one.
    """
    wide = panel.copy()
    for frame, name in ((revenue, "protocol revenue"), (tvl, "protocol TVL")):
        if frame.empty:
            continue
        duplicated = frame.duplicated(subset=["symbol", "metric_date"]).sum()
        if duplicated:
            raise ValueError(
                f"{name} has {duplicated} duplicate (symbol, metric_date) row(s); joining it "
                f"would fan out the panel and double-count every estimate for those days"
            )
        wide = wide.merge(frame, on=["symbol", "metric_date"], how="left")

    if not assets.empty:
        if assets.duplicated(subset=["symbol"]).any():
            raise ValueError("dim_asset has duplicate symbols; the panel join is not well defined")
        wide = wide.merge(assets, on="symbol", how="left")
    return wide


def _read_panel(settings: Settings) -> pd.DataFrame:
    """The conformed panel every method reads, widened with the Family C inputs."""
    import duckdb

    con = duckdb.connect(str(pathlib.Path(settings.data_root) / "cryptoval.duckdb"), read_only=True)
    try:
        def maybe(sql: str, columns: list[str]) -> pd.DataFrame:
            # A mart that does not exist yet (TVL, if Task 6 was skipped) yields an empty
            # frame, and the methods requiring its columns refuse with a measurement.
            try:
                return con.execute(sql).fetchdf()
            except duckdb.CatalogException:
                return pd.DataFrame(columns=columns)

        panel = con.execute("select * from fct_asset_metrics_daily").fetchdf()
        revenue = maybe(
            "select symbol, metric_date, total_fees_usd, net_revenue_usd, revenue_regime, "
            "protocol_count from fct_protocol_revenue_daily",
            ["symbol", "metric_date", "total_fees_usd", "net_revenue_usd", "revenue_regime", "protocol_count"],
        )
        tvl = maybe(
            "select symbol, metric_date, tvl_usd from fct_protocol_tvl_daily",
            ["symbol", "metric_date", "tvl_usd"],
        )
        assets = maybe(
            "select symbol, sector, total_supply, max_supply, supply_as_of, genesis_date from dim_asset",
            ["symbol", "sector", "total_supply", "max_supply", "supply_as_of", "genesis_date"],
        )
        return _widen_panel(panel, revenue, tvl, assets)
    finally:
        con.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v && .venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 5: Confirm the widened panel did not move any existing estimate**

This is the step that protects Plan 2's verified output. Re-run the full estimate and compare against
the stored run: the widening must change **no** fair value of any pre-existing method.

```bash
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
con.execute("create or replace table _before as select method_id, symbol, metric_date, fair_value, applicable from fct_valuation_estimate")
PY
.venv/bin/python -m cryptoval.cli estimate && make dbt
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
print(con.execute("""
  select b.method_id, count(*) as changed
  from _before b join fct_valuation_estimate a
    on a.method_id=b.method_id and a.symbol=b.symbol and a.metric_date=b.metric_date
  where coalesce(a.fair_value,-1) <> coalesce(b.fair_value,-1) or a.applicable <> b.applicable
  group by 1 order by 2 desc
""").fetchdf().to_string())
PY
```

Expected: zero changed rows for all 14 of Plan 2's methods and for Tasks 1-3's nine. A non-zero count
means a join changed the grain — diagnose it before Task 8, because every later task builds on this
frame.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/cli.py tests/test_cli.py
git commit -m "feat: widen the estimate panel with revenue, TVL and supply metadata

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

# Stage 3a-3 — The data-dependent methods

## Task 8: Family D remainder — P/F, P/S, FDV/revenue, dFDV/revenue, P/TVL

Spec §10.5. Five multiples, each on the widened panel. Every one of them is a **diagnostic, not a
fair value** in the sense Family E's cost floors are: a multiple is a price relative to a
fundamental, and turning it into a price requires a target multiple, which is an assumption. So each
method takes an explicit `target_multiple` parameter, names its denominator in `inputs_used`, and
states on its card that the target is a choice.

Review Focus item 3 is this task's: `FdvToRevenue` and `DfdvToRevenue` require `max_supply`, which is
null for ETH, and must refuse rather than substitute `total_supply`.

**Files:**
- Modify: `cryptoval/methods/multiples.py`
- Create: `config/methods/price_to_fees.yml`, `config/methods/price_to_sales.yml`,
  `config/methods/fdv_to_revenue.yml`, `config/methods/dfdv_to_revenue.yml`,
  `config/methods/price_to_tvl.yml`
- Modify: `tests/test_methods_multiples.py`

**Interfaces:**
- Consumes: the widened panel's `total_fees_usd`, `net_revenue_usd`, `tvl_usd`, `max_supply`,
  `total_supply`, `supply_as_of`; `_finite`, `_refuse`, `register`, `param_set_hash`.
- Produces: `_multiple_estimates(frame, spec, params, denominator_column, denominator_label)` — the
  shared loop, since all five differ only in which column is the denominator and how the numerator
  is formed; classes `PriceToFees`, `PriceToSales`, `FdvToRevenue`, `DfdvToRevenue`, `PriceToTvl`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_methods_multiples.py -- append
import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.multiples import (
    DfdvToRevenue, FdvToRevenue, PriceToFees, PriceToSales, PriceToTvl,
)


def _wide_frame(days: int = 500, max_supply: float | None = 2.1e7) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")
    price = 100.0 + np.arange(days) * 0.1
    supply = np.full(days, 1.0e7)
    return pd.DataFrame(
        {
            "symbol": "AAVE", "metric_date": [d.date() for d in dates],
            "price_usd": price, "supply": supply, "market_cap_usd": price * supply,
            "total_fees_usd": np.full(days, 5.0e5), "net_revenue_usd": np.full(days, 2.0e5),
            "tvl_usd": np.full(days, 1.0e10), "total_supply": np.full(days, 1.6e7),
            "max_supply": np.full(days, max_supply if max_supply is not None else np.nan),
            "supply_as_of": [pd.Timestamp("2026-10-13").date()] * days,
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_price_to_fees_annualises_the_trailing_window_and_names_its_denominator():
    last = [e for e in PriceToFees().estimate(_wide_frame(), dict(PriceToFees.spec.params))
            if e.applicable][-1]
    # 365 rows x 5.0e5 per day of fees, against a target multiple.
    assert last.inputs_used["denominator"] == "total_fees_usd"
    assert last.inputs_used["annual_value_usd"] == pytest.approx(365 * 5.0e5)
    assert last.diagnostics["realised_multiple"] > 0


def test_price_to_sales_uses_net_revenue_not_gross_fees():
    last = [e for e in PriceToSales().estimate(_wide_frame(), dict(PriceToSales.spec.params))
            if e.applicable][-1]
    assert last.inputs_used["denominator"] == "net_revenue_usd"
    assert last.inputs_used["annual_value_usd"] == pytest.approx(365 * 2.0e5)


def test_fdv_to_revenue_refuses_when_max_supply_is_null():
    estimates = FdvToRevenue().estimate(_wide_frame(max_supply=None), dict(FdvToRevenue.spec.params))
    assert all(not e.applicable for e in estimates)
    assert "max_supply" in estimates[-1].reason
    # And it must NOT have fallen back to total_supply.
    assert all(e.fair_value is None for e in estimates)


def test_dfdv_haircuts_only_the_locked_portion():
    frame = _wide_frame()
    params = {**DfdvToRevenue.spec.params, "locked_haircut": 0.5, "window_rows": 365,
              "target_multiple": 20.0}
    last = [e for e in DfdvToRevenue().estimate(frame, params) if e.applicable][-1]
    locked = 1.6e7 - 1.0e7  # total minus circulating
    assert last.inputs_used["locked_supply"] == pytest.approx(locked)
    assert last.inputs_used["effective_supply"] == pytest.approx(1.0e7 + 0.5 * locked)


def test_price_to_tvl_refuses_an_asset_with_no_tvl():
    frame = _wide_frame().assign(tvl_usd=np.nan)
    estimates = PriceToTvl().estimate(frame, dict(PriceToTvl.spec.params))
    assert all(not e.applicable for e in estimates)
    assert "tvl" in estimates[-1].reason.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_multiples.py -k "fees or sales or fdv or tvl" -v`
Expected: FAIL — `ImportError: cannot import name 'PriceToFees'`

- [ ] **Step 3: Write the configs**

All five share this shape; `requires`, the denominator and the citation differ.

```yaml
# config/methods/price_to_fees.yml
method_id: price_to_fees
family: D_multiples
requires: [price_usd, supply, total_fees_usd]
min_density: 0.5
min_rows: 365
params: {window_rows: 365, target_multiple: 20.0, denominator: total_fees_usd}
citation: "P/F, spec 10.5; denominator discipline per spec 10.4"
failure_modes:
- 'GROSS FEES, not revenue: most of this number is routed to liquidity providers and never reaches a
  token holder. P/F is systematically the most flattering of the three denominators, which is why
  the row names its denominator.'
- 'target_multiple 20.0 is a CHOICE, not a measurement: the estimate scales linearly in it, and no
  cross-asset work has yet established what multiple this sector supports.'
- An annualised trailing window assumes the last year of fees repeats; for a protocol in a fee
  regime change it does not.
- The protocol rollup sums a symbol's configured protocols, so a symbol whose slug list is
  incomplete understates its own fees.
assumptions:
- Trailing-window fees annualise, and 20x fees is a defensible multiple.

# config/methods/price_to_sales.yml -- as above, with:
method_id: price_to_sales
requires: [price_usd, supply, net_revenue_usd]
params: {window_rows: 365, target_multiple: 15.0, denominator: net_revenue_usd}
citation: "P/S on NET protocol revenue, spec 10.4-10.5"
# failure modes: the retained share is what this measures, so a protocol in
# fees_fully_distributed regime has a near-zero denominator and the multiple explodes --
# check revenue_regime before reading it.

# config/methods/fdv_to_revenue.yml
method_id: fdv_to_revenue
requires: [price_usd, max_supply, net_revenue_usd]
params: {window_rows: 365, target_multiple: 15.0, denominator: net_revenue_usd}
citation: "FDV/revenue, spec 10.5; FDV assumes full dilution at today's price"
# failure modes: FDV assumes the entire maximum supply exists today at today's price,
# which is the critique spec 10.4 raises against it; an asset with no maximum supply is
# refused rather than given its total supply.

# config/methods/dfdv_to_revenue.yml
method_id: dfdv_to_revenue
requires: [price_usd, supply, total_supply, net_revenue_usd]
params: {window_rows: 365, target_multiple: 15.0, locked_haircut: 0.5, denominator: net_revenue_usd}
citation: "dFDV, spec 10.4 - a haircut on locked supply rather than naive full dilution"
# failure modes: locked_haircut 0.5 is an ASSUMPTION standing in for an unlock schedule the
# free tier does not serve, so dFDV here is a sensitivity parameter, not a vesting model;
# total_supply is a point-in-time snapshot (supply_as_of), not a series.

# config/methods/price_to_tvl.yml
method_id: price_to_tvl
requires: [price_usd, supply, tvl_usd]
params: {window_rows: 30, target_multiple: 0.3, denominator: tvl_usd, annualise: false}
citation: "P/TVL, spec 10.5"
# failure modes: TVL is not revenue and not a claim on assets -- it is deposits, which can
# leave in a day; it is also reflexively inflated by the price of the very token being
# valued when that token is itself collateral.
```

- [ ] **Step 4: Write the implementation**

```python
# cryptoval/methods/multiples.py -- append

def _annualised(series: pd.Series, window: int, annualise: bool = True) -> pd.Series:
    """Trailing-window mean, scaled to a year for a flow and left alone for a stock.

    A mean, not a sum: the window may contain fewer non-null days than rows, and summing
    nulls-as-zero would understate a protocol whose feed had an outage.

    ``annualise`` is False for TVL, which is a stock: multiplying a deposit balance by 365
    would produce a denominator with no meaning and a target multiple nobody could
    reconstruct.
    """
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=max(1, window // 2)).mean()
    return mean * 365.0 if annualise else mean


def _multiple_estimates(
    frame: pd.DataFrame,
    spec: MethodSpec,
    params: dict,
    numerator_supply: Callable[[Any], tuple[float | None, dict[str, Any], str]],
) -> list[Estimate]:
    """One loop for every multiple: annualise the denominator, divide by the supply the
    numerator implies, scale by the target multiple.

    ``numerator_supply`` returns (supply, extra inputs, refusal reason) for a row, which is
    the only thing the five methods disagree about -- circulating supply for P/F and P/S,
    maximum supply for FDV, circulating plus a haircut on locked for dFDV.
    """
    column = params["denominator"]
    window = int(params.get("window_rows", 365))
    target = float(params.get("target_multiple", 15.0))
    phash = param_set_hash(params)
    annual = _annualised(frame[column], window, bool(params.get("annualise", True)))
    price = pd.to_numeric(frame["price_usd"], errors="coerce")

    out: list[Estimate] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        value = _finite(annual.iloc[position])
        supply, extra, refusal = numerator_supply(row)
        if refusal:
            out.append(_refuse(spec, row, params, refusal))
            continue
        if value is None or value <= 0:
            out.append(_refuse(spec, row, params, f"no positive annualised {column} over {window} rows"))
            continue
        if supply is None or supply <= 0:
            out.append(_refuse(spec, row, params, "no usable supply on this day"))
            continue
        today_price = _finite(price.iloc[position])
        out.append(
            Estimate(
                symbol=row.symbol, metric_date=row.metric_date, method_id=spec.method_id,
                param_set_hash=phash, fair_value=value * target / supply,
                band_low=None, band_high=None,
                inputs_used={"denominator": column, "annual_value_usd": value,
                             "target_multiple": target, "window_rows": window, **extra},
                diagnostics={
                    "realised_multiple": (today_price * supply / value) if today_price else None,
                    "status": "diagnostic_target_multiple_is_an_assumption",
                },
                applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
            )
        )
    return out


def _circulating(row) -> tuple[float | None, dict, str]:
    supply = _finite(row.supply)
    return supply, {"supply": supply, "supply_basis": "circulating"}, ""


def _maximum(row) -> tuple[float | None, dict, str]:
    supply = _finite(getattr(row, "max_supply", None))
    if supply is None or supply <= 0:
        # No fallback to total_supply: that would publish a market-cap multiple as an FDV one.
        return None, {}, "no max_supply for this asset, so a fully diluted valuation does not exist"
    return supply, {"supply": supply, "supply_basis": "max_supply",
                    "supply_as_of": getattr(row, "supply_as_of", None)}, ""


def _haircut_locked(haircut: float) -> Callable[[Any], tuple[float | None, dict, str]]:
    def inner(row) -> tuple[float | None, dict, str]:
        circulating, total = _finite(row.supply), _finite(getattr(row, "total_supply", None))
        if circulating is None or total is None or total < circulating:
            return None, {}, "no total_supply at or above circulating supply for this asset"
        locked = total - circulating
        return (
            circulating + haircut * locked,
            {"supply_basis": "circulating_plus_haircut_locked", "circulating_supply": circulating,
             "total_supply": total, "locked_supply": locked, "locked_haircut": haircut,
             "effective_supply": circulating + haircut * locked,
             "supply_as_of": getattr(row, "supply_as_of", None)},
            "",
        )
    return inner


@register
class PriceToFees:
    spec = _SPECS["price_to_fees"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _multiple_estimates(frame, self.spec, params, _circulating)


@register
class PriceToSales:
    spec = _SPECS["price_to_sales"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _multiple_estimates(frame, self.spec, params, _circulating)


@register
class FdvToRevenue:
    spec = _SPECS["fdv_to_revenue"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _multiple_estimates(frame, self.spec, params, _maximum)


@register
class DfdvToRevenue:
    spec = _SPECS["dfdv_to_revenue"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _multiple_estimates(
            frame, self.spec, params, _haircut_locked(float(params.get("locked_haircut", 0.5)))
        )


@register
class PriceToTvl:
    spec = _SPECS["price_to_tvl"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _multiple_estimates(frame, self.spec, params, _circulating)
```

**TVL is a stock, not a flow, so `PriceToTvl` must not annualise it.** Give `_annualised` an
`annualise: bool` argument read from `params` (default `true`), and set `annualise: false` in
`price_to_tvl.yml`, so P/TVL's denominator is a trailing *mean* TVL in dollars while the four flow
multiples keep their x365. Do not instead absorb the 365 into `target_multiple`: that produces a
target whose units nobody can reconstruct from the row. Add this assertion to the test:

```python
def test_price_to_tvl_does_not_annualise_a_stock():
    last = [e for e in PriceToTvl().estimate(_wide_frame(), dict(PriceToTvl.spec.params))
            if e.applicable][-1]
    assert last.inputs_used["annual_value_usd"] == pytest.approx(1.0e10), "TVL is a stock"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_methods_multiples.py -v && .venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 6: Run on live data and record which assets each multiple can answer**

```bash
.venv/bin/python -m cryptoval.cli estimate --methods price_to_fees,price_to_sales,fdv_to_revenue,dfdv_to_revenue,price_to_tvl
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
print(con.execute("""
  select method_id,
         sum(case when applicable then 1 else 0 end) as applicable_rows,
         count(distinct case when applicable then symbol end) as symbols,
         any_value(reason) filter (where not applicable) as a_refusal
  from fct_valuation_estimate
  where method_id in ('price_to_fees','price_to_sales','fdv_to_revenue','dfdv_to_revenue','price_to_tvl')
  group by 1 order by 1
""").fetchdf().to_string())
PY
```

Record per method: applicable rows, distinct symbols, and one example refusal. Expected from the
live data: only the nine symbols with DefiLlama coverage can answer the revenue multiples, ETH is
refused by both FDV methods for want of a maximum supply, and `price_to_tvl` answers only if Task 6
was implemented.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/methods/multiples.py config/methods/price_to_fees.yml \
        config/methods/price_to_sales.yml config/methods/fdv_to_revenue.yml \
        config/methods/dfdv_to_revenue.yml config/methods/price_to_tvl.yml \
        tests/test_methods_multiples.py
git commit -m "feat: Family D remainder - P/F, P/S, FDV and dFDV over revenue, P/TVL

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: Sector-cohorted z-scores — the one shape the run loop cannot express

Spec §10.5 ends with "sector-cohorted z-scores so comparables are drawn within a cohort rather than
across incomparable projects". Every method so far is a function of one asset's own history.
`run_methods` encodes that: it `groupby("symbol")` and hands each method one symbol's frame
(`registry.py:142`). A cross-sectional z-score needs **every symbol on one date**, which that loop
cannot provide.

The smallest change that makes it expressible: a `cross_sectional: true` flag in the method config,
and one branch in `run_methods` that hands such a method the whole panel grouped by date instead.
Nothing else about the contract changes — same `estimate()` signature, same `Estimate` rows, same
applicability resolution, just a different grouping.

Review Focus item 4 is this task's: `dim_asset.sector` has five single-member sectors and seven
symbols with a null sector, and a z-score over a cohort of one is `inf` or `NaN`, not a score.

**Files:**
- Create: `cryptoval/methods/cohort.py`
- Modify: `cryptoval/methods/base.py` (one field on `MethodSpec`),
  `cryptoval/methods/registry.py` (the dispatch branch and the spec loader)
- Create: `config/methods/sector_cohort_zscore.yml`
- Create: `tests/test_methods_cohort.py`

**Interfaces:**
- Consumes: `Estimate`, `MethodSpec`, `resolve_applicability`, `covered_span`.
- Produces: `MethodSpec.cross_sectional: bool = False`; `run_cross_sectional(panel, method)` →
  `list[Estimate]`; class `CohortZScore`. `run_methods` dispatches on the flag.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_methods_cohort.py
import pandas as pd
import pytest

from cryptoval.methods.cohort import CohortZScore
from cryptoval.methods.registry import run_methods


def _panel() -> pd.DataFrame:
    # Three DeFi assets on one date (a usable cohort) plus one lone oracle asset.
    return pd.DataFrame(
        {
            "symbol": ["AAVE", "UNI", "CRV", "LINK"],
            "metric_date": [pd.Timestamp("2026-10-12").date()] * 4,
            "sector": ["defi", "defi", "defi", "oracle"],
            "price_usd": [150.0, 8.0, 0.5, 20.0],
            "supply": [1.0e7, 6.0e8, 1.0e9, 6.0e8],
            "market_cap_usd": [1.5e9, 4.8e9, 5.0e8, 1.2e10],
            "net_revenue_usd": [2.0e5, 3.0e5, 1.0e5, 5.0e4],
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_cohort_zscore_scores_within_the_sector():
    estimates = CohortZScore().estimate(_panel(), {"min_cohort_size": 3, "ratio": "market_cap_to_revenue"})
    by_symbol = {e.symbol: e for e in estimates}
    defi = [by_symbol[s] for s in ("AAVE", "UNI", "CRV")]
    assert all(e.applicable for e in defi)
    assert all(e.inputs_used["cohort"] == "defi" for e in defi)
    assert all(e.inputs_used["cohort_size"] == 3 for e in defi)
    # A z-score set has mean zero by construction: that is the check that it is a z-score.
    assert sum(e.diagnostics["z_score"] for e in defi) == pytest.approx(0.0, abs=1e-9)
    assert all(e.fair_value is None for e in defi), "a z-score is not a price"


def test_cohort_of_one_is_refused_with_its_measured_size():
    estimates = CohortZScore().estimate(_panel(), {"min_cohort_size": 3, "ratio": "market_cap_to_revenue"})
    link = next(e for e in estimates if e.symbol == "LINK")
    assert not link.applicable
    assert "1" in link.reason and "3" in link.reason


def test_null_sector_is_its_own_refusal_not_a_cohort():
    panel = _panel().assign(sector=[None] * 4)
    estimates = CohortZScore().estimate(panel, {"min_cohort_size": 3, "ratio": "market_cap_to_revenue"})
    assert all(not e.applicable for e in estimates)
    assert "sector" in estimates[0].reason


def test_run_methods_dispatches_a_cross_sectional_method_on_the_whole_panel():
    rows = run_methods(_panel(), {"sector_cohort_zscore": CohortZScore()}, ["sector_cohort_zscore"])
    # Four symbols in, four rows out -- and NOT four refusals from a per-symbol loop that
    # handed each method a one-row cohort.
    assert len(rows) == 4
    assert int(rows["applicable"].sum()) == 3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_methods_cohort.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.methods.cohort'`

- [ ] **Step 3: Write the config and the spec field**

```yaml
# config/methods/sector_cohort_zscore.yml
method_id: sector_cohort_zscore
family: D_multiples
requires: [market_cap_usd, net_revenue_usd, sector]
min_density: 0.3
min_rows: 1
cross_sectional: true
returns_fair_value: false
params: {min_cohort_size: 3, ratio: market_cap_to_revenue}
citation: "Sector-cohorted relative valuation (spec 10.5)"
failure_modes:
- 'The cohort is dim_asset.sector, which is a hand-assigned label: a wrong label puts an asset in the
  wrong comparable set, and the z-score then measures the labelling rather than the asset.'
- 'Five configured sectors have exactly one member and seven symbols have no sector at all, so this
  method refuses most of the universe by design. The refusal names the measured cohort size.'
- A z-score over three members is arithmetic, not statistics; min_cohort_size 3 is the floor at which
  the number exists at all, not the size at which it is trustworthy.
- 'Cross-sectional, so it is the only method here whose answer changes when an unrelated asset''s
  data changes: a revision to one cohort member moves every member''s score.'
assumptions:
- Assets within a configured sector are economically comparable.
- Market cap over annualised net revenue is the ratio worth cohorting on.
```

```python
# cryptoval/methods/base.py -- one field on MethodSpec
    cross_sectional: bool = False
```

```python
# cryptoval/methods/registry.py -- load it, and branch on it
        spec = MethodSpec(
            ...,
            cross_sectional=bool(raw.get("cross_sectional", False)),
        )
```

- [ ] **Step 4: Write the implementation**

```python
# cryptoval/methods/cohort.py
"""Cross-sectional, sector-cohorted scores (spec 10.5).

Every other method in this repo is a function of one asset's own history, which is what
run_methods' groupby("symbol") encodes. A cohort z-score is not: it needs every symbol on
one date, and its answer for one asset moves when another asset's data is revised.

That is the only thing special about it. Same estimate() signature, same Estimate rows,
same applicability resolution -- the registry just groups by date instead of by symbol.
"""

from __future__ import annotations

import pandas as pd

from cryptoval.methods.base import Estimate
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()

RATIOS = {
    # numerator / annualised denominator, both columns of the widened panel.
    "market_cap_to_revenue": ("market_cap_usd", "net_revenue_usd"),
    "market_cap_to_fees": ("market_cap_usd", "total_fees_usd"),
}


@register
class CohortZScore:
    spec = _SPECS["sector_cohort_zscore"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        minimum = int(params.get("min_cohort_size", 3))
        numerator_col, denominator_col = RATIOS[params.get("ratio", "market_cap_to_revenue")]
        phash = param_set_hash(params)

        numerator = pd.to_numeric(frame[numerator_col], errors="coerce")
        denominator = pd.to_numeric(frame[denominator_col], errors="coerce")
        ratio = numerator / denominator.where(denominator > 0)
        work = frame.assign(_ratio=ratio)

        out: list[Estimate] = []
        for row in work.itertuples(index=False):
            sector = getattr(row, "sector", None)
            if sector is None or pd.isna(sector):
                out.append(self._refuse(row, phash, "no configured sector, so there is no cohort to score within"))
                continue
            cohort = work[(work["sector"] == sector) & work["_ratio"].notna()]
            size = int(len(cohort))
            if size < minimum:
                out.append(self._refuse(row, phash,
                    f"cohort {sector!r} has {size} member(s) with a usable ratio, below the minimum of {minimum}"))
                continue
            # Population standard deviation (ddof=0): the cohort IS the population here,
            # not a sample drawn from a larger one.
            sigma = float(cohort["_ratio"].std(ddof=0))
            mine = getattr(row, "_ratio")
            if sigma <= 0 or pd.isna(mine):
                out.append(self._refuse(row, phash,
                    f"cohort {sector!r} has no dispersion in its ratio, so a z-score is undefined"))
                continue
            mean = float(cohort["_ratio"].mean())
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash, fair_value=None, band_low=None, band_high=None,
                    inputs_used={"cohort": sector, "cohort_size": size, "ratio": float(mine),
                                 "cohort_mean": mean, "cohort_stdev": sigma,
                                 "numerator": numerator_col, "denominator": denominator_col},
                    diagnostics={"z_score": (float(mine) - mean) / sigma, "status": "relative_score_not_a_price"},
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out

    def _refuse(self, row, phash: str, reason: str) -> Estimate:
        return Estimate(
            symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
            param_set_hash=phash, fair_value=None, band_low=None, band_high=None,
            inputs_used={}, diagnostics={}, applicable=False, reason=reason,
            panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
        )
```

```python
# cryptoval/methods/registry.py -- in run_methods, before the per-symbol loop

        if spec.cross_sectional:
            # One date at a time, every symbol: the cohort is the cross-section. Grouping by
            # symbol here would hand the method a cohort of one and refuse the whole universe.
            for _, slice_ in panel.groupby("metric_date", sort=True):
                try:
                    estimates = method.estimate(slice_.reset_index(drop=True), dict(spec.params))
                except Exception as exc:
                    rows.append(_refusal("*", spec, f"method raised {type(exc).__name__}: {exc}", slice_))
                    continue
                rows.extend(_row(e, circular) for e in estimates)
            continue
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_methods_cohort.py tests/test_methods_registry.py -v && .venv/bin/python -m pytest -q`
Expected: PASS, whole suite green. `tests/test_methods_registry.py` must stay green untouched — the
per-symbol path is unchanged for all 23 other methods.

- [ ] **Step 6: Measure how much of the universe is actually scoreable**

```bash
.venv/bin/python -m cryptoval.cli estimate --methods sector_cohort_zscore
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
print(con.execute("""
  select json_extract_string(inputs_used,'$.cohort') as cohort,
         count(distinct symbol) as symbols, sum(case when applicable then 1 else 0 end) as rows
  from fct_valuation_estimate where method_id='sector_cohort_zscore'
  group by 1 order by 3 desc
""").fetchdf().to_string())
PY
```

Record which cohorts cleared `min_cohort_size` and which did not. Expected: `defi` clears it, every
other sector does not, and that is the finding — spec §10.5's cohorting is only available for one
sector on this universe, and the report must say so rather than implying a cross-sector comparison
happened.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/methods/cohort.py cryptoval/methods/base.py cryptoval/methods/registry.py \
        config/methods/sector_cohort_zscore.yml tests/test_methods_cohort.py
git commit -m "feat: sector-cohorted z-scores, with a cross-sectional path in the registry

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 10: Family C — the DCF chapter and the counter-model that disputes it

Spec §10.4, the one family with no counterpart in the dissertation. Six methods, and the sixth
exists to disagree with the other five: the standing critique is that staking yield and fee burn
change the *supply path*, not network value, so DCF arithmetic does not transfer to an L1 token.
Spec §10.4 is explicit that this disagreement is **implemented as a sibling method so it is
measured rather than argued**, and that where the two diverge is itself the finding.

**The sensitivity grid is a parameter-set grid, not a new return shape.** Spec §10.4 wants "a
sensitivity grid over (growth, discount rate, terminal multiple), not a single number", and the
registry already has the mechanism: `param_set_hash` exists so "a DCF at 12% and at 16% coexist as
rows" (`registry.py:66`). So a grid cell is a param set, and the method emits one `Estimate` per
cell per date, each carrying its own hash. No new table, no nested structure, no change to
`fct_valuation_estimate`.

**Before writing the ETH DCF, verify what ETH's `net_revenue_usd` actually is.** DefiLlama's
`dailyRevenue` for a *chain* is not self-evidently the burned fee; the plan must not assume it. Step
1 of this task is a measurement, not code.

**Files:**
- Create: `cryptoval/methods/cashflow.py`
- Create: `config/methods/eth_three_stage_dcf.yml`, `config/methods/defi_net_revenue_dcf.yml`,
  `config/methods/bond_target_yield.yml`, `config/methods/emissions_adjusted_real_yield.yml`,
  `config/methods/supply_effect_counter_model.yml`
- Create: `tests/test_methods_cashflow.py`

**Interfaces:**
- Consumes: the widened panel's `total_fees_usd`, `net_revenue_usd`, `revenue_regime`,
  `issuance_usd`, `supply`, `price_usd`, `market_cap_usd`.
- Produces: `three_stage_pv(cash_flow, growth, discount, years, terminal_multiple) -> tuple[float, float]`
  returning `(present_value, terminal_share)`; `grid(params) -> list[dict]` expanding the
  `sensitivity` block into param sets; classes `EthThreeStageDcf`, `DefiNetRevenueDcf`,
  `BondTargetYield`, `EmissionsAdjustedRealYield`, `SupplyEffectCounterModel`.

- [ ] **Step 1: Measure what ETH's revenue column contains, before writing any method**

```bash
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
print(con.execute("""
  select metric_date, total_fees_usd, net_revenue_usd,
         round(net_revenue_usd / nullif(total_fees_usd,0), 3) as revenue_share_of_fees
  from fct_protocol_revenue_daily
  where symbol='ETH' and metric_date > date '2026-09-01' order by metric_date desc limit 10
""").fetchdf().to_string())
PY
```

Compare the `net_revenue_usd` figures against ETH's published daily burn for the same dates (any
public burn tracker; this is a one-off manual cross-check, not a dependency). Record in the task
report which of these the column is:

- **the burned fee** — then the ETH DCF's buyback term is `net_revenue_usd` directly;
- **the validator/proposer take** — then the buyback term is `total_fees_usd - net_revenue_usd`;
- **neither, or not decidable from the data** — then `eth_three_stage_dcf` ships with
  `requires: [total_fees_usd, issuance_usd, supply, price_usd]` and treats **total fees** as the
  cash-flow proxy, with the card stating plainly that the burn/validator split is not separable on
  free data.

Write the decision into `eth_three_stage_dcf.yml`'s `assumptions` before Step 3. Do not guess, and
do not let the method's name imply a decomposition the data cannot support.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_methods_cashflow.py
import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.cashflow import (
    BondTargetYield, DefiNetRevenueDcf, EmissionsAdjustedRealYield, EthThreeStageDcf,
    SupplyEffectCounterModel, grid, three_stage_pv,
)


def _cash_frame(days: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2025-06-01", periods=days, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": "ETH", "metric_date": [d.date() for d in dates],
            "price_usd": np.full(days, 2500.0), "supply": np.linspace(1.2e8, 1.21e8, days),
            "market_cap_usd": np.full(days, 3.0e11),
            "total_fees_usd": np.full(days, 3.0e6), "net_revenue_usd": np.full(days, 1.0e6),
            "issuance_usd": np.full(days, 2.0e6), "revenue_regime": "retains_revenue",
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_three_stage_pv_is_a_discounted_sum_and_reports_its_terminal_share():
    pv, terminal_share = three_stage_pv(
        cash_flow=100.0, growth=(0.2, 0.1, 0.03), discount=0.15, years=(3, 3, 4), terminal_multiple=10.0
    )
    assert pv > 0
    assert 0.0 <= terminal_share <= 1.0
    # A zero discount rate makes every future dollar worth a dollar today: the PV must rise.
    pv_cheap, _ = three_stage_pv(100.0, (0.2, 0.1, 0.03), 0.05, (3, 3, 4), 10.0)
    assert pv_cheap > pv


def test_three_stage_pv_refuses_a_discount_rate_below_terminal_growth():
    # The Gordon terminal term is undefined when g >= r: it must raise, not return a
    # negative or an infinity that would travel into a fair value.
    with pytest.raises(ValueError, match="terminal growth"):
        three_stage_pv(100.0, (0.2, 0.1, 0.12), 0.10, (3, 3, 4), None)


def test_grid_expands_a_sensitivity_block_into_param_sets():
    cells = grid({"discount_rate": 0.15, "sensitivity": {"discount_rate": [0.10, 0.20],
                                                         "terminal_multiple": [10.0, 20.0]}})
    assert len(cells) == 4
    assert {c["discount_rate"] for c in cells} == {0.10, 0.20}
    assert all("sensitivity" not in c for c in cells), "a cell must not carry the grid that made it"


def test_eth_dcf_emits_one_row_per_grid_cell_with_distinct_hashes():
    frame = _cash_frame()
    estimates = EthThreeStageDcf().estimate(frame, dict(EthThreeStageDcf.spec.params))
    latest = [e for e in estimates if e.metric_date == frame["metric_date"].iloc[-1] and e.applicable]
    cells = len(grid(dict(EthThreeStageDcf.spec.params)))
    assert len(latest) == cells
    assert len({e.param_set_hash for e in latest}) == cells
    assert all("terminal_share" in e.diagnostics for e in latest)


def test_eth_dcf_flags_a_cell_whose_value_is_mostly_terminal():
    frame = _cash_frame()
    params = {**EthThreeStageDcf.spec.params, "discount_rate": 0.06, "terminal_multiple": 60.0,
              "sensitivity": {}, "max_terminal_share": 0.8}
    last = [e for e in EthThreeStageDcf().estimate(frame, params) if e.applicable][-1]
    assert last.diagnostics["terminal_share"] > 0.8
    assert last.diagnostics["terminal_dominated"] is True
    # Flagged, not refused: the number is arithmetically right and interpretively weak.
    assert last.fair_value is not None


def test_defi_dcf_refuses_a_protocol_that_retains_no_revenue():
    frame = _cash_frame().assign(revenue_regime="fees_fully_distributed", net_revenue_usd=0.0)
    estimates = DefiNetRevenueDcf().estimate(frame, dict(DefiNetRevenueDcf.spec.params))
    assert all(not e.applicable for e in estimates)
    assert "retain" in estimates[-1].reason


def test_bond_target_yield_divides_annual_fees_by_an_explicit_target():
    params = {"window_rows": 365, "risk_free_rate": 0.04, "staking_risk_premium": 0.06,
              "denominator": "total_fees_usd"}
    last = [e for e in BondTargetYield().estimate(_cash_frame(), params) if e.applicable][-1]
    assert last.inputs_used["target_yield"] == pytest.approx(0.10)
    annual = 365 * 3.0e6
    assert last.fair_value == pytest.approx(annual / 0.10 / 1.21e8, rel=1e-3)


def test_real_yield_subtracts_dilution_and_says_the_incentive_term_is_missing():
    last = [e for e in EmissionsAdjustedRealYield().estimate(
        _cash_frame(), dict(EmissionsAdjustedRealYield.spec.params)) if e.applicable][-1]
    assert last.fair_value is None, "a yield is not a price"
    assert last.diagnostics["incentive_share_available"] is False
    assert last.diagnostics["real_yield"] < last.diagnostics["fee_yield"]


def test_counter_model_holds_network_value_constant_and_says_so():
    last = [e for e in SupplyEffectCounterModel().estimate(
        _cash_frame(), dict(SupplyEffectCounterModel.spec.params)) if e.applicable][-1]
    assert last.diagnostics["holds_network_value_constant"] is True
    assert last.inputs_used["projected_supply"] > 0
    # Supply is growing in the fixture, so the counter-model's value must sit BELOW price.
    assert last.fair_value < 2500.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_cashflow.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.methods.cashflow'`

- [ ] **Step 4: Write the configs**

```yaml
# config/methods/eth_three_stage_dcf.yml
method_id: eth_three_stage_dcf
family: C_cashflow
# Set from Step 1's measurement. If the burn/validator split proved inseparable, this is
# [total_fees_usd, issuance_usd, supply, price_usd] and the card says why.
requires: [net_revenue_usd, total_fees_usd, issuance_usd, supply, price_usd]
min_density: 0.5
min_rows: 365
params:
  window_rows: 365
  stage_years: [3, 3, 4]
  growth: [0.20, 0.10, 0.03]
  discount_rate: 0.15
  terminal_multiple: 15.0
  max_terminal_share: 0.8
  # The grid spec 10.4 requires. Each combination becomes its own param set and its own row.
  sensitivity:
    discount_rate: [0.10, 0.15, 0.20, 0.25]
    terminal_multiple: [10.0, 15.0, 25.0]
citation: "Spec 10.4, ETH three-stage DCF; discount rate decomposed per spec 10.4"
failure_modes:
- 'The output is a GRID, not a number. A single cell quoted without its (discount_rate,
  terminal_multiple) pair is not this method''s answer, and the spread across cells is wider than
  most method-to-method disagreement in this repo.'
- 'Terminal value dominates: every row carries terminal_share, and a row above max_terminal_share is
  flagged terminal_dominated. Such a row is a statement about the terminal multiple, not about
  Ethereum.'
- 'The cash-flow proxy treats burned fees as a buyback and issuance as stock-based compensation,
  which is an analogy to equity, not an identity. The counter-model
  (supply_effect_counter_model) exists to dispute exactly this and is reported beside it.'
- A trailing-year mean of fees annualised assumes the last year represents the next ten; ETH's fee
  revenue is cyclical with onchain activity and L2 migration has moved it structurally.
assumptions:
- Burned fees accrue to holders like a buyback and issuance dilutes like compensation.
- Three stages of 3, 3 and 4 years, and a terminal multiple on the final year's cash flow.

# config/methods/defi_net_revenue_dcf.yml
method_id: defi_net_revenue_dcf
family: C_cashflow
requires: [net_revenue_usd, revenue_regime, supply, price_usd, issuance_usd]
min_density: 0.5
min_rows: 180
params:
  window_rows: 365
  stage_years: [3, 3, 4]
  growth: [0.15, 0.08, 0.02]
  discount_rate: 0.20
  terminal_multiple: 10.0
  max_terminal_share: 0.8
  sensitivity: {discount_rate: [0.15, 0.20, 0.30]}
citation: "Spec 10.4, DeFi net-revenue DCF - the retained share, never gross fees"
failure_modes:
- 'NET revenue, never gross fees: the gross figure is mostly routed to liquidity providers. A
  protocol whose revenue_regime is fees_fully_distributed is refused, because discounting a cash
  flow it does not retain would value money that never reaches a holder.'
- 'Per-token value divides by supply grown net of emissions, so an aggressive emissions schedule
  reduces the per-token answer even when the protocol''s revenue grows.'
- DefiLlama''s per-protocol revenue definition is the vendor''s, and its methodology hash changing is
  a silent redefinition of this method''s denominator - the hash is stored for that reason.
- A discount rate of 20% is a judgement about a sector with no established cost of capital.
assumptions:
- The protocol retains the revenue DefiLlama reports as revenue.
- Emissions continue at the trailing rate across the forecast.

# config/methods/bond_target_yield.yml
method_id: bond_target_yield
family: C_cashflow
requires: [total_fees_usd, supply, price_usd]
min_density: 0.5
min_rows: 180
params:
  window_rows: 365
  # Spec 10.4 requires both to be EXPLICIT parameters rather than fitted.
  risk_free_rate: 0.04
  staking_risk_premium: 0.06
  denominator: total_fees_usd
citation: "Spec 10.4, bond-style target yield = annual fees / (risk free + staking premium)"
failure_modes:
- 'A perpetuity with no growth term: value = annual fees / target yield assumes this year''s fees
  forever, which is neither a forecast nor a floor.'
- 'Both rate components are parameters set by hand. The risk-free rate is not read from a treasury
  series (the free tier here serves gold and the dollar index, not the 10-year), so it is as of
  whenever it was last edited - the value travels on every row for that reason.'
- The staking risk premium is the entire judgement of the method compressed into one number.
assumptions:
- Fee revenue is perpetual and flat, and 10% is the yield such a claim should offer.

# config/methods/emissions_adjusted_real_yield.yml
method_id: emissions_adjusted_real_yield
family: C_cashflow
requires: [total_fees_usd, issuance_usd, market_cap_usd]
min_density: 0.5
min_rows: 365
returns_fair_value: false
params: {window_rows: 365, incentive_share_usd: 0.0}
citation: "Spec 10.4, emissions-adjusted real yield"
failure_modes:
- 'INCOMPLETE BY CONSTRUCTION: spec 10.4 defines real yield as fee share + incentive share - dilution,
  and no free source serves per-asset incentive emissions. The incentive term is a parameter
  defaulting to zero, every row carries incentive_share_available=false, and the published number is
  therefore a LOWER bound on the incentive side and must not be read as the complete figure.'
- 'Returns no fair value by declaration: a yield is not a price.'
- Dilution is measured as issuance over market cap at today's price, so a price crash inflates the
  apparent dilution rate without any change in emissions.
assumptions:
- Fees accrue to holders, and issuance dilutes them at the current market cap.

# config/methods/supply_effect_counter_model.yml
method_id: supply_effect_counter_model
family: C_cashflow
requires: [market_cap_usd, supply, issuance_usd, price_usd, total_fees_usd]
min_density: 0.5
min_rows: 365
params: {window_rows: 365, horizon_years: 10, burn_share_of_fees: 1.0}
citation: "Spec 10.4's counter-model: fee burn and staking yield change the SUPPLY PATH, not value"
failure_modes:
- 'This method exists to DISAGREE with the two DCFs, and it does so by construction: it holds network
  value constant and projects only the supply path, so its answer is today''s market cap over
  tomorrow''s supply. It is a null hypothesis, not a valuation, and the gap between it and
  eth_three_stage_dcf is the finding spec 10.4 asks for.'
- 'Holding network value constant is itself a strong claim: it says nothing the protocol earns
  changes what the network is worth.'
- 'Projecting net issuance at the trailing rate across ten years ignores every scheduled change
  (a halving, an EIP, a staking-rate shift).'
assumptions:
- Value accrues only through the supply path, never through earnings.
- Trailing-year net issuance and burn continue for horizon_years.
```

- [ ] **Step 5: Write the implementation**

```python
# cryptoval/methods/cashflow.py
"""Family C -- discounted cash flow, and the counter-model that disputes it (spec 10.4).

Two things here are deliberate and should not be "tidied".

THE OUTPUT IS A GRID. Spec 10.4 asks for a sensitivity grid over growth, discount rate
and terminal multiple rather than a single number, and the registry already hashes
parameter sets so cells coexist as rows. A grid cell is therefore a param set, and a
single cell quoted alone is not this family's answer.

THE COUNTER-MODEL IS A SIBLING METHOD. The standing critique -- that fee burn and staking
yield move the supply path rather than network value -- is implemented, not argued, so the
disagreement between it and the DCFs is measured. Where they diverge is the finding.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.cost import _finite
from cryptoval.methods.monetary import _refuse
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def grid(params: dict) -> list[dict]:
    """Expand a ``sensitivity`` block into one param set per combination.

    The cell does NOT carry the sensitivity block: if it did, every cell would hash to the
    same param_set_hash and the grid would collapse to one row in the fact table.
    """
    sensitivity = {k: v for k, v in (params.get("sensitivity") or {}).items() if v}
    base = {k: v for k, v in params.items() if k != "sensitivity"}
    if not sensitivity:
        return [base]
    names = sorted(sensitivity)
    return [
        {**base, **dict(zip(names, combination, strict=True))}
        for combination in itertools.product(*(sensitivity[n] for n in names))
    ]


def three_stage_pv(
    cash_flow: float,
    growth: tuple[float, float, float],
    discount: float,
    years: tuple[int, int, int],
    terminal_multiple: float | None,
) -> tuple[float, float]:
    """Present value of three growth stages, and the fraction of it that is terminal.

    ``terminal_multiple`` None uses a Gordon terminal value at the third stage's growth,
    which REQUIRES discount > that growth. Raising rather than returning a negative or an
    infinity is deliberate: a negative present value from an inverted Gordon term is
    arithmetically valid and economically meaningless, and it would reach the fact table
    as a fair value.
    """
    if discount <= 0:
        raise ValueError(f"discount rate must be positive, got {discount}")
    if terminal_multiple is None and discount <= growth[2]:
        raise ValueError(
            f"terminal growth {growth[2]} is at or above the discount rate {discount}; "
            f"the Gordon terminal value is undefined there"
        )

    explicit = 0.0
    flow = float(cash_flow)
    period = 0
    for stage_growth, stage_years in zip(growth, years, strict=True):
        for _ in range(int(stage_years)):
            period += 1
            flow *= 1.0 + stage_growth
            explicit += flow / (1.0 + discount) ** period

    if terminal_multiple is None:
        terminal_flow = flow * (1.0 + growth[2]) / (discount - growth[2])
    else:
        terminal_flow = flow * float(terminal_multiple)
    terminal = terminal_flow / (1.0 + discount) ** period
    total = explicit + terminal
    return total, (terminal / total if total > 0 else 1.0)


def _annual(series: pd.Series, window: int) -> pd.Series:
    """Trailing-window mean annualised. Mean, not sum: see Task 8's _annualised."""
    return pd.to_numeric(series, errors="coerce").rolling(window, min_periods=max(1, window // 2)).mean() * 365.0


def _dcf_estimates(
    frame: pd.DataFrame,
    spec: MethodSpec,
    params: dict,
    cash_flow_column_or_expr,
    label: str,
    gate=None,
) -> list[Estimate]:
    """One row per (row, grid cell). ``gate`` returns a refusal reason or ""."""
    out: list[Estimate] = []
    cells = grid(params)
    window = int(params.get("window_rows", 365))
    cash = _annual(cash_flow_column_or_expr(frame), window)
    supply = pd.to_numeric(frame["supply"], errors="coerce")

    for position, row in enumerate(frame.itertuples(index=False)):
        refusal = gate(row) if gate else ""
        annual_flow = _finite(cash.iloc[position])
        today_supply = _finite(supply.iloc[position])
        for cell in cells:
            phash = param_set_hash(cell)
            if refusal:
                out.append(_refuse(spec, row, cell, refusal))
                continue
            if annual_flow is None or annual_flow <= 0:
                out.append(_refuse(spec, row, cell, f"no positive annualised {label} over {window} rows"))
                continue
            if today_supply is None or today_supply <= 0:
                out.append(_refuse(spec, row, cell, "no supply on this day"))
                continue
            try:
                present_value, terminal_share = three_stage_pv(
                    annual_flow, tuple(cell["growth"]), float(cell["discount_rate"]),
                    tuple(cell["stage_years"]), cell.get("terminal_multiple"),
                )
            except ValueError as exc:
                out.append(_refuse(spec, row, cell, str(exc)))
                continue
            ceiling = float(cell.get("max_terminal_share", 0.8))
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=spec.method_id,
                    param_set_hash=phash, fair_value=present_value / today_supply,
                    band_low=None, band_high=None,
                    inputs_used={"cash_flow_basis": label, "annual_cash_flow_usd": annual_flow,
                                 "supply": today_supply, "discount_rate": cell["discount_rate"],
                                 "terminal_multiple": cell.get("terminal_multiple"),
                                 "growth": list(cell["growth"]), "stage_years": list(cell["stage_years"])},
                    diagnostics={"terminal_share": terminal_share,
                                 "terminal_dominated": terminal_share > ceiling,
                                 "present_value_usd": present_value},
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
    return out


@register
class EthThreeStageDcf:
    spec = _SPECS["eth_three_stage_dcf"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        # Buyback + validator take - issuance, per Step 1's measurement. If that step found
        # the split inseparable, this reduces to total_fees_usd - issuance_usd and the
        # config's requires/assumptions already say so.
        def cash_flow(f: pd.DataFrame) -> pd.Series:
            fees = pd.to_numeric(f["total_fees_usd"], errors="coerce")
            issuance = pd.to_numeric(f["issuance_usd"], errors="coerce")
            return fees - issuance

        return _dcf_estimates(frame, self.spec, params, cash_flow, "fees_net_of_issuance")


@register
class DefiNetRevenueDcf:
    spec = _SPECS["defi_net_revenue_dcf"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def gate(row) -> str:
            regime = getattr(row, "revenue_regime", None)
            if regime == "fees_fully_distributed":
                return ("this protocol does not retain revenue (revenue_regime="
                        "fees_fully_distributed), so there is no cash flow to discount")
            return ""

        return _dcf_estimates(
            frame, self.spec, params,
            lambda f: pd.to_numeric(f["net_revenue_usd"], errors="coerce"),
            "net_protocol_revenue", gate,
        )


@register
class BondTargetYield:
    spec = _SPECS["bond_target_yield"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        target = float(params["risk_free_rate"]) + float(params["staking_risk_premium"])
        phash = param_set_hash(params)
        annual = _annual(frame[params.get("denominator", "total_fees_usd")], window)
        supply = pd.to_numeric(frame["supply"], errors="coerce")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            flow, today_supply = _finite(annual.iloc[position]), _finite(supply.iloc[position])
            if target <= 0:
                out.append(_refuse(self.spec, row, params, f"target yield {target} is not positive"))
                continue
            if flow is None or flow <= 0 or today_supply is None or today_supply <= 0:
                out.append(_refuse(self.spec, row, params, "no annualised fee flow or supply on this day"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash, fair_value=flow / target / today_supply,
                    band_low=None, band_high=None,
                    inputs_used={"annual_fees_usd": flow, "target_yield": target,
                                 "risk_free_rate": params["risk_free_rate"],
                                 "staking_risk_premium": params["staking_risk_premium"],
                                 "denominator": params.get("denominator", "total_fees_usd"),
                                 "supply": today_supply},
                    diagnostics={"implied_current_yield": (flow / (today_supply * _finite(row.price_usd)))
                                 if _finite(row.price_usd) else None,
                                 "status": "flat_perpetuity"},
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class EmissionsAdjustedRealYield:
    spec = _SPECS["emissions_adjusted_real_yield"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        incentive = float(params.get("incentive_share_usd", 0.0))
        phash = param_set_hash(params)
        fees = _annual(frame["total_fees_usd"], window)
        issuance = _annual(frame["issuance_usd"], window)
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            annual_fees, annual_issuance = _finite(fees.iloc[position]), _finite(issuance.iloc[position])
            cap = _finite(market_cap.iloc[position])
            if annual_fees is None or annual_issuance is None or cap is None or cap <= 0:
                out.append(_refuse(self.spec, row, params,
                                   "no annualised fees, issuance or market cap on this day"))
                continue
            fee_yield = annual_fees / cap
            dilution = annual_issuance / cap
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash, fair_value=None, band_low=None, band_high=None,
                    inputs_used={"annual_fees_usd": annual_fees, "annual_issuance_usd": annual_issuance,
                                 "market_cap_usd": cap, "incentive_share_usd": incentive},
                    diagnostics={
                        "fee_yield": fee_yield, "dilution": dilution,
                        "real_yield": fee_yield + (incentive / cap) - dilution,
                        # Load-bearing: spec 10.4's incentive term has no free source, so the
                        # published figure is a lower bound on that side and says so on every row.
                        "incentive_share_available": incentive > 0,
                        "status": "yield_not_a_price",
                    },
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class SupplyEffectCounterModel:
    spec = _SPECS["supply_effect_counter_model"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        horizon = int(params.get("horizon_years", 10))
        burn_share = float(params.get("burn_share_of_fees", 1.0))
        phash = param_set_hash(params)

        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        issuance_coins = pd.to_numeric(frame["issuance_usd"], errors="coerce") / price.where(price > 0)
        burn_coins = pd.to_numeric(frame["total_fees_usd"], errors="coerce") * burn_share / price.where(price > 0)
        net_annual = (issuance_coins - burn_coins).rolling(window, min_periods=max(1, window // 2)).mean() * 365.0
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
        supply = pd.to_numeric(frame["supply"], errors="coerce")

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            cap, today_supply = _finite(market_cap.iloc[position]), _finite(supply.iloc[position])
            net = _finite(net_annual.iloc[position])
            if cap is None or today_supply is None or today_supply <= 0 or net is None:
                out.append(_refuse(self.spec, row, params,
                                   "no market cap, supply or net issuance rate on this day"))
                continue
            projected = today_supply + net * horizon
            if projected <= 0:
                out.append(_refuse(self.spec, row, params,
                                   f"projected supply over {horizon} years is not positive"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash,
                    # Today's network value over tomorrow's supply: the counter-model's whole claim.
                    fair_value=cap / projected,
                    band_low=None, band_high=None,
                    inputs_used={"market_cap_usd": cap, "supply": today_supply,
                                 "net_annual_issuance_coins": net, "projected_supply": projected,
                                 "horizon_years": horizon, "burn_share_of_fees": burn_share},
                    diagnostics={"holds_network_value_constant": True,
                                 "supply_change_fraction": (projected / today_supply) - 1.0,
                                 "status": "null_hypothesis_against_the_dcfs"},
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_methods_cashflow.py -v && .venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 7: Run on live data and measure the disagreement spec §10.4 asks for**

```bash
.venv/bin/python -m cryptoval.cli estimate --methods eth_three_stage_dcf,defi_net_revenue_dcf,bond_target_yield,emissions_adjusted_real_yield,supply_effect_counter_model
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
latest = "(select max(metric_date) from fct_valuation_estimate)"
print("--- the ETH grid ---")
print(con.execute(f"""
  select json_extract_string(inputs_used,'$.discount_rate') as r,
         json_extract_string(inputs_used,'$.terminal_multiple') as tm,
         round(fair_value,2) as per_token,
         round(cast(json_extract_string(diagnostics,'$.terminal_share') as double),3) as terminal_share,
         json_extract_string(diagnostics,'$.terminal_dominated') as flagged
  from fct_valuation_estimate
  where method_id='eth_three_stage_dcf' and symbol='ETH' and applicable and metric_date={latest}
  order by cast(r as double), cast(tm as double)
""").fetchdf().to_string())
print("--- DCF against the counter-model ---")
print(con.execute(f"""
  select symbol, method_id, round(min(fair_value),4) as lo, round(max(fair_value),4) as hi
  from fct_valuation_estimate
  where applicable and metric_date={latest}
    and method_id in ('eth_three_stage_dcf','defi_net_revenue_dcf','bond_target_yield','supply_effect_counter_model')
  group by 1,2 order by 1,2
""").fetchdf().to_string())
PY
```

Record in the task report: the full ETH grid with its terminal shares, how many cells are
`terminal_dominated`, and for each asset the ratio of the DCF range to the counter-model's single
value. **That ratio is this task's deliverable** — spec §10.4 asks for the disagreement to be
measured, and this is the measurement.

- [ ] **Step 8: Commit**

```bash
git add cryptoval/methods/cashflow.py config/methods/eth_three_stage_dcf.yml \
        config/methods/defi_net_revenue_dcf.yml config/methods/bond_target_yield.yml \
        config/methods/emissions_adjusted_real_yield.yml \
        config/methods/supply_effect_counter_model.yml tests/test_methods_cashflow.py
git commit -m "feat: Family C - the DCF chapter and the counter-model that disputes it

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 11: Family A remainder — the Kampakis log-linear equation of exchange

**This task exists because the self-review at the end of this plan found a hole in Plan 2, not in
this one.** Spec §10.2 lists four Family A methods: crypto QTM, the **Kampakis log-linear EoE**, the
holding-time correction, and the J-curve. Plan 2's Task 13 built QTM in both readings, the J-curve
and the holding-time correction — four *methods* but only three of the four *specifications*. The
Kampakis log-linear form was never implemented, and Plan 2's spec-coverage appendix recorded §10.2 as
covered. It is covered here.

Spec §10.2 is specific: `log(price) = β₁·log(T) + β₂·log(M) + β₃·log(1/V)`, published fit
β = (0.88, 0.84, 1.15) with adjusted R² 0.97, and **we re-estimate rather than adopt the
coefficients**.

**The finding this method produces is about its own identity, and the card must lead with it.**
`M` is market cap, which is `price × supply`; `V` is `T / market_cap`. So the right-hand side
contains the left-hand side twice over, and the published adjusted R² of 0.97 is in part an
accounting identity rather than an empirical result. The registry's automatic circularity flag will
**not** catch this: `MARKET_CAP_DERIVED` (`base.py:30-38`) lists columns derived *from* market cap,
and `market_cap_usd` is deliberately not among them because Metcalfe uses it as a regressand. Here it
is a *regressor against price*, which is a different relationship the flag cannot distinguish. So
this method declares it itself, and the limitation of the automatic flag is recorded on the card.

**Files:**
- Modify: `cryptoval/methods/monetary.py`
- Create: `config/methods/kampakis_log_linear_eoe.yml`
- Modify: `tests/test_methods_monetary.py`

**Interfaces:**
- Consumes: `volume_total_usd`, `market_cap_usd`, `velocity_total`, `price_usd`, `supply` from the
  panel; `statsmodels.api` as already used by `fit_metcalfe`.
- Produces: `fit_log_linear_eoe(frame, min_rows)` → `dict | None` with keys `beta_t`, `beta_m`,
  `beta_inv_v`, each with `_ci_low`/`_ci_high`, plus `adj_r_squared`, `durbin_watson`, `n`; class
  `KampakisLogLinearEoe`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_methods_monetary.py -- append
import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.monetary import KampakisLogLinearEoe, fit_log_linear_eoe


def _eoe_frame(days: int = 600) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")
    price = 100.0 * np.exp(np.linspace(0.0, 0.8, days))
    supply = np.full(days, 1.0e8)
    market_cap = price * supply
    volume = market_cap * 0.04
    return pd.DataFrame(
        {
            "symbol": "ETH", "metric_date": [d.date() for d in dates], "price_usd": price,
            "supply": supply, "market_cap_usd": market_cap, "volume_total_usd": volume,
            "velocity_total": volume / market_cap,
            "panel_vintage_ts": pd.Timestamp("2026-10-13", tz="UTC"),
        }
    )


def test_fit_log_linear_eoe_returns_three_betas_each_with_an_interval():
    fit = fit_log_linear_eoe(_eoe_frame(), min_rows=180)
    assert fit is not None
    for name in ("beta_t", "beta_m", "beta_inv_v"):
        assert name in fit
        assert fit[f"{name}_ci_low"] <= fit[name] <= fit[f"{name}_ci_high"]
    assert "adj_r_squared" in fit and "durbin_watson" in fit


def test_kampakis_declares_its_own_circularity_because_the_registry_cannot_see_it():
    from cryptoval.methods.base import circularity

    # velocity_total is caught automatically...
    assert "velocity_total" in circularity(KampakisLogLinearEoe.spec)
    last = [e for e in KampakisLogLinearEoe().estimate(_eoe_frame(), dict(KampakisLogLinearEoe.spec.params))
            if e.applicable][-1]
    # ...but market cap as a REGRESSOR against price is not, so the method says it itself.
    assert last.diagnostics["structurally_circular"] is True
    assert "price" in last.diagnostics["circularity_note"]


def test_kampakis_reports_its_own_betas_not_the_published_ones():
    last = [e for e in KampakisLogLinearEoe().estimate(_eoe_frame(), dict(KampakisLogLinearEoe.spec.params))
            if e.applicable][-1]
    published = (0.88, 0.84, 1.15)
    estimated = (last.diagnostics["beta_t"], last.diagnostics["beta_m"], last.diagnostics["beta_inv_v"])
    assert estimated != published, "the coefficients are re-estimated, never adopted"
    assert last.diagnostics["published_beta"] == list(published)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_methods_monetary.py -k kampakis -v`
Expected: FAIL — `ImportError: cannot import name 'fit_log_linear_eoe'`

- [ ] **Step 3: Write the config**

```yaml
# config/methods/kampakis_log_linear_eoe.yml
method_id: kampakis_log_linear_eoe
family: A_monetary
# velocity_total is market-cap-derived, so the registry flags this circular automatically.
# market_cap_usd as a regressor against price is ALSO circular and the registry cannot see
# it -- the method declares that itself. See the failure modes.
requires: [price_usd, volume_total_usd, market_cap_usd, velocity_total, supply]
min_density: 0.6
min_rows: 365
params: {fit_window_rows: 365, min_fit_rows: 180, published_beta: [0.88, 0.84, 1.15]}
citation: "Kampakis et al. (2024), log-linear equation of exchange; spec 10.2"
failure_modes:
- 'CIRCULAR ON BOTH SIDES OF THE EQUATION. M is market cap, which IS price times supply, and V is
  volume over market cap. So log(price) is regressed on two transforms of log(price), and the
  published adjusted R-squared of 0.97 is in part an accounting identity rather than an empirical
  finding. This is the single most important thing to know about this method.'
- 'The registry''s automatic circularity flag does NOT catch the market-cap term: MARKET_CAP_DERIVED
  lists columns derived FROM market cap, and market_cap_usd is deliberately excluded because
  Metcalfe uses it as a regressand. Here it is a regressor against price, which the flag cannot
  distinguish, so the method sets structurally_circular itself.'
- 'The 1/V term is the inversion the holding-time correction invalidates (spec 10.2): measured on
  this panel, H = 1/V understates holding time by 16-57% per asset.'
- The equation of exchange presumes a medium of exchange; for a store-of-value asset the fit is
  weakly identified, and USDC is the control case.
- T is exchange volume, not on-chain settlement, and the two volume series differ by roughly 5x.
assumptions:
- Re-estimated coefficients, never the published (0.88, 0.84, 1.15).
- volume_total_usd stands in for T; market cap for M; velocity_total for V.
```

- [ ] **Step 4: Write the implementation**

```python
# cryptoval/methods/monetary.py -- append
import numpy as np
import statsmodels.api as sm

EOE_TERMS = ("beta_t", "beta_m", "beta_inv_v")


def fit_log_linear_eoe(frame: pd.DataFrame, min_rows: int) -> dict | None:
    """OLS of log(price) on log(T), log(M) and log(1/V) -- spec 10.2's Kampakis form.

    None below min_rows or when too few rows have all four quantities strictly positive.
    The caller turns that into a refusal naming the measurement.
    """
    work = pd.DataFrame(
        {
            "price": pd.to_numeric(frame["price_usd"], errors="coerce"),
            "t": pd.to_numeric(frame["volume_total_usd"], errors="coerce"),
            "m": pd.to_numeric(frame["market_cap_usd"], errors="coerce"),
            "v": pd.to_numeric(frame["velocity_total"], errors="coerce"),
        }
    )
    work = work[(work > 0).all(axis=1)].dropna()
    if len(work) < min_rows:
        return None

    design = sm.add_constant(
        np.column_stack([np.log(work["t"]), np.log(work["m"]), np.log(1.0 / work["v"])])
    )
    result = sm.OLS(np.log(work["price"].to_numpy()), design).fit()
    intervals = result.conf_int()
    fit: dict = {
        "alpha": float(result.params[0]),
        "adj_r_squared": float(result.rsquared_adj),
        "durbin_watson": float(sm.stats.stattools.durbin_watson(result.resid)),
        "n": int(len(work)),
    }
    for index, name in enumerate(EOE_TERMS, start=1):
        fit[name] = float(result.params[index])
        fit[f"{name}_ci_low"] = float(intervals[index][0])
        fit[f"{name}_ci_high"] = float(intervals[index][1])
    return fit


@register
class KampakisLogLinearEoe:
    spec = _SPECS["kampakis_log_linear_eoe"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("fit_window_rows", 365))
        min_fit = int(params.get("min_fit_rows", 180))
        published = list(params.get("published_beta", [0.88, 0.84, 1.15]))
        phash = param_set_hash(params)

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            start = max(0, position - window)
            fit = fit_log_linear_eoe(frame.iloc[start:position], min_fit)
            if fit is None:
                out.append(_refuse(self.spec, row, params,
                                   f"fewer than {min_fit} usable rows in the trailing {window}-row window"))
                continue
            t, m, v = (_finite(row.volume_total_usd), _finite(row.market_cap_usd), _finite(row.velocity_total))
            if not all(x and x > 0 for x in (t, m, v)):
                out.append(_refuse(self.spec, row, params, "no positive volume, market cap or velocity on this day"))
                continue
            predicted = float(
                np.exp(fit["alpha"] + fit["beta_t"] * np.log(t) + fit["beta_m"] * np.log(m)
                       + fit["beta_inv_v"] * np.log(1.0 / v))
            )
            out.append(
                Estimate(
                    symbol=row.symbol, metric_date=row.metric_date, method_id=self.spec.method_id,
                    param_set_hash=phash, fair_value=predicted, band_low=None, band_high=None,
                    inputs_used={"T": t, "M": m, "V": v, "fit_window_rows": window},
                    diagnostics={
                        **fit,
                        "published_beta": published,
                        # Declared here because the registry's flag structurally cannot see it.
                        "structurally_circular": True,
                        "circularity_note": (
                            "M is price x supply and V is T / M, so log(price) is regressed on two "
                            "transforms of log(price); the fit is in part an identity"
                        ),
                    },
                    applicable=True, panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_methods_monetary.py -v && .venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 6: Measure the re-estimated coefficients against the published ones**

```bash
.venv/bin/python -m cryptoval.cli estimate --methods kampakis_log_linear_eoe
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
print(con.execute("""
  select symbol,
         round(cast(json_extract_string(diagnostics,'$.beta_t') as double),3) as beta_T,
         round(cast(json_extract_string(diagnostics,'$.beta_m') as double),3) as beta_M,
         round(cast(json_extract_string(diagnostics,'$.beta_inv_v') as double),3) as beta_invV,
         round(cast(json_extract_string(diagnostics,'$.adj_r_squared') as double),4) as adj_r2,
         round(cast(json_extract_string(diagnostics,'$.durbin_watson') as double),3) as dw
  from fct_valuation_estimate
  where method_id='kampakis_log_linear_eoe' and applicable
    and metric_date=(select max(metric_date) from fct_valuation_estimate)
  order by symbol
""").fetchdf().to_string())
PY
```

Record per asset: the three β estimates against the published (0.88, 0.84, 1.15), the adjusted R²
against the published 0.97, and the Durbin-Watson. This is the task's finding, and Task 12's report
section states it.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/methods/monetary.py config/methods/kampakis_log_linear_eoe.yml \
        tests/test_methods_monetary.py
git commit -m "feat: the Kampakis log-linear EoE, with the circularity the registry cannot see

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 12: Cards, the full estimate run, and the Phase 3a chapter

Plan 2's Task 16 built `make cards` (reading `failure_modes` and `assumptions` from each method's
YAML via `load_card_extras`) and `reports/dissertation-comparison.md`. This task regenerates every
card, runs the whole registry, and writes the chapter that is Phase 3a's deliverable — including a
**declared-gap record** for the §10.5 multiples no free source supports, because a documented
refusal with a measurement is this project's output format for an unavailable method.

**Files:**
- Regenerate: `docs/methods/*.md` (14 existing + 21 new = 35)
- Create: `reports/modern-layer.md`
- Modify: `tests/test_cards.py`

**Interfaces:**
- Consumes: `write_cards`, `load_card_extras`, `render_card` from `cryptoval.cards`; the registry.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cards.py -- append
import pathlib

from cryptoval.cards import load_card_extras
from cryptoval.methods.registry import METHODS, load_method_specs


def test_every_registered_method_has_a_config_a_card_and_written_failure_modes():
    specs = load_method_specs()
    extras = load_card_extras()
    cards = {p.stem for p in pathlib.Path("docs/methods").glob("*.md")}

    assert set(METHODS) == set(specs), "a registered method with no config, or the reverse"
    for method_id in specs:
        assert method_id in cards, f"{method_id} has no method card"
        modes = extras.get(method_id, {}).get("failure_modes") or []
        assert modes, f"{method_id} has no written failure modes"
        assert all(m.strip() for m in modes), f"{method_id} has an empty failure mode"


def test_the_registry_holds_every_method_this_plan_and_plan_2_built():
    # 14 from Plan 2 + 21 from Plan 3. A number that drifts without a plan entry is a
    # method someone added without a card or a config.
    assert len(METHODS) == 35
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cards.py -v`
Expected: FAIL on the card existence assertion — the 21 new cards are not generated yet. (If it
fails on the count instead, a task was skipped; reconcile before continuing.)

- [ ] **Step 3: Regenerate the cards and run the whole registry**

```bash
make cards
make dbt
.venv/bin/python -m cryptoval.cli estimate
.venv/bin/python -m cryptoval.cli validate; echo "validate exit=$?"
.venv/bin/python -m pytest -q
.venv/bin/ruff check cryptoval tests
```

- [ ] **Step 4: Measure what to write about, before writing it**

```bash
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
latest = "(select max(metric_date) from fct_valuation_estimate)"
print("--- coverage per method ---")
print(con.execute(f"""
  select m.family, e.method_id,
         sum(case when e.applicable then 1 else 0 end) as applicable_rows,
         count(distinct case when e.applicable then e.symbol end) as symbols,
         count(distinct e.param_set_hash) as param_sets
  from fct_valuation_estimate e join dim_method m on m.method_id = e.method_id
  group by 1,2 order by 1,2
""").fetchdf().to_string())
print("--- the spread per asset, latest date, fair-value methods only ---")
print(con.execute(f"""
  select symbol, count(distinct method_id) as methods,
         round(min(fair_value),4) as low, round(max(fair_value),4) as high,
         round(max(fair_value)/nullif(min(fair_value),0),1) as spread_x
  from fct_valuation_estimate
  where applicable and fair_value is not null and metric_date={latest}
  group by 1 order by spread_x desc nulls last
""").fetchdf().to_string())
print("--- circular vs independent ---")
print(con.execute(f"""
  select case when circular_inputs <> '' then 'circular' else 'independent' end as kind,
         count(distinct method_id) as methods
  from fct_valuation_estimate group by 1
""").fetchdf().to_string())
PY
```

- [ ] **Step 5: Write `reports/modern-layer.md`**

Six sections. Every number in it comes from Step 4's queries or from a task report, and every claim
that rests on a parameter says which parameter.

1. **What Phase 3a adds, and what it does not.** 21 methods across five families; the free-data
   boundary that shaped them; and the explicit statement that Family E(b) (Cointime, HODL waves,
   SLRV/STHPL/RPV) is absent pending the BigQuery extraction of Plan 3b, with spec §6.3's
   all-or-nothing reason.
2. **The DCF chapter.** The ETH grid in full, with terminal shares, and how many cells are
   `terminal_dominated`. The DeFi DCF's coverage — which of the nine revenue-bearing symbols retain
   revenue and which are refused for not retaining it. The bond-style target yield's dependence on
   two hand-set parameters.
3. **Where the DCF and its counter-model disagree** (spec §10.4's requirement). Per asset: the DCF
   range, the counter-model's value, and the ratio. State plainly which assets the two agree on,
   if any, and what the disagreement implies about applying cash-flow arithmetic to an L1 token.
4. **The falsified shelf, measured.** Stock-to-flow's and the power law's β, R², Durbin-Watson and
   fair-value-to-price ratio. The point is that a high R² with a Durbin-Watson near zero is what
   a spurious fit looks like, and both shelf methods show it.
5. **The circularity audit, extended.** Plan 2's comparison report has a circularity section; this
   extends it with the methods added here, and in particular with `kampakis_log_linear_eoe`, whose
   circularity the registry's automatic flag cannot detect, and `energy_value`, which is 2× a
   smoothed `hayes_cost_of_production` and therefore not independent evidence.
6. **Declared gaps: the methods free data cannot support.** One row per gap, each with the
   measurement that caused it:

   | Spec | Method | Why it is not implemented |
   |---|---|---|
   | §10.5 | Market-cap-to-net-assets | No free source serves protocol treasury or net asset value; DefiLlama's free surface carries fees, revenue and TVL, none of which is a balance sheet |
   | §10.4 | Real yield's incentive term | No free per-asset incentive-emissions series; implemented with the term parameterised to zero and every row flagged `incentive_share_available: false`, so the published figure is a lower bound |
   | §10.5 | P/TVL *(only if Task 6 was skipped)* | `GET /protocol/<slug>` returns 30.2 MB per protocol with no way to exclude the unused breakdowns — ~600 MB for one backfill, for one multiple |
   | §10.6(b) | Cointime, HODL waves, SLRV, STHPL, RPV, reserve risk | Requires the §6.3 BigQuery UTXO extraction; cumulative from genesis, so partial history yields wrong values rather than partial ones. Plan 3b |
   | §10.3 | Peterson's `b_t` (bitcoin-days-destroyed) | Not on the free tier, so Peterson's scale constant is *fitted* on a trailing window; the functional form is his, the calibration is not |

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`
Expected: whole suite green, lint clean, and `test_the_registry_holds_every_method_this_plan_and_plan_2_built` passing at 35.

- [ ] **Step 7: Commit**

```bash
git add docs/methods reports/modern-layer.md tests/test_cards.py
git commit -m "feat: regenerate 35 method cards and write the Phase 3a chapter

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Done criteria for Plan 3

**Stage 3a-1:**
- `make lint && make test` green; roughly 420 unit tests, no network.
- Nine new methods registered with configs, cards and `dim_method` rows: `stock_to_flow`,
  `power_law`, `puell_multiple`, `mayer_multiple`, `mvrv_z_score`, `energy_value`,
  `peterson_metcalfe`, `nvm_ratio`, `lppls_bubble_flag`.
- `lppls_bubble_flag` reports `bubble_regime: false` with `critical_time: null` where its `tc` bound
  or its RMSE floor binds — a date appears only when both gates pass.
- `energy_value`'s diagnostics name `hayes_cost_of_production` as their source on every row, and the
  measured energy/Hayes ratio is recorded in the task report.

**Stage 3a-2:**
- `fct_protocol_revenue_daily` holds **years, not days**: ETH ≈ 4,086 rows from 2015-08, AAVE ≈ 1,671
  from 2022-03, each row carrying `series_source`.
- No leading-zero pre-existence rows are stored for any protocol, and the trim count per protocol
  appears in the ingest run's errors.
- `dim_asset` carries `total_supply`, `max_supply` and `supply_as_of`, with ETH's `max_supply` NULL.
- `_read_panel` returns the widened frame, and re-running `estimate` changes **no** fair value or
  applicability of any method that existed before Task 7.
- `make validate` exits 0, with any new warnings either unmapped-entity warnings or named in a task
  report with their measurement.

**Stage 3a-3:**
- All 35 methods registered, each with a config, a card whose failure modes are written, and a
  `dim_method` row; `tests/test_cards.py` enforces all three.
- `eth_three_stage_dcf` emits one row per grid cell per date with distinct `param_set_hash` values,
  and every row carries `terminal_share`.
- `three_stage_pv` raises rather than returning a value when the discount rate is at or below
  terminal growth, and the refusal reaches the fact table as a reason.
- `defi_net_revenue_dcf` refuses every protocol whose `revenue_regime` is `fees_fully_distributed`,
  naming the regime.
- `fdv_to_revenue` and `dfdv_to_revenue` refuse ETH for want of a `max_supply` and never substitute
  `total_supply`.
- `sector_cohort_zscore` runs cross-sectionally, scores only cohorts at or above
  `min_cohort_size`, and refuses the rest with the measured cohort size.
- `kampakis_log_linear_eoe` reports re-estimated β against the published (0.88, 0.84, 1.15) and
  declares `structurally_circular: true`.
- `reports/modern-layer.md` states the DCF/counter-model disagreement per asset and carries the
  declared-gap table with a measurement in every row.

**Plan 4 starts from:** 35 methods in `fct_valuation_estimate` with `panel_vintage_ts` on every row,
the `param_set_hash` grid mechanism (which the §11 walk-forward needs to score a parameterisation
rather than a method), the Family F shelf as a deliberately weak benchmark to beat, the
`cross_sectional` dispatch (which §11.5's factor models need), and the declared-gap record so Plan 4
does not score a method that was never built.

**Plan 3b (separate, written against spec §6.3 alone):** the BigQuery sandbox extraction behind the
dry-run byte gate and monthly byte ledger, materialised to our own Parquet, then Family E(b) flipping
from `applicable=False` to live in one step.

---

## Self-review

Run against the spec with fresh eyes after writing the plan, per the writing-plans skill.

**1. Spec coverage.** One real gap found, and it was in Plan 2 rather than in this plan: spec §10.2
lists four Family A specifications and Plan 2 implemented three of them, omitting the Kampakis
log-linear EoE while recording §10.2 as covered. **Task 11 was added to this plan for it.** Section
by section: §10.3's remaining three → Task 3; §10.4 in full → Task 10 (six methods, with the
incentive term's missing source declared rather than faked); §10.5's remaining multiples → Task 8,
cohort z-scores → Task 9, market-cap-to-net-assets → declared gap with its reason; §10.6(a)'s
remaining four → Task 2; §10.7 → Task 1. §10.6(b), §10.8, §11 and §12's full report are out of scope
by the boundary stated in the header, each with the plan that owns it named.

**2. Placeholder scan.** No "TBD", no "add error handling", no "similar to Task N", no test step
without test code. Three places defer a *decision* to a measurement rather than deferring work:
Task 6's implement-or-declare-a-gap choice (with the 30.2 MB measurement that decides it), Task 10
Step 1's determination of what ETH's revenue column contains before the DCF is written, and Task 4
Step 6's calibration of `max_null_fraction` from live nullity. Each names the measurement to take,
where to record it, and what each outcome implies — that is a measurement step, not a placeholder.

**3. Type consistency.** `_finite` and `_refuse` keep their Plan 2 signatures throughout.
`fit_log_log` is defined once in Task 1 and reused by Tasks 3 and 9's modules rather than
reimplemented. `MethodSpec.cross_sectional` is added in Task 9 with a default of `False`, so every
config written in Tasks 1-8 stays valid. The column names the methods declare in `requires`
(`total_fees_usd`, `net_revenue_usd`, `tvl_usd`, `max_supply`, `total_supply`, `supply_as_of`,
`sector`, `genesis_date`, `revenue_regime`) are exactly the ones Task 7's `_widen_panel` produces.
Note the deliberate rename at the mart boundary: the snapshot columns were `total_fees_24h` and
`net_revenue_24h`, and the history columns are `total_fees_usd` and `net_revenue_usd` — Task 4's mart
model maps the former onto the latter, and no method references the `_24h` names.

**4. Review Focus.** All five items have a test in the task that owns the code: terminal-value
dominance → Task 10 (`test_eth_dcf_flags_a_cell_whose_value_is_mostly_terminal`); revenue zeros that
are not zeros → Task 4 (`test_trim_leading_zeros_drops_pre_existence_but_keeps_interior_zeros`);
FDV without a max supply → Task 8 (`test_fdv_to_revenue_refuses_when_max_supply_is_null`); a cohort
of one → Task 9 (`test_cohort_of_one_is_refused_with_its_measured_size`); an LPPLS fit with no bubble
→ Task 3 (`test_lppls_refuses_when_the_critical_time_hits_its_bound`).

**One risk the executor must carry.** Task 7 widens the frame every method reads. Plan 2's Task 16
already demonstrated that a change to what the registry sees — `covered_span` in that case — moves
live numbers while every unit test stays green. Task 7 Step 5 exists for exactly that: it diffs every
pre-existing estimate against the stored run and expects zero changes. Do not skip it, and do not
proceed to Task 8 with a non-zero count.
