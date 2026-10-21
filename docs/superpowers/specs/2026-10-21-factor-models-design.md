# Cross-Sectional Asset Pricing — Design (Plan 4c-factors, spec §11.5)

Spec §11.5: "Liu-Tsyvinski-Wu three-factor model (market, size, momentum) plus a value factor built
from active addresses over market cap; Fama-MacBeth regressions; decile portfolios sorted on each
valuation signal. This answers the question the dissertation's correlation chapter could not:
**does a valuation signal actually earn returns?** Built on a point-in-time top-N universe (§5)
rather than the curated 23, with point-in-time sector labels (§8.3)."

§11.5 was blocked on that universe. The 2026-10-18 spike found no free source for a true
point-in-time top-N list: CoinGecko, CoinPaprika and CryptoCompare/CoinDesk all refuse history
beyond 365 days without a key or a paid plan. The user chose (2026-10-21) to build on the **Coin
Metrics community universe**: every asset whose catalog entry serves `CapMrktCurUSD` at daily
frequency, 135 listings at the time of writing, 30 of them already ended. Written without a design
dialogue at the user's standing instruction; every call a reviewer might make differently is marked
**Decision**.

## 1. What the universe is, and what it is not

* **Membership is point-in-time inside the catalog.** An asset is in week *t*'s cross-section only
  if Coin Metrics serves a market cap for it dated *t*. No asset enters before its data does, and a
  dead asset stays in every week it traded (30 ended listings, from `eos_eth` in 2018 to `ftt`).
* **The catalog itself is chosen ex post.** Coin Metrics decides which assets to cover today. The
  holes in a true top 20 are known: BNB has no market cap after 2019-04-22, SOL is absent, DOT
  ends 2022-06-03, and TON, IOTA and Nano are missing. This is **selection bias**, declared in the
  chapter's first paragraph, not corrected. The direction is unknown: covered assets skew to large
  and long-lived, but the set includes dead ones.
* **XRP's market cap counts a different supply** (it ranks above BTC on 2018-01-07). Kept as served
  and declared: value weighting gives XRP more weight than circulating-supply data would.
* **No sector labels.** Coin Metrics serves none, and §8.3's point-in-time labels exist only for the
  curated 23. Sector-neutral sorts are a **declared gap**.

## 2. Decision: exclusions are a config list, with a reason per entry

`config/factors.yml` lists the listings removed from the universe. Exclusions depend on what an asset
*is*, never on how it performed:

* **Pegged assets** (stablecoins on every chain, plus `xaut` and `paxg`): their price is not a market
  risk quantity.
* **Wrappers** (`wbtc`, `weth`, `renbtc`, `hbtc`, `wnxm`): a claim on another listed asset, which
  would enter the cross-section twice.
* **The same token on another chain** (`avaxc` and `avaxx` beside `avaxp`, `leo_eos` beside `leo_eth`,
  `bnb_eth` beside `bnb`, `flow` beside `flow_native`): keep one listing, the one with the larger cap
  or the one with active addresses. Coin Metrics prices them identically, so returns are unaffected;
  size is understated for multi-chain tokens. Declared.

The loader refuses an exclusion id that is not in the fetched catalog (a typo would silently keep
the asset), and refuses a duplicate entry.

## 3. Data

* **Fetch** `PriceUSD, CapMrktCurUSD, AdrActCnt, CapMVRVCur`, daily, full history, for every catalog
  id, with `ignore_unsupported_errors=true` (assets lack some metrics). About 390,000 rows in 39
  pages, under a minute. The fetch writes silver `cm_universe_daily` (`asset, metric_date,
  price_usd, market_cap_usd, active_addresses, mvrv`) as a full replacement: the history is small,
  and an incremental merge would add code to save a minute.
* **Validation at the boundary:** values are coerced to numbers; a non-positive price or market cap
  becomes null; a duplicate `(asset, metric_date)` raises.
* `CapRealUSD` and `TxTfrValAdjUSD` are not free, so NVT is not testable here. MVRV is
  (`CapMVRVCur` on 125 listings).

## 4. Decision: weekly, Sunday to Sunday, as Liu-Tsyvinski-Wu

* A week is the observation **dated each Sunday**. No forward fill: an asset with no row on a
  Sunday is absent that week.
* **Eligible at week t:** not excluded, price and market cap present at *t*, market cap at least
  **$1M** (LTW's floor), and a price present at *t+1* (the return being explained).
* **Return:** simple, `p(t+1)/p(t) − 1`. An asset with no price at *t+1* has no return and leaves
  week *t*'s cross-section: a **delisting bias**, declared, not assigned −100%, because the end of
  Coin Metrics coverage is not proof of death.
* **No risk-free rate.** Long-short factors cancel it; the market premium is raw. A weekly T-bill
  return of about 0.1% against a crypto weekly standard deviation near 20% does not move a test.
* **Sample start:** the first week with at least **50 eligible assets**, measured as 2017-12-03 (about
  5 per decile). Earlier weeks hold fewer than 30.
* Measured: the largest eligible weekly return is +962%, and the smallest is FTT's −93% in the FTX
  week. No glitch filter is needed.

## 5. The signals

| Signal | Week-t value | Reading |
|---|---|---|
| `size` | log market cap | LTW CSMB |
| `momentum` | `p(t)/p(t−3 weeks) − 1` | LTW CMOM, 3-week lookback |
| `value` | `active_addresses / market_cap` | §11.5's value factor; high is cheap |
| `mvrv` | `CapMVRVCur` | the dissertation's MVRV; high is expensive |

A signal missing for an asset that week drops the asset from that signal's sort only.

## 6. The tests

**Decile portfolios.** Each week, every signal with at least 50 assets carrying it is sorted into
10 deciles (ascending, ties by asset id). Returns are **value-weighted** by week-*t* market cap,
LTW's choice, because equal weights hand the result to micro-caps. `HL = D10 − D1`. The hypothesis
is HL's mean is zero, Newey-West t with **4 lags**, two-sided. **4 hypotheses.**

**LTW factors.** Single 30/40/30 sorts, value-weighted within each leg: `CMKT` = value-weighted
eligible market; `CSMB` = bottom 30% by size minus top 30%; `CMOM` = top 30% by momentum minus bottom
30%. LTW double-sort momentum within size halves; with 50-110 assets a 2×3 sort leaves legs of
8-18 assets, so **Decision:** single sorts, declared. The hypothesis is that each factor's mean is
zero (LTW report all three priced). **3 hypotheses.**

**Alphas.** HL for `value` and for `mvrv` regressed on `CMKT, CSMB, CMOM` (OLS, Newey-West 4 lags).
The hypothesis is the intercept is zero: does the valuation signal earn anything the three factors
do not already price? Size and momentum are factors themselves, so they get no alpha. **2 hypotheses.**

**Fama-MacBeth.** Each week, the assets carrying all four signals (at least 50, else the week is
skipped) are regressed on each signal's cross-sectional percentile rank in [0, 1] (so a slope is
the top-minus-bottom return spread). Returns are **winsorized at the week's 5th and 95th
percentiles** for this regression only: OLS weights every asset equally, and one +900% micro-cap
would set the slope. The weekly slopes' means are tested with Newey-West 4 lags. **4 hypotheses.**

**One Holm family of 13**, through the shared `correct_family_wise`, which also writes the count
row. Full sample, descriptive. Unlike Plan 4's forecasts, sorts on week-*t* data against week-*t+1*
returns have no look-ahead, but the choice of signals, lookbacks and floors was made in 2026 with
the literature in hand. The chapter claims "this signal earned returns in this sample", never
"tradable".

## 7. Output

Gold `factor_test`, one long table in the house shape (`fct_volatility_test`):

`model_id, signal, portfolio, metric, value, n_weeks, reason, params_hash, panel_vintage_ts`

* `model_id` ∈ `decile, factor, alpha, fama_macbeth`, and `'*'` for the hypothesis-count row.
* `portfolio` is `D1`..`D10` and `HL` for decile rows, and null otherwise.
* Metrics: `mean_return`, `t_stat`, `p_value`, `p_value_holm`, plus `beta_cmkt`, `beta_csmb` and
  `beta_cmom` on alpha rows, and `mean_assets` (average cross-section size).
* A test that cannot run (too few weeks) is a `refused` row with the reason, never an absent row.

Gold `factor_returns` (`week, series, value`) holds the weekly HL and factor series, so the chapter
and any later reader can recompute. dbt marts `fct_factor_test` and `fct_factor_returns` read both,
with a uniqueness test on each grain.

## 8. Wiring

* `cryptoval factors fetch` (network) and `cryptoval factors run`; `make factors` runs both, and
  `make all` runs it after `vol` and before the last `dbt`.
* Chapter `reports/do-valuation-signals-earn-returns.md` from a live run, with §1's biases first.
  The scorecard row in `reports/dissertation-comparison.md` for §11.5 is updated.

## 9. Out of scope

Sector-neutral sorts (no labels); NVT and realized-cap signals (not free); transaction costs and
capacity (no tradability claim); a key-based point-in-time top-N universe (option (b), not chosen).
