# Cryptoasset Valuation Pipeline — Design Specification

| | |
|---|---|
| **Date** | 2026-10-11, revised 2026-10-12 |
| **Status** | Phase 0-1 implemented and merged; Phase 2 planned |
| **Author** | Yasir Arafat (with Claude Code) |
| **Source material** | `Cryptoassets-Valuation-Dissertation-Yasir-Arafat.pdf` — MSc FinTech, University of Liverpool, Dec 2022 |

### Revision log

**2026-10-12 — revised against the live panel Plan 1 produced.** Building the pipeline taught
things the design could not have known, and several of them contradicted this document. Where the
measured world and the spec disagreed, the spec was wrong:

| § | Change | Driver |
|---|---|---|
| 5, **5.1 (new)** | An asset's identity is a map from metric to vendor id, not one id. TRX, AVAX and POL corrected; coverage gaps for TRX/POL/BNB recorded as declared limits | Coin Metrics served the right metrics for the wrong entity: `trx_eth` is a near-dead ERC-20 wrapper, and Plan 1 put it in the panel as TRX |
| 5, 15 | Gold and the dollar index added; both open questions resolved | Q1 and Q2 answered |
| 5.1 | Protocol identity is a *list* of DefiLlama slugs, version-summed and rebrand-aware | "Maker" no longer exists in the feed; it is Sky Lending plus Sky Money. Uniswap, Aave, Compound and Synthetix each span multiple versions |
| 7.4 | Bronze partitions carry `run_id` | Date-only partitioning made a same-day re-run destroy the earlier run's audit copy — the one thing the layer guarantees |
| 8.1 | Facts key on `symbol`; `dim_asset` seeded from config; equity fact separated | Nothing joined the four sources: each keyed on its own vendor's identifier |
| 9.1 | A fourth contract, per-source freshness, and the quarantine table | Equity markets close, so a global 48h SLA failed every Monday; and 14 impossible volume values needed somewhere honest to go |
| 9.2 | Volume anomaly warns on a 30-observation window, an absolute bound blocks, null-rate ceilings added, gross fees floored but net revenue unbounded below, blocking scoped to mapped protocols | The original guard flagged 68.8% of healthy asset-days; the original revenue floor blocked the build on insurance, options and betting protocols whose negative revenue is real |
| 10.1 | Applicability computed from declared dependencies and coverage; both volume definitions carried, never coalesced | The two sources' volume differs 5× on BTC, and velocity, NVT and P/U all turn on it |
| 13 | Phase 2 absorbs a remediation stage and the holding-time correction | Every item above corrupts a method silently if found after the methods exist |

---

## 1. Context

The source dissertation surveyed cryptoasset valuation across five approaches (monetary, income,
market, cost, profit-and-loss), then ran an empirical chapter comparing linear regression, SVR and a
Keras neural network on Bitcoin closing prices, plus Pearson correlation matrices across crypto,
equities and NFT/metaverse assets.

Three things have changed since it was written, and one thing in it needs correcting.

**Changed — DCF became applicable.** The dissertation concluded cryptoassets "generate neither cash
flows nor earnings." For a large share of the market that is no longer true. Post-Merge Ethereum is
routinely modelled as a cash-flow asset (EIP-1559 burn as buyback, validator issuance as
stock-based compensation, staking yield as dividend), DeFi protocols publish auditable revenue, and
the CFA Institute codified DCF plus relative valuation for cryptoassets in November 2023.

**Changed — the on-chain cost family was rebuilt.** Cointime Economics (Puell/ARK with
Check/Glassnode) decomposes supply into active and vaulted regions via Liveliness, yielding
Cointime-adjusted MVRV, a Cointime stock-to-flow and AVIV. The dissertation's MVRV, MVTV, Investor
Cap and SLRV sections all now sit inside that newer framework.

**Changed — the QTM chapter has a direct successor.** Kampakis et al. (2024) demonstrate that the
`H = 1/V` holding-time inversion underpinning the dissertation's QTM and Price-to-Utility sections
is incorrect at the mean, and fit an empirical log-linear replacement.

**Needs correcting — Chapter 7-8 results are a methodological artifact.** The code creates
`Prediction = Close.shift(-N)`, uses `Close` as the sole feature, and splits with
`train_test_split`, which shuffles by default — so test rows are interleaved between training rows.
Predicting price from price on a shuffled split guarantees a near-perfect score at N=30 and
degrading scores at longer horizons. The finding that "linear regression beat SVR and the neural
network" measures leakage plus trivial persistence, not model quality. Correcting this is the most
valuable single change in this project.

---

## 2. Goals

1. **Reproduce** every valuation method in the dissertation from live data, as a verified baseline.
2. **Extend** the framework with the post-2022 methods above, including DCF where it genuinely applies.
3. **Re-run the empirical chapter correctly** — strict time-ordered validation, a naive baseline that
   every model must beat, and significance testing of differences.
4. **Ship it as production-grade data engineering** — contracts, tests, lineage, point-in-time
   correctness, CI — deployable to cloud on a schedule without a rewrite.
5. **Produce two consumers off one validated core**: a written research study and a live dashboard.

## 3. Non-goals

- Investment advice or a trading system. Outputs are valuation estimates with stated assumptions.
- Real-time or intraday fundamentals. The underlying metrics are daily-close concepts (§6).
- Security tokens and NFTs as asset classes. The dissertation excluded security tokens because
  conventional equity/debt machinery applies; that exclusion stands.
- Streaming infrastructure (§7.1).
- Beating the market. If a model does not beat a random walk, the project reports that it did not.

## 4. Success criteria

| Criterion | Test |
|---|---|
| Every dissertation method reproduced | Each has a `dim_method` row, a method card, and non-null estimates across the sample |
| No look-ahead bias anywhere | Point-in-time queries reconstruct any past vintage; backtests join point-in-time sector labels |
| Honest predictive claims | Every model scored against naive persistence with a Diebold-Mariano p-value; non-results reported as non-results |
| Reproducible | A clean clone plus `make all` reproduces every figure and table in the report from raw sources |
| Cloud-portable | Switching to object storage plus a warehouse target is configuration only, verified by running the suite against both |
| Costs nothing to run | Free tiers only, no exceptions. BigQuery access via the no-billing-account sandbox, with a dry-run byte gate and a monthly byte ledger (§6.3) |

---

## 5. Asset scope

24 cryptoassets — 22 valuation targets, one stablecoin control and one documented gap (SOL) —
chosen against verified free-tier coverage, spanning the sectors each method family needs. Plus
seven equities and macro series to extend the dissertation's §8.4.4.

| Sector | Assets | Why |
|---|---|---|
| PoW monetary | BTC, LTC, BCH, DOGE | Cost-of-production and Thermocap require proof-of-work |
| Smart-contract L1 | ETH, ADA, DOT, AVAX, ALGO, TRX, ICP | Fee/DCF models; the dissertation's own comparison set |
| Scaling | POL | Fee capture under a rollup-centric roadmap |
| DeFi | UNI, AAVE, CRV, MKR, COMP, SNX, LDO | Net protocol revenue, real yield, multiples |
| Oracle | LINK | The dissertation's set; utility-token archetype |
| Exchange token | BNB | Burn mechanics plus exchange revenue |
| Metaverse | MANA | Continuity with §8.4.5 |
| Stablecoin control | USDC | Velocity/EoE control group (Kampakis use USDT identically) |
| Equities | TSLA, AAPL, AMZN, GOOGL, META | Extends §8.4.4 correlation analysis |
| Macro | GC=F (gold), DX-Y.NYB (dollar index) | §15 Q1 resolved 2026-10-12: the macro-correlation question has become the more interesting one since 2022, and both series are two more tickers through an adapter that already works |

**Known gaps, measured 2026-10-12 against the loaded panel.** Free-tier coverage is not uniform, and
three assets carry limits that must reach the method layer as `applicable=False` with a reason rather
than as quiet nulls:

| Asset | Limit | Consequence |
|---|---|---|
| **SOL** | Absent from Coin Metrics' free tier (`CapMrktCurUSD` returns 403) | CoinGecko price and market cap only; excluded from every network-dependent method. AVAX and ALGO substitute |
| **TRX** | See §5.1 — no free market cap, MVRV or supply on the real TRON series | Families A, B and D apply via CoinGecko market cap; the cost family (MVRV, realized cap, Thermocap, MVTV, Investor Cap) does not apply at all |
| **POL** | History starts 2023-10-25 (1,082 days). The pre-rebrand MATIC series returns no data on the free tier, so **no splice is possible** | Excluded from long-horizon cointegration and factor work; short-window methods apply |
| **BNB** | Market cap, MVRV and supply begin 2019-04-22 and are only 19% dense across the loaded window | Cost and MVRV families apply only where present; density is a method precondition, not a null to average over |

**Survivorship bias** is addressed separately in §11.4. These are availability gaps, recorded rather
than hidden — the opposite failure, carrying them as nulls into a mean, is what §9.2's null-rate
ceiling now blocks.

**Survivorship bias:** this set is chosen in 2026 with knowledge of which assets survived. The
cross-sectional factor work (§11.4) therefore uses a separate point-in-time top-N universe rather
than this curated list, and the limitation is stated in the report.

### 5.1 Vendor identity and the conformed key

**Added 2026-10-12.** Each vendor names assets its own way, and for three assets the free tier
serves the right *metrics* under one id and the right *entity* under another. A single
`cm_id` per asset cannot express that, and Plan 1's one-id-per-asset choice silently put the wrong
entity in the panel for TRX.

Measured on the free tier 2026-10-12:

| Asset | Id serving network metrics | Id serving market cap / MVRV / supply | Reading |
|---|---|---|---|
| **TRX** | `trx` — real TRON. AdrActCnt, TxCnt, reported spot volume, price, with history to at least 2020-01-01 | *none free.* `trx_eth` serves them, but for the near-dead ERC-20 wrapper: market cap NULL on the latest day, volume 0, credible maximum volume $165K against real TRON's $254M/day | Take network metrics from `trx`, market cap from CoinGecko. MVRV and realized cap are unavailable, so the cost family does not apply |
| **AVAX** | `avaxc` — the C-Chain, the genuine primary EVM chain, **not** a wrapper | `avaxc`, but it serves **no** reported spot volume (0% across the panel) | Take volume from `avax`, everything else from `avaxc`. This is why velocity was uncomputable for AVAX |
| **POL** | `pol_eth` (7 metrics, from 2023-10-25) | `pol_eth` | `pol` serves only price and volume; `matic` returns no data. Short history is irreducible |

**The rule this establishes.** An asset's identity is a *map from metric to vendor id*, not a single
id. `config/assets.yml` carries that map; bronze and silver stay vendor-faithful, keyed by the
vendor's own id, because rewriting a vendor's key in the landing zone would destroy the audit trail
the layer exists for.

**The conformed key is `symbol`.** Resolution from vendor id to symbol happens in the staging layer,
and `dim_asset` is seeded from `assets.yml` — symbol, sector, consensus, `is_stablecoin`, and every
vendor id as an attribute — rather than derived from whatever the panel happens to contain. Every
fact table downstream of staging keys on `symbol`, which is what makes the four sources joinable at
all: before this, Coin Metrics keyed on `trx_eth`, CoinGecko on `tron`, DefiLlama on protocol display
names and yfinance on tickers, and nothing joined them.

**Protocol identity is a list, not an id.** DefiLlama splits a protocol across versions and renames
it on rebrand. Measured 2026-10-12: Uniswap is V4 ($1.37M fees/day) plus V3 ($451K) plus V2 ($40K)
with the mix actively shifting from V3 to V4; Aave spans V2/V3/V4; Compound V2/V3; Synthetix four
entries. MKR's revenue no longer appears under "Maker" at all — it is **Sky Lending** plus **Sky
Money** after the rebrand. So `assets.yml` carries `defillama_slugs` as a *list* per asset, summed
before any multiple or DCF is computed, and an unmapped asset returns `applicable=False` rather than
zero. A single-version, pre-rebrand mapping would have produced confidently wrong DCF inputs for six
of the seven DeFi assets.

---

## 6. Data sources

### 6.1 Verified available (tested 2026-10-11, free, no API key)

| Source | Provides | Verified |
|---|---|---|
| **Coin Metrics Community** | 31 catalogued series × 100 assets (including reference rates and completion timestamps): `CapMrktCurUSD`, `CapMVRVCur`, `HashRate`, `IssTotUSD`, `FeeTotNtv`, `AdrActCnt`, `AdrBalCnt`, `TxCnt`, `TxTfrCnt`, `SplyCur`, `FlowInExUSD`/`FlowOutExUSD`, `PriceUSD`, `ReferenceRate` | BTC 2026-10-05: mcap 1.725e12, MVRV 1.5967 |
| **DefiLlama** | 2,991 protocols with fee/revenue data; 4,084 days of Ethereum fee history; per-protocol `methodology` field | Confirmed via `/overview/fees` and `/summary/fees/ethereum` |
| **CoinGecko Demo** | price / market_cap / total_volume history — the EoE inputs, same source as Kampakis | Confirmed via `/market_chart` |
| **yfinance** | Equity OHLCV for §8.4.4, plus the gold and dollar-index macro series (§5) | Standard |

### 6.2 The key derivation

Realized capitalisation is **not** free on Coin Metrics (`CapRealUSD` is Pro-tier — confirmed by a
403 response). But `CapMVRVCur` is free, and MVRV is by definition market cap over realized cap, so:

```
realized_cap = CapMrktCurUSD / CapMVRVCur
```

Verified on BTC 2026-10-05: 1.725e12 / 1.5967 = 1.080e12.

This one identity unlocks the dissertation's entire cost approach for free — MVRV Z-score, Thermocap,
MVTV and Investor Capitalisation. It is also a **fragile derivation**: if either input's definition
changes upstream, the result is silently wrong. Hence the identity assertions in §9.2, which are the
highest-value tests in the project.

Derived from free inputs:
- `thermocap = cumsum(IssTotUSD + FeeTotNtv × PriceUSD)`
- `mvtv = market_cap / thermocap`
- `investor_cap = realized_cap − thermocap`
- `puell = IssTotUSD / MA_365(IssTotUSD)`

### 6.3 UTXO-level metrics — free-tier staged

UTXO-level metrics — Cointime primitives (coinblocks created/destroyed, Liveliness, Vaultedness),
HODL waves, and therefore SLRV, STHPL and RPV — have no free API. They require BigQuery's public
`bigquery-public-data.crypto_bitcoin` dataset.

**Decision (2026-10-11): free only. No payment, now or later, without an explicit new decision.**

#### Mechanism: BigQuery sandbox

Access is via the **BigQuery sandbox**, which Google documents as usable "without providing a credit
card or creating a billing account", with the free tier's 1 TB of query processing and 10 GB of
storage per month. With no billing account attached there is no payment instrument to charge.

Honest caveat: Google's documentation does not state in so many words that a sandbox project can
*never* be billed, so the sandbox is treated as the outer guard and our own byte gate as the inner
one, rather than relying on either alone.

#### Two sandbox constraints that shape the design

| Constraint | Consequence |
|---|---|
| **Tables, views and partitions expire after 60 days** | BigQuery is strictly a compute surface, never storage. Every query's aggregates are exported to our own Parquet in the same run. Nothing of value is ever left to live in BigQuery |
| **DML (INSERT/MERGE/UPDATE) is unavailable in the sandbox** | No incremental tables inside BigQuery. The pattern is: aggregate in a `SELECT`, export the small result, accumulate on our side. Fine here — outputs are daily aggregates of roughly 6,000 rows, not raw UTXOs |

#### Cost control

1. **Dry-run byte gate.** Every query dry-runs first and compares estimated bytes against a
   configured budget; over budget, it refuses to execute.
2. **Monthly byte ledger.** Cumulative bytes scanned per calendar month are recorded in
   `run_manifest` and checked before each query, so the gate is aware of the 1 TB monthly envelope
   rather than only of single-query size.
3. **Column pruning.** BigQuery bills by columns scanned, not table size, and the computation needs
   only a handful of narrow fields (value, creation timestamp, spend timestamp). Published figures
   put `crypto_bitcoin.transactions` at roughly 2 TB in total, but no public figure exists for the
   columns we actually touch — so the real number comes from the dry run, not from an estimate here.
4. **Partition pruning** on `block_timestamp_month` is mandatory, not an optimisation. One published
   comparison on this dataset shows 412 GiB scanned unpartitioned versus 161 MiB partitioned.
5. **Staged by halving epoch across calendar months.** The backfill is divided into epoch-sized
   chunks, each sized to fit inside a month's free terabyte, with progress checkpointed so an
   interrupted backfill resumes rather than restarts.
6. **Scan once, ever.** Extracted per-output facts are materialised to our own Parquet, so no epoch
   is ever re-scanned. Re-deriving a metric re-reads our Parquet at zero cost.

#### The consequence that matters

Cointime primitives are **cumulative from genesis** — Liveliness is
`cumsum(coinblocks destroyed) / cumsum(coinblocks created)` over all history. A partially completed
backfill therefore yields *wrong* values, not partial ones. This family cannot be released
incrementally: it stays `applicable=False` until the staged backfill completes in full, which may
take several months of free-tier allowance.

That is a scheduling consequence, not a blocker — §13 splits Phase 3 so that everything else
proceeds while the staged extraction runs in the background.

**Documented alternative, not chosen now:** running a Bitcoin full node with a custom UTXO-age
indexer is also free of monetary cost and has no monthly envelope, at the price of roughly 700 GB of
disk, an initial sync, and materially more engineering. Recorded in case the staged timeline proves
too slow.

### 6.4 Freshness reality

Fundamentals are daily-close by construction and published T+1 after UTC day close, **with
revisions**. Prices can be polled intraday. So the honest definition of "live" is: fresh price
overlaid on daily-cadence fundamentals, refreshed once daily ~1 hour after UTC midnight, with an
optional price-only refresh for the dashboard. Nothing faster is meaningful for these metrics.

---

## 7. Architecture

### 7.1 Decision record

Running the senior-data-engineer decision framework explicitly, because an engineering showcase
invites over-engineering and the framework argues against it here:

| Question | Answer | Consequence |
|---|---|---|
| Real-time insight required? | No — all fundamentals are daily-close concepts | **Batch** |
| Daily volume > 1 TB? | No — ~23 assets × ~31 series × ~4,000 days is single-digit millions of rows | **dbt + local compute, not Spark** |
| Lambda or Kappa? | Neither — no streaming layer to reconcile. Source APIs are the replayable log; bronze is the immutable landing zone, giving Kappa-style reprocessability with no streaming infrastructure | **Pure batch, fully reprocessable** |
| Warehouse or lakehouse? | **Lakehouse-lite** — open Parquet, schema-on-read at bronze, schema-on-write at silver | Same property that makes cloud migration a URI swap |

**Explicitly ruled out: Kafka, Flink, Spark, Delta/Iceberg, Great Expectations, Prometheus,
OpenLineage.** Each is justified only at volumes or latencies this project does not have. Showcase
value goes into contracts, testing, lineage and point-in-time correctness instead. Recorded here so
the omissions read as decisions rather than gaps.

### 7.2 Stack

- **Ingestion**: Python, `httpx` with retry/backoff, date-parameterised idempotent tasks
- **Storage**: Parquet, medallion layout, local filesystem or object storage behind one path resolver
- **Transform**: dbt-core on DuckDB, with a second profile target for BigQuery/Postgres
- **Compute**: DuckDB as a *query engine over Parquet*, never the system of record
- **Models**: Python modules implementing the method-registry contract (§10.1)
- **Dashboard**: Streamlit over the gold layer
- **Orchestration**: `make` plus a `run.py` task DAG initially; Dagster optional at Tier 2
- **CI**: GitHub Actions — unit tests, dbt build on fixtures, Slim CI, `dbt docs` to Pages

### 7.3 Portability contract

The property that makes §13 Phase 6 configuration rather than a rewrite:

| Decision | Local | Cloud |
|---|---|---|
| DuckDB queries Parquet; Parquet is the record | `./data/...` | `gs://bucket/...` via httpfs |
| All paths through one resolver | filesystem | `fsspec` object storage |
| dbt-core, two profile targets | `duckdb` | `bigquery` / `postgres` |
| DuckDB-specific SQL isolated in macros | — | only macros change |
| Tasks are `(source, date_range) → rows` | backfill = wide range | scheduler = narrow range |
| 12-factor secrets, env vars only | `.env` | Actions secrets / Secret Manager |
| Dockerfile + lockfile | `docker run` | identical image on Cloud Run |
| Dashboard reads gold via a data-access module | Parquet on disk | Parquet in object storage |

Coin Metrics Community and DefiLlama require no key, so there is very little secret material.

### 7.4 Layers

- **bronze/** — immutable raw API responses as Parquet, landed per run under
  `source=<s>/ingest_date=<d>/run_id=<r>/`, carrying `fetched_at`, endpoint, request params and
  response hash. Never mutated; the audit trail and the replay source.

  **Revised 2026-10-12.** Plan 1 partitioned on `ingest_date` alone, which made a same-day re-run
  *replace* the partition wholesale — so an operator re-running a subset of assets after a partial
  failure destroyed the response record of everything fetched earlier that day. That contradicted
  this layer's one guarantee. Adding `run_id` to the path makes every run its own partition: nothing
  is ever overwritten, re-running a subset is safe, and the cost is disk measured in megabytes
  (bronze was 6.3 MB at the end of Plan 1). Readers take the latest `run_id` per `ingest_date` unless
  they are explicitly auditing a past run.
- **silver/** — conformed, typed, tested. The bi-temporal SCD2 panel (§8.2) and the dimensions.
  Compacted to `year=` partitions.
- **gold/** — derived metrics, valuation estimates, model scores.
- **marts/** — the two consumers: research tables/figures and the dashboard's wide table.

**Partition sizing.** Daily partitions in silver/gold would create ~4,000 files of ~20 KB — a
textbook small-file problem. Bronze stays per-run partitioned for idempotency and audit; silver and
gold compact to yearly partitions via an explicit compaction task.

---

## 8. Data model

### 8.1 Star schema

| Table | Grain | Notes |
|---|---|---|
| `dim_asset` | symbol | **Seeded from `config/assets.yml`** (§5.1), not derived from the panel: symbol, sector, consensus, `is_stablecoin`, every vendor id, and the metric-to-id map. **SCD2 on sector/classification** (§8.3); SCD1 on cosmetic attributes |
| `dim_date` | date | Carries `halving_epoch`, `days_since_halving`, `cycle_phase` — makes cycle analysis a join, not a recalculation |
| `dim_method` | method_id | Family, formula, citation, assumptions, failure modes, data dependencies. Makes "which methods disagree and why" a SQL question, and is the source of the method cards |
| `fct_asset_metrics_daily` | (symbol, date) | The conformed panel. Keyed on **symbol** (§5.1), joining Coin Metrics network metrics, CoinGecko price/market cap/volume and the resolved vendor ids. Carries **both** volume definitions as distinct columns |
| `fct_protocol_revenue_daily` | (protocol, date) | DefiLlama fees/revenue, with methodology hash. Rolled up to `symbol` through the `defillama_slugs` list (§5.1), summing versions |
| `fct_equity_prices_daily` | (ticker, date) | yfinance equities and macro series (§5), kept a separate fact: different calendar, different freshness SLA, no network metrics |
| `fct_valuation_estimate` | (asset, date, method_id, param_set_hash) | Method registry output. `param_set_hash` lets a DCF at 12% and at 16% coexist as rows rather than overwrite |
| `fct_model_score` | (model_id, asset, fold, horizon, metric) | Walk-forward results. The naive baseline is stored as just another `model_id`, so "did it beat the random walk" is a query, not a claim |
| `obt_asset_valuation_wide` | (asset, date) | Methods pivoted; the single table Streamlit reads |

### 8.2 The bi-temporal panel (SCD Type 2)

Coin Metrics revises published metrics. Point-in-time correctness is therefore not a bespoke
requirement but a standard SCD Type 2, which supplies the merge logic, change detection and query
patterns for free.

```
silver_asset_metrics_scd2
  asset, metric_date                                -- natural key: what the value describes
  <31 metric columns>
  effective_start_ts, effective_end_ts, is_current  -- knowledge axis: when we knew it
  row_hash                                          -- change detection over the metric payload
  source, run_id
```

- Research asks *"what did we know as of date X?"* → `effective_start_ts <= X < effective_end_ts`.
  This is what makes a backtest honest.
- Dashboard asks `is_current`.
- `row_hash` means an identical daily reload writes **nothing**; new versions appear only on genuine
  revisions. The revision history becomes a reportable dataset in its own right.

**Trailing-window refetch.** Because revisions arrive late, the daily run re-fetches a **trailing
30-day window**, not only T-1. Fetching only yesterday would silently miss every revision.

### 8.3 Point-in-time dimensions

Sector classification changes (MATIC→POL; assets migrating category). Using 2026 sector labels on
2019 data in a cross-sectional factor sort is look-ahead bias — the same class of error as the
dissertation's shuffled split, just less visible. `dim_asset` is therefore SCD2 on classification,
and portfolio construction in §11.4 joins point-in-time sector.

### 8.4 Idempotency guarantee

Every ingestion task is `(source, date_range) → deterministic rows`, applied by MERGE on the natural
key within a transaction. Re-running a range is a no-op when nothing upstream changed and a new
SCD2 version when it did. Backfill and daily increment are the same code path with different
arguments — which is also what makes the Phase 6 scheduler a thin wrapper.

---

## 9. Data contracts and quality

### 9.1 Contracts

One YAML contract per source — `coinmetrics_asset_metrics`, `defillama_fees`,
`coingecko_market_chart`, `equity_prices` — each carrying schema, owner, semantic-versioning rules
(which changes are breaking) and quality SLAs: freshness, completeness, uniqueness on
(asset, metric_date), validity ranges, per-column null-rate ceilings, and volume anomaly thresholds.

**Freshness is per source, not global.** The original T+1-by-02:00-UTC SLA assumed a market that
never closes. Equities do: a Monday morning check sees Friday's close, so `equity_prices` allows 144
hours, measured from the worst healthy week observed. Crypto sources keep 48.

**Quarantine, added 2026-10-12.** Some vendor values are impossible rather than merely extreme — the
loaded panel carries 14 reported-spot-volume figures between 1e36 and 1e55 USD against a credible
maximum of 2.3e10, concentrated on `usdc` (12) and `trx_eth` (2). Dropping the whole row would
discard the healthy market cap sitting beside the bad volume, and carrying the value poisons every
velocity figure downstream. So the offending **cell** is nulled in silver and the full record written
to `silver/quarantine` — `(source, asset, metric_date, column, raw_value, reason, run_id)` — with
bronze untouched, so the raw value is always recoverable. Quarantine row-count is itself a monitored
metric: a jump means the vendor broke something new. This is the one place silver deviates from the
vendor's numbers, and it is logged rather than silent.

### 9.2 Domain-specific guards

These matter more than the generic checks:

| Guard | Rationale |
|---|---|
| **Identity assertions**: `mvrv × realized_cap ≈ market_cap`; `supply × price ≈ market_cap`; `thermocap` monotonically non-decreasing; `liveliness + vaultedness = 1` | Realized cap is *derived* from an identity (§6.2). If either input's definition shifts upstream, everything downstream is silently wrong. The single most important test in the project |
| **Volume anomaly threshold** (±50% vs a trailing 30-observation mean) — **warns, does not block** | A DefiLlama methodology revision can 10× a protocol's fees overnight and silently corrupt a DCF. Calibrated against live data 2026-10-11: measured as deviation from an all-time expanding mean it flagged 68.8% of asset-days, because a decade of secular growth reads as a permanent anomaly. On a 30-observation baseline it still flags 34.1% of Coin Metrics asset-days, and no ratio threshold separates Bitcoin Cash's fork day (+246×) or DOGE 2020 (+102×) from vendor garbage. A gate that fires on a third of healthy days is noise, so the ratio warns and an absolute bound blocks |
| **Absolute volume bound** of 1.0e+12 USD — **blocks** | Catches exactly the impossible values (the 14 rows above) and nothing else; the largest credible figure in the same panel is 7.5e10. Residual limit, accepted: a vendor-wide systematic rescale that stays under the bound only warns, because no ratio test distinguishes it from a market event |
| **Per-column null-rate ceiling** — **blocks** | The guard §9.1 promises. A metric silently withdrawn from the feed arrives present-and-all-null, which passes required-columns, uniqueness, not-null and range checks alike; nothing else detects a column going wholly null. Ceilings are calibrated from observed nullity with headroom — market cap 11.7% observed against a 30% ceiling, volume 6.3% against 25%, net revenue 21.8% against 45% |
| **Methodology hash** on DefiLlama's `methodology` field | The field is in the API response; hashing it converts an invisible upstream redefinition into a failed build |
| **Free-tier availability check** | `CapMVRVCur` is free *today*. If it moves to Pro, the contract fails loudly rather than the panel going half-null |
| **Supply monotonicity** for fixed-supply assets | Catches source-side unit or scaling errors |
| **Gross fees ≥ 0**, but **net revenue unbounded below** | Corrected 2026-10-12. The original contract floored all three revenue columns at zero, which is economically wrong for net revenue and was blocking the build on 24 protocols. Negative net revenue is *real*: Ensuro is insurance (claims above premiums), Hegic options (writers lost), WINR and Azuro betting (the house lost), and Fragment shows $1.05M gross fees against −$852K net revenue, which is a rebate structure working as designed. Gross fees cannot be negative, so those rows are vendor errors and go to quarantine |
| **Blocking scope is the valuation universe** | The revenue feed carries 2,991 protocols; at most a handful map to our assets (§5.1). Garbage in the long tail must not block a build that will never value it, so contract failures block only for mapped protocols and warn for the rest |

### 9.3 Tooling decision

**dbt tests as primary; Great Expectations rejected.** dbt is already in the stack, is lineage-aware
and costs nothing extra; GE would be a second overlapping framework for a 24-asset panel. Cross-row and
cross-table checks (identity assertions, distribution drift) live in a small Python check module
invoked by the same task graph.

### 9.4 Observability

A `run_manifest` table (run_id, git_sha, source, rows in/out, bytes scanned, status, duration) plus
persisted dbt `run_results.json` and `manifest.json` per run. That delivers run history and lineage
with zero infrastructure. `dbt docs generate` published to GitHub Pages gives a hosted lineage graph —
the highest showcase value per unit of effort in the project.

### 9.5 Runbooks

`docs/runbooks/` for failures that will actually occur: a metric leaving the free tier, a DefiLlama
protocol rename or methodology change, CoinGecko 429s, BigQuery budget exceeded, an asset rebrand,
a chain reorg affecting UTXO metrics.

---

## 10. Method registry

### 10.1 The contract

Every valuation method is a plugin with one uniform signature:

```
estimate(asset, panel, params) -> {
    fair_value,          # point estimate, USD per unit (or None if not applicable)
    band_low, band_high, # uncertainty band where the method admits one
    inputs_used,         # dict of input values, for provenance
    diagnostics,         # fit statistics, assumption violations, warnings
    applicable,          # bool + reason if False
}
```

Methods are pure functions over the panel. Adding one is a module plus a YAML config plus a
`dim_method` row. Because every method reads the same conformed numbers, estimates are directly
comparable — and the comparison *is* the research output. Methods declare their data dependencies,
so a method whose inputs are unavailable (e.g. Cointime before the §6.3 backfill) returns
`applicable=False` with a reason rather than a fabricated number.

**Applicability is computed, not hand-written** (added 2026-10-12). A method declares the columns it
needs and a minimum coverage density; the registry resolves `applicable` against the panel and the
§5 gap table, so TRX's missing MVRV, AVAX's missing Coin Metrics volume and BNB's 19%-dense market
cap each produce one declared refusal with a reason instead of per-method special cases. The
`consensus` field already in `config/assets.yml` predicts the proof-of-work methods exactly: hash
rate exists on the free tier for BTC, LTC, BCH and DOGE, plus ETH to the Merge at 63.7% density.

**Volume discipline, the §10.4 denominator rule applied to `T`** (added 2026-10-12). Coin Metrics'
reported spot volume and CoinGecko's total volume are different series with different exchange
coverage — measured 2.97e9 against 1.48e10 for BTC on the same day, a 5× gap. Velocity, NVT and
Price-to-Utility all turn on `T`, so the two are carried as distinct columns, never coalesced; each
method names which it used in `inputs_used`, and the headline results report both as a sensitivity.
A method that silently picked one would make its own fair value unreproducible.

Alternatives rejected: per-method vertical slices (duplicated IO; methods that fetched different
data on different days cannot be honestly compared) and SQL-only in dbt (DCF sensitivity grids,
log-normal velocity fitting and LPPLS fits are unreasonable in SQL).

### 10.2 Family A — Monetary / equation of exchange (income)

| Method | Specification |
|---|---|
| **Crypto QTM** (dissertation §3.1) | `price = Q / (M* × f × V)`, with `M = M* × f` for the circulating fraction |
| **Kampakis log-linear EoE** (2024) | `log(price) = β₁·log(T) + β₂·log(M) + β₃·log(1/V)`; published fit β = (0.88, 0.84, 1.15), adjusted R² 0.97. We re-estimate rather than adopt the coefficients. `V = T/MC` daily, `H = MC/T` in days |
| **Holding-time correction** | Fit the velocity distribution per asset (the paper finds log-normal dominant), then derive mean holding time by simulation rather than assuming `H = 1/V`. The paper's central result is that these differ at the mean, which invalidates the inversion used in the dissertation's QTM and P/U sections |
| **Crypto J-curve** (Burniske) | `CUV = network transaction value / (circulating supply × velocity)`; `DEUV = price − CUV`; track the CUV/DEUV mix over an asset's life |

**Ambiguity resolved:** the dissertation renders CUV as "token's current supply / (token supply ×
token velocity)", which is dimensionally incoherent as printed (it yields a dimensionless ratio, not
a price). We implement the economically intended form above — utility value per token as network
transaction value divided by effective circulating supply — and record the deviation in the method
card.

**Known caveat to carry into the report:** the EoE requires the asset to function as a medium of
exchange. For assets held predominantly as stores of value the method is weakly identified. USDC
serves as the control case, exactly as USDT does in Kampakis et al.

### 10.3 Family B — Network / adoption (income)

| Method | Specification |
|---|---|
| **Metcalfe, estimated** | `log(market_cap) = α + β·log(AdrActCnt)`, reporting β with confidence intervals. Published estimates give β ≈ 1.69, not the theoretical 2.0, so the exponent is **estimated, never assumed** |
| **Peterson generalised Metcalfe** | `V = a × (n(n−1)/2 × 1/bₜ)` as specified in the dissertation §3.3 |
| **LPPLS bubble flag** | Log-periodic power-law singularity fit (Wheatley/Sornette) as a bubble-regime indicator accompanying the Metcalfe residual |
| **NVM ratio** | Network value to Metcalfe-implied value; the deviation series is the signal |

**Caveat:** an active address is not a user, and a high R² between two trending series is weak
evidence. Residual diagnostics and out-of-sample deviation are reported alongside the fit.

### 10.4 Family C — Cash flow / DCF (income) — new chapter

| Method | Specification |
|---|---|
| **ETH three-stage DCF** | Cash flow proxy = burned fees (treated as buyback) + validator fee revenue − issuance (treated as stock-based compensation). Three-stage growth; discount rate decomposed as risk-free + explicit crypto risk premium; output is a sensitivity grid over (growth, discount rate, terminal multiple), not a single number |
| **DeFi net-revenue DCF** | Per-protocol net revenue (the share the protocol retains, **not** gross fees routed to liquidity providers), discounted; per-token value divides by supply grown net of emissions and burns |
| **Emissions-adjusted real yield** | `real_yield = fee share + incentive share − dilution from emissions at current price`. A nominally high yield funded by emissions is not a yield |
| **dFDV** | Market cap blended with a haircut on locked supply, as a less misleading alternative to naive FDV, which assumes full dilution at today's price |
| **Bond-style target yield** | `value = annual fees / target yield`, with `target yield = 10Y treasury + staking risk premium`, both as explicit parameters |
| **Counter-model: supply-effect-only** | The standing critique holds that staking yield and fee burn affect the *supply path*, not network value, so DCF arithmetic does not transfer to L1 tokens. Implemented as a sibling method so the disagreement between the two is **measured** rather than argued. Where they diverge is itself a finding |

**Denominator discipline.** The literature is consistent that the denominator decides the answer:
total fees, net protocol revenue, and value actually reaching token holders after dilution are three
different quantities. All three are computed and labelled distinctly; no multiple is published
without naming which denominator it used.

### 10.5 Family D — Relative multiples (market)

NVT ratio and NVT golden ratio (NVT over its 2-year moving average); the dissertation's
Price-to-Utility ratio with `token_utility = (velocity × staking_ratio) / dilution_rate`, now using
the corrected holding time from §10.2; P/F, P/S, FDV/revenue, dFDV/revenue, market-cap-to-net-assets,
P/TVL; and sector-cohorted z-scores so comparables are drawn within a cohort rather than across
incomparable projects.

**Caveat carried from the dissertation:** NVT assumes value derives primarily from medium-of-exchange
use and is blind to store-of-value demand; Woo's own position is that it is retrospective, not
predictive. The P/U ratio was designed to relieve this but inherits the heterogeneity problem. Both
are reported as diagnostics, not fair values, and the method cards say so.

### 10.6 Family E — Cost approach

**(a) Available from free data now**

| Method | Specification |
|---|---|
| **Hayes cost of production** | `E_day = electricity cost × hours/day × hash rate × energy efficiency`; `BTC/day = (block reward × hash rate × 3600) / (difficulty × 2^32) × 24`; `p* = E_day / (BTC/day)`. Electricity price and ASIC efficiency are explicit parameters with a modern efficiency curve, not 2015 constants |
| **Thermocap** | `cumsum(IssTotUSD + FeeTotNtv × PriceUSD)` |
| **MVTV** | `market_cap / thermocap` — the crypto analogue of EV/EBITDA |
| **Investor capitalisation** | `realized_cap − thermocap` |
| **MVRV / MVRV Z-score** | `CapMVRVCur` direct; `Z = (market_cap − realized_cap) / stdev(market_cap)` |
| **Puell multiple** | `IssTotUSD / MA₃₆₅(IssTotUSD)` |
| **Mayer multiple** | `price / MA₂₀₀(price)` |
| **Energy value** | Capriole-style energy-based fair value |

**Applicability constraint:** cost-of-production is native to proof-of-work and does not transfer to
proof-of-stake, where electricity is not a relevant input. PoS assets return `applicable=False` with
that reason rather than a meaningless number — a limitation the dissertation itself identified.

**(b) Requires the §6.3 UTXO backfill — deferred, free-tier staged**

Cointime primitives (coinblocks created/destroyed, Liveliness = `cumsum(destroyed)/cumsum(created)`,
Vaultedness = `1 − Liveliness`, active and vaulted supply), Cointime-adjusted MVRV, Cointime
stock-to-flow, AVIV; HODL waves and therefore the dissertation's SLRV
(`1d wave / 6m-1y wave`, realized-cap weighted), STHPL and RPV ratios; reserve risk.

Because these are cumulative from genesis (§6.3), the family is **all-or-nothing**: every method here
returns `applicable=False` with the reason "awaiting staged UTXO backfill" until the extraction is
complete. No approximations, no partial-history values, and no substitution of a plain-MVRV proxy
dressed up as a Cointime metric — a wrong number here would silently contaminate the §10.8 composite
and the §11 model scores.

### 10.7 Family F — Falsified and contested benchmark shelf

| Method | Treatment |
|---|---|
| **Stock-to-flow** | `ln(market_cap) = α + β·ln(stock/flow)`. Fitted and reported **as a model that broke after 2021** |
| **Power law** | `ln(price) = α + β·ln(days since genesis)`. The most-cited long-horizon model and the most contested — overfitting objections, and a 2026 paper listing nine methodological concerns against the claimed mechanistic derivation. Reported with those objections attached |

Including these visibly, labelled as failed or contested, is more honest than omitting them, and
gives the predictive layer a deliberately weak benchmark to beat.

### 10.8 Family G — Composite

A method-agreement panel: cross-method dispersion per asset-date, a football-field view of the fair
value range, and a weighted composite where weights derive from each method's **out-of-sample**
track record from §11, not from judgement.

---

## 11. Statistical and validation layer

This replaces the dissertation's Chapters 7-8. The ordering is deliberate: baselines before models.

### 11.1 Baselines, established first

Naive persistence (random walk), random walk with drift, and historical mean. Every model is scored
against these. Published evidence suggests full-history BTC is statistically indistinguishable from a
random walk, so a model that fails to beat persistence is reported as having failed to beat it.

### 11.2 Validation protocol

- **Target**: log returns at horizons h ∈ {1, 7, 30, 90, 365} days. **Never price levels** — a model
  predicting levels from levels is the dissertation's error restated.
- **Splitting**: expanding-window walk-forward, strictly time-ordered, with **purging and embargo**
  around each fold boundary so that overlapping horizon windows cannot leak across the split.
  No `train_test_split`, no shuffling, anywhere in the codebase.
- **Metrics**: MAE, RMSE, MASE (scaled against naive), directional accuracy.
- **Significance**: Diebold-Mariano test of each model against the baseline. A difference without a
  p-value is not reported as a difference.
- **Feature hygiene**: no feature may be a transformation of the target at or after the prediction
  timestamp; all features point-in-time via §8.2.

### 11.3 Models

The dissertation's three (OLS, SVR with RBF and polynomial kernels, feed-forward NN) are retained for
continuity, re-run correctly, plus ridge/elastic net, gradient boosting and an LSTM/GRU. The
expectation, stated in advance, is that most will not beat persistence at most horizons. That is a
result, not a failure.

### 11.4 Econometrics the dissertation lacked

| Test | Question it answers |
|---|---|
| ADF / KPSS | Are these series stationary? Determines what regressions are even valid |
| Engle-Granger / Johansen cointegration | **Does price maintain a long-run equilibrium with fundamentals?** This is the correct test for "does MVRV/NVT/Metcalfe anchor price" — far stronger than the correlation evidence in §8.4 |
| VECM | Short-run dynamics and the speed of mean reversion toward that equilibrium |
| GARCH(1,1) / EGARCH | Volatility clustering and asymmetry; feeds the seller-exhaustion analysis in the dissertation's §6.3 |
| Markov-switching | Regime identification — which directly tests the dissertation's finding that correlations behave differently at a bull-market peak |
| **DCC-GARCH** | Rolling conditional correlations, replacing five static Pearson windows. The dissertation's own conclusion — that the BTC relationship "shifts over time and resists confident prediction" — is precisely what DCC measures and static Pearson cannot |

### 11.5 Cross-sectional asset pricing

Liu-Tsyvinski-Wu three-factor model (market, size, momentum) plus a value factor built from active
addresses over market cap; Fama-MacBeth regressions; decile portfolios sorted on each valuation
signal. This answers the question the dissertation's correlation chapter could not: **does a
valuation signal actually earn returns?** Built on a point-in-time top-N universe (§5) rather than
the curated 23, with point-in-time sector labels (§8.3).

### 11.6 Multiple-testing discipline

This project evaluates many methods against many assets at many horizons. Some will look
significant by chance. The report therefore records the **total number of hypotheses tested** and
applies a family-wise correction to headline claims. Without this, the breadth of the method
registry becomes a data-snooping machine.

---

## 12. Deliverables

1. **Repository** — pipeline, contracts, tests, CI, Dockerfile, README with a one-command reproduce path.
2. **`reports/`** — the research study: method-by-method results; the cointegration and factor
   evidence; the honest verdict on what beats a baseline; and an explicit chapter
   **"2022 dissertation findings vs what holds in 2026"**, including a direct account of the
   Chapter 7-8 correction.
3. **`dashboard/`** — Streamlit: per-asset football-field fair-value range across methods, cycle
   indicator panel, method-agreement heatmap, live DCF sensitivity sliders, data-freshness banner.
4. **`docs/`** — one method card per method (formula, lineage, assumptions, failure modes, citation);
   runbooks; `dbt docs` lineage graph on GitHub Pages.

**Disclaimer requirement:** the report and dashboard both carry a prominent statement that outputs
are valuation estimates under stated assumptions, not investment advice, and that these assets are
volatile and these methods disagree by design.

---

## 13. Phasing

| Phase | Content | Exit criterion |
|---|---|---|
| **0 — Foundation** | Repo scaffold, config-as-code, DuckDB + dbt-core with dual targets, Dockerfile, CI, path resolver, byte-budget gate, three data contracts, one source end-to-end | `make all` runs green on a fixture sample in CI |
| **1 — Ingestion** | Coin Metrics, DefiLlama, CoinGecko, yfinance → bronze; SCD2 silver panel; star schema dimensions; trailing-window refetch; compaction; identity assertions | Full history loaded; every contract test passing; a point-in-time query reconstructs a past vintage |
| **2a — Remediation** *(added 2026-10-12)* | The §5.1 conformed key and vendor-id map; DefiLlama slug lists; quarantine; the §9.1-9.2 contract corrections; bronze `run_id`; dead column removal; re-backfill | `make validate` exits 0 on live data with every failure either fixed or quarantined with a reason; the panel joins all four sources on `symbol` |
| **2b — Reproduce the dissertation** | Families A, B, D, E(a) — QTM, J-curve, Metcalfe, NVT, P/U, Hayes, Thermocap, MVTV, MVRV, Investor Cap, **plus the holding-time correction** | Every dissertation method produces estimates with a method card; results compared against the figures in the PDF |
| **3a — Modern layer (free data)** | Family C (DCF and the counter-model), remaining multiples, Family F benchmark shelf | DCF sensitivity grids reproduce published reference valuations within a stated tolerance |
| **3b — UTXO extraction** *(background, spans months)* | BigQuery sandbox staged epoch-by-epoch behind the byte gate and monthly ledger, materialised to our own Parquet → Family E(b) Cointime, HODL waves, SLRV/STHPL/RPV | All epochs extracted; Cointime identities hold (`liveliness + vaultedness = 1`); family flips from `applicable=False` to live in one step |
| **4 — Statistical rebuild** | §11 in full: baselines, walk-forward, DM tests, econometrics, factor models, multiple-testing correction | Every model has a DM p-value against persistence; `fct_model_score` populated including baselines |
| **5 — Consumers** | Research report and Streamlit dashboard off the same gold layer | Every figure regenerates from `make all`; dashboard served locally |
| **6 — Cloud** | Tier 0: GitHub Actions cron → object storage → Streamlit Community Cloud, $0. Tier 1: Cloud Run Job + Cloud Scheduler + GCS, a few dollars monthly, no sleeping or inactivity disabling. Tier 2 (optional): Dagster + OpenLineage | A scheduled run completes unattended and the dashboard shows T+1 data |

Phases 2 and 3a are deliberately ordered so the dissertation's methods form a **verified baseline**
before new methods are layered on. If a reproduction disagrees with the PDF, that is diagnosed before
anything is built on top of it.

**Why 2a exists** (added 2026-10-12). Each item in it was found by querying the panel Plan 1
produced, and each one silently corrupts a method rather than failing it: a TRX velocity computed
from a dead wrapper, a UNI DCF against an unmapped protocol, a mean taken over a 19%-dense column.
A wrong number that looks right is far more expensive to find in Phase 4's backtests than to fix
now, so remediation precedes the method registry rather than trailing it.

**Why the holding-time correction sits in 2b rather than 3a.** It is formally a modern addition, but
§10.2's finding is that mean holding time and `1/V` differ at the mean, which invalidates the
inversion the dissertation's own QTM and P/U sections rely on. Reproducing those two methods without
it would reproduce a known error and call it a baseline.

### Deployment tier notes

- **Tier 0 caveats to document**: GitHub disables scheduled workflows after ~60 days of repository
  inactivity; cron minimum interval is 5 minutes and runs may be delayed under load; Streamlit
  Community Cloud apps sleep after inactivity and show a wake-up screen. Acceptable for a research
  artifact or portfolio piece.
- **Tier 1** is the recommendation once the output is relied upon.

---

## 14. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Derived realized cap breaks** if Coin Metrics redefines either input | High — silently corrupts the whole cost family | Identity assertions as blocking tests (§9.2); free-tier availability check |
| **BigQuery scan volume** unknown until dry-run | Low (cost), Medium (schedule) | Free-only decision (§6.3): sandbox with no billing account, dry-run byte gate, monthly byte ledger. Cost risk is structurally contained; the residual risk is that staging takes more months than hoped |
| **Phase 3b slips indefinitely** if the monthly envelope is tighter than expected | Medium | Family E(b) is isolated behind `applicable=False`, so every other deliverable ships without it; the full-node alternative (§6.3) is documented if the timeline becomes unacceptable |
| **Sandbox 60-day table expiry** silently destroys intermediate work | Medium | BigQuery is compute-only by design; every run exports aggregates to our own Parquet before completing |
| **Free tiers shrink** (at least one provider retired its free tier in 2026) | Medium | Contracts fail loudly; bronze retains everything already fetched, so history survives a source loss |
| **Method circularity** — MVRV and realized cap both derive from market cap, so price appears on both sides of some "valuation" ratios | High for interpretation | Method cards state it explicitly; such metrics are labelled diagnostics, not independent fair values; cointegration tests are run on genuinely independent fundamentals |
| **Data snooping** across many methods × assets × horizons | High | §11.6 — count hypotheses, apply family-wise correction to headline claims |
| **Survivorship bias** in a 2026-chosen asset list | Medium | Point-in-time universe for the factor work; limitation stated in the report |
| **DeFi methodology drift** silently changes revenue history | Medium | Methodology hash + volume anomaly threshold (§9.2) |
| **Over-engineering** the pipeline relative to the data | Medium | §7.1 decision record; streaming and big-data tooling explicitly ruled out |
| **SOL absent** from the free tier | Low | Documented gap; AVAX/ALGO substitute; excluded rather than null-carried |

---

## 15. Open questions for review

**Resolved 2026-10-11:** BigQuery backfill — free only, staged inside the sandbox’s monthly free
terabyte. No payment, now or later, without an explicit new decision. See §6.3.

**Resolved 2026-10-12 — Q1, equity set:** keep all five tech stocks *and* add gold (`GC=F`) and the
dollar index (`DX-Y.NYB`). The five are not negotiable, because the "2022 findings vs 2026" chapter
needs them for comparability; the two macro series are two more tickers through an adapter that
already works, and the macro-correlation question is the more interesting one now. Recorded in §5.

**Resolved 2026-10-12 — Q2, report format:** Quarto. It executes Python inline, which makes §12's
"every figure regenerates from `make all`" structurally true instead of a discipline to maintain; it
renders both HTML and PDF, so the report can sit beside the dissertation and next to the dashboard;
and it handles cross-references and citations natively. Markdown plus figures cannot guarantee
regeneration, and LaTeX alone forfeits the HTML path.

**No open questions remain.** New ones are recorded here with their resolution date as they arise.

---

## 16. References

**Source dissertation**
- Arafat, Y. (2022) *A Critical Analysis of Cryptoassets Valuation Method*. MSc FinTech, University of Liverpool.

**Post-2022 methods**
- CFA Institute (2023) *Valuation of Cryptoassets: A Guide for Investment Professionals*. https://www.cfainstitute.org/about/press-releases/2023/cryptoassets-valuation-guide-2023
- Kampakis, S., Yuan, M., Ikpobe, O.P., Stankevicius, L. (2024) *Improving the Equation of Exchange for Cryptoasset Valuation Using Empirical Data*. arXiv:2403.04914. https://arxiv.org/pdf/2403.04914
- Puell, D. (ARK Invest) and Check, J. (Glassnode) — *Cointime Economics*. https://studio.glassnode.com/charts/cointime-primitives-liveliness-vaultedness
- 21Shares (2025) *Primer on Crypto Asset Valuations, Q1 2025*. https://www.21shares.com/en-us/insights/primer-on-crypto-asset-valuations-q1-2025
- CoinShares — *Ethereum 5-Year Valuation Framework*. https://coinshares.com/us/insights/research-data/ethereum-5-year-valuation-framework/
- *Why DCFs are illogical for valuing layer 1s* (the §10.4 counter-model). https://omnida.substack.com/p/why-dcfs-are-illogical-for-valuing
- *DeFi versus TradFi: Valuation Using Multiples and Discounted Cash Flows*. arXiv:2210.16846. https://arxiv.org/pdf/2210.16846
- TokenIntel — value-accrual and emissions-adjusted real yield framework. https://tokenintel.org/reports/value-accrual-five-questions-framework-2026-05-23
- Wheatley, S., Sornette, D. et al. *Are Bitcoin Bubbles Predictable? Combining a Generalized Metcalfe's Law and the LPPLS Model*. arXiv:1803.05663. https://arxiv.org/pdf/1803.05663
- Liu, Y., Tsyvinski, A., Wu, X. *Common Risk Factors in Cryptocurrency*. https://economics.yale.edu/research/common-risk-factors-cryptocurrency
- Uras, N. et al. *Forecasting Bitcoin closing price series using linear regression and neural networks models* (random-walk indistinguishability). arXiv:2001.01127
- Power-law critique (2026), nine methodological concerns. https://zenodo.org/records/19975701

**Data sources**
- Coin Metrics Community API — https://community-api.coinmetrics.io/v4
- DefiLlama API — https://api.llama.fi
- CoinGecko API — https://api.coingecko.com/api/v3
- BigQuery public datasets — https://docs.cloud.google.com/bigquery/public-data
- BigQuery sandbox (no credit card, no billing account; 1 TB/month; 60-day table expiry; no DML) — https://cloud.google.com/bigquery/docs/sandbox

**Engineering patterns**
- `senior-data-engineer` skill: `references/data_modeling_patterns.md` (SCD2, incremental models, partitioning), `references/dataops_best_practices.md` (data contracts, CI/CD, observability, runbooks).
