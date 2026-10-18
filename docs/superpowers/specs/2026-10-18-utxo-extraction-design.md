# UTXO Extraction and Family E(b) — Design (Plan 3b)

Parent spec: `2026-10-11-cryptoasset-valuation-pipeline-design.md` §6.3, §10.6(b), §13 phase 3b.
Status: written 2026-10-18 without a review round, at the user's standing instruction to proceed
without check-ins. Every judgement call is listed under **Rulings** so it can be overturned.

## 1. What this delivers

Bitcoin's coin-age data, extracted once from BigQuery's public `crypto_bitcoin` dataset inside the
free sandbox. From that data the plan derives:

- the Cointime primitives (Liveliness, Vaultedness and active/vaulted supply);
- HODL waves;
- the dissertation's SLRV, RPV and STHPL;
- Cointime's AVIV, active price and true market mean.

These turn Family E(b) from `applicable=False` to live in one step (§13 exit criterion).

## 2. Measured facts this design rests on (dry runs, 2026-10-18, project `cryptoval-sandbox-3141`)

| Query | Bytes billed (upper bound) |
|---|---|
| `outputs` (hash, index, block, ts, value) | 421.6 GB |
| `inputs` (spent hash, spent index, block, ts) | 318.1 GB |
| `transactions` (hash, block, ts) | 120.0 GB |
| Spend matrix via `outputs ⋈ inputs` | 739.7 GB |
| **Spend matrix via `inputs ⋈ transactions`** | **466.3 GB** |
| Same, one spend-month (`i.block_timestamp >= '2026-09-01'`) | 113.3 GB |
| Created-by-day from `outputs` (ts, value) | 95.5 GB |
| Blocks per day | ~0 GB |

- **Data freshness:** the dataset is current to the tip (block 972,037 at 18:35 UTC that day).
- **Views:** `inputs`, `outputs`, `transactions` and `blocks` are views.
- **Pruning:** a `block_timestamp` predicate prunes partitions through the views (318 GB → 5 GB
  for one month of `inputs`).
- **Spend values:** `inputs.value` is the value of the output being spent.

So the full backfill costs about 562 GB, which fits inside one month's 0.9 TB gate. §6.3's
"staged across months" becomes "staged by spend-date range, and one range happens to fit". The
staging machinery is kept, because the monthly refresh is the same operation over a short range.

## 3. The sufficient statistic: the creation × spend matrix

All the metrics above are functions of three small tables, extracted per spend-date range `[s0, s1]`:

| Bronze table | Grain | Columns | Source |
|---|---|---|---|
| `btc_utxo_spent` | (created_date, spent_date) | `value_sat` INT64, `coinblocks_destroyed` FLOAT64 (BTC·blocks), `n_inputs` | `inputs ⋈ transactions` on `spent_transaction_hash = hash`, spent_date in range |
| `btc_utxo_created` | created_date | `value_sat`, `n_outputs` | `outputs`, created_date in range |
| `btc_blocks_daily` | date | `n_blocks`, `min_height`, `max_height` | `blocks`, date in range |

- **Bounds:** `value_sat` sums stay below 2.1e15 and fit INT64. `coinblocks_destroyed` can reach
  about 1e21 sat·blocks, so it is divided by 1e8 and cast to FLOAT64 inside BigQuery.
- **Coinbase inputs:** these have no `spent_transaction_hash` and drop out of the inner join. That
  is correct: they spend nothing.
- **Unspent value:** `unspent(c, d) = created(c) − Σ_{s ≤ d} spent(c, s)`. Nothing about unspent
  outputs is ever queried.
- **Coinblocks created:** this comes from the block heights alone. Every existing coin earns one
  coinblock per block, so `created_coinblocks(d) = Σ_{h in day d} supply(h)`. Here `supply(h)` is
  the protocol subsidy schedule summed to height h (50 BTC halving every 210,000 blocks), which is
  deterministic. That makes Liveliness exact rather than a daily approximation.

## 4. Extraction (cost control is §6.3's, unchanged in force)

`cryptoval/utxo/` package, `cryptoval utxo extract` CLI, `make utxo`. The target is NOT in
`make all`, because it spends budget.

1. **Range:** `cutoff = today_utc − 2 days`, because the dataset lags the tip and a just-mined day
   can still reorg. The first run covers `[2009-01-03, cutoff]`. Each later run covers
   `[last_cutoff + 1, cutoff]`, read from the checkpoint. A run with nothing new exits 0 and says so.
2. **Each query:**
   - It dry-runs first, and the estimate goes through the existing `ByteBudgetGate.guard`
     (0.9 TB/month, ledger in `run_manifest`).
   - If the gate refuses, it raises `BudgetExceeded` and nothing executes.
   - A query that passes is run, and `manifest.finish(bytes_scanned=<actual billed bytes>)`
     records the cost.
3. **Writing:**
   - Each query's result is written to bronze, partitioned by range (`write_bronze`, source
     `btc_utxo_*`, with `request_params` carrying the range and the SQL hash).
   - A range counts as complete only when all three tables are written. The checkpoint is the set
     of complete ranges, so an interrupted run redoes only its own range.
   - Overlapping ranges are refused.
4. **Download:** results come down through the REST `tabledata.list` path
   (`create_bqstorage_client=False`). The Storage Read API is a separately priced product and is
   not used.
5. **Free-only, outer guard:** a client built against any project other than the configured
   `bigquery_project` is refused. The project is named in `config/settings.yml`, never inherited
   from the gcloud default (this machine's default project is a different, possibly billed one).

## 5. Derivation (local, free, in `make all`)

`cryptoval utxo derive` reads all bronze ranges and refuses unless together they are contiguous
from 2009-01-03, which is §6.3's all-or-nothing rule. It writes silver `btc_cointime_daily`, one
row per date:

| Column | Definition |
|---|---|
| `coinblocks_created`, `coinblocks_destroyed` | §3 |
| `liveliness` | `cumsum(destroyed) / cumsum(created)` |
| `vaultedness` | `1 − liveliness` |
| `utxo_supply` | `Σ_c unspent(c, d)` |
| `active_supply`, `vaulted_supply` | `utxo_supply × liveliness`, `utxo_supply × vaultedness` |
| `utxo_realized_cap` | `Σ_c unspent(c, d) · price(c)` |
| `hodl_<band>` / `hodl_rc_<band>` | share of supply / of realized cap with age in band (12 bands: 0d, 1d–1w, 1w–1m, 1–3m, 3–6m, 6m–1y, 1–2y, 2–3y, 3–5y, 5–7y, 7–10y, 10y+) |
| `sth_realized_cap` | realized cap of coins aged < 155 days |
| `sth_supply` | supply of coins aged < 155 days |
| `realized_profit_usd` | `Σ_c spent(c, d) · max(price(d) − price(c), 0)` |

- **Price:** `price(c)` is the panel's BTC `price_usd`, Coin Metrics' daily close.
- **Pre-price coins:** coins created before the first price (2009-01-03 to 2010-07-17) carry
  price 0, which is the realized-cap convention.
- **Computation:**
  - The computation is dense numpy over the matrix. The matrix is about 6,500 × 6,500 float64,
    roughly 340 MB, and is built per value column.
  - The matrix is cumsummed along the spend axis to get unspent value.
  - Age bands are diagonal slices.

**Identity checks** are wired into `make validate`:
- `liveliness + vaultedness = 1`, using the existing `check_cointime_complement`.
- `utxo_supply` against Coin Metrics `SplyCur`, with relative error ≤ 0.5% on every date from
  2011. The two differ by unspendable outputs and lost coinbase.
- `utxo_realized_cap` against Coin Metrics `realized_cap`, with relative error ≤ 5% on every date
  from 2012.
- These two are this plan's correctness proof: an independent vendor computes both series from the
  same chain. The thresholds are a first guess and are reported, not tuned. If live data breaches
  them, that is a finding, diagnosed before the family ships.

## 6. Panel and methods

- **Panel:** `stg_btc_cointime` is joined into `fct_asset_metrics_daily` on (symbol = 'BTC',
  metric_date). Every other asset gets NULLs.
- **Methods:** one module, `cryptoval/methods/cointime.py`, family `E_cost`. Every method is a
  signal (`returns_fair_value: false`, value in `diagnostics`) unless marked as a price.

| method_id | Value | Source of definition |
|---|---|---|
| `slrv` | `hodl_rc_0d / hodl_rc_6m_1y` | Dissertation §5.5 |
| `rpv` | `realized_profit_usd / market_cap_usd` | Dissertation §6.1 |
| `sthpl` | `price_usd / (sth_realized_cap / sth_supply)` | Dissertation §6.2 (see Ruling 3) |
| `aviv` | `(market_cap_usd × liveliness) / (realized_cap − thermocap)` | Cointime Economics |
| `active_price` (price) | `realized_cap / active_supply` | Cointime Economics |
| `true_market_mean` (price) | `(realized_cap − thermocap) / active_supply` | Cointime Economics |

`realized_cap` and `thermocap` are the panel's existing Coin Metrics columns, so all six methods
carry the MVRV circularity flag (§6.2). On any asset other than BTC, every method refuses with
"UTXO coin-age data is extracted for BTC only".

## 7. Report

`reports/what-do-coin-ages-say.md`, written from a live run. It covers:
- the identity checks against Coin Metrics, with numbers;
- each signal's history against the dissertation's stated thresholds (SLRV < 0.04 in bear
  markets; RPV ≥ 2% at tops and ≤ 0.001% at bottoms; STHPL unity crossings).

The chapter is descriptive only: no new hypothesis family, no Holm correction, and it says so.

## 8. Testing

- **Extraction:**
  - A fake BigQuery client (`dry_run(sql) -> bytes`, `run(sql) -> (DataFrame, billed)`) is
    injected into the extractor.
  - Tests cover the gate refusing, checkpoint resume, overlap refusal, the "nothing new" case,
    the wrong-project refusal, and manifest bytes.
  - SQL text is asserted for column pruning and range predicates.
- **Derivation:**
  - Each output column is checked against a hand-computed 4-day toy chain.
  - The subsidy schedule is checked at the 210,000 boundary.
  - Contiguity refusal is tested.
- **Methods:** standard estimate tests, and a non-BTC refusal test.
- **Live:** `make utxo` is run once by hand from this plan. It is never run in CI.

## 9. Rulings (decisions taken without a review round)

1. **Join through `transactions`, not `outputs`.** It is 37% cheaper and gives the same creation
   block, because inputs carry their own value.
   - BIP30's two duplicated coinbase hashes would fan a spend out to two transaction rows. So the
     transaction side is deduplicated to one row per hash, keeping the latest block, which is the
     copy that overwrote the first. Deduplication costs compute, not billed bytes.
   - Cost if wrong: two 50 BTC outputs carry a creation date about a day off.
2. **Daily grain, so the "1-day wave" is age 0 days.** Coins created on day d and unspent at its
   end stand in for Glassnode's 24 h band. Cost if wrong: SLRV's level shifts; the thresholds are
   read with that caveat.
3. **STHPL is short-term-holder MVRV at 155 days.** The dissertation's literal definition (STH P/L
   ÷ LTH P/L) is undefined when LTH P/L crosses zero. Its own reading, "unity is breakeven, below
   unity the cohort that moved coins in the last 155 days is at a loss", is exactly STH-MVRV's.
   Cost if wrong: the series is a different ratio from the one the dissertation charted.
4. **Out of scope:**
   - Cointime stock-to-flow and reserve risk, which §10.6(b) lists. Both have several published
     variants with no single canonical formula, and the dissertation computes neither. They are
     recorded as absent, not approximated.
   - Other UTXO chains (LTC, DOGE, BCH are in BigQuery too). Same for these: absent, not
     approximated.
5. **Thresholds of the identity checks:** 0.5% (supply) and 5% (realized cap) are guesses set
   before seeing data. The live numbers are reported whatever they are.
6. **Extraction is outside `make all`.** It spends a scarce monthly allowance, so it runs on
   purpose, not as a side effect.
