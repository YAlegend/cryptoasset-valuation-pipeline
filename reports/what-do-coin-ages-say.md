# What do coin ages say? — Phase 3b

**These are valuation estimates under stated assumptions, not investment advice. These assets are
volatile and these methods disagree by design.**

Run of 2026-10-18 against `data/cryptoval.duckdb`, panel vintage `2026-10-18 16:44:17+01:00`.
Coin-age data: one BigQuery sandbox extraction (run `cf502431`), spend dates 2009-01-03 to
2026-10-16. Derived table: `fct_btc_cointime_daily`, 6,496 days. Estimates: `fct_valuation_estimate`,
six methods × 4,309 BTC days. Every number below is a query against those tables.

**The result, in one sentence: the coin-age data extracted from the chain reproduces Coin Metrics'
supply to a median of 0.0004% and its realized cap to a median of 0.014%, so Family E(b) is now
live; on that data, the dissertation's SLRV and STHPL readings line up with the 2018 and 2022 bear
markets and the December 2017 top as it says, but RPV never reaches its "top" threshold of 2% and
reaches its "bottom" threshold of 0.001% on one day only.**

**Descriptive only.** This chapter adds no hypothesis family and no multiple-testing correction. It
reports each signal's history against thresholds the dissertation stated. It does not test whether
those thresholds predict anything; the Phase 4 out-of-sample null
(`reports/does-anything-beat-a-random-walk.md`) is the standing answer to that question for every
method here.

---

## 1. What was extracted, and what it cost

Spec §6.3 kept these metrics out until a free source of UTXO ages existed. They now come from
BigQuery's public `crypto_bitcoin` dataset, read through a sandbox project with no billing account.
Three queries return everything the metrics need. Nothing about individual outputs is downloaded,
and nothing about unspent outputs is queried at all.

| Query | Rows returned | Bytes billed |
|---|---|---|
| Spent value by (creation day, spend day), with coinblocks destroyed | 9,248,744 | 466.0 GB |
| Value created per day | 6,491 | 95.4 GB |
| Block heights and their dates | 971,783 | 0.02 GB |
| **Total** | | **561.4 GB** (of the 0.9 TB monthly gate) |

- **Cost against estimate:** the bill came in under the dry run's estimate (561.5 GB).
- **Unspent value:** for each creation day c and each day d,
  `unspent(c, d) = created(c) − Σ spent(c, s ≤ d)`.
- **Coinblocks created:** these come from the subsidy schedule and the block heights, so
  Liveliness is exact.
- **Coin prices:** realized-cap weighting uses Coin Metrics' daily BTC close back to 2010-07-18,
  fetched over the free API. Coins created before that carry price 0.
- **Refresh:** a monthly `make utxo` adds the next spend-date range. Its dry run is about 113 GB.

## 2. Is the extraction right?

Coin Metrics computes BTC supply and realized cap from the same chain, independently. Both series
are checked in `make validate`, against tolerances set before the data was seen.

| Series | Days compared | Median gap | 95th percentile | Worst | Tolerance |
|---|---|---|---|---|---|
| Supply (from 2011) | 5,768 | 0.0004% | 0.0027% | 0.080% (2014-11-06) | 0.5% |
| Realized cap (from 2012) | 5,403 | 0.014% | 0.042% | 1.42% (2013-12-24) | 5% |

- **Realized cap:** the worst days fall in the 2013 bubble, when a single day's price range was
  wide enough that "the daily close of the creation day" is a coarse cost basis.
- **Today:** on 2026-10-16 we have $1,082.6bn against Coin Metrics' $1,082.7bn.
- **`liveliness + vaultedness = 1`** holds on every day.

## 3. Liveliness and HODL waves

| Year end | 2011 | 2013 | 2015 | 2017 | 2019 | 2021 | 2023 | 2025 | 2026-10-16 |
|---|---|---|---|---|---|---|---|---|---|
| Liveliness | 0.383 | 0.493 | 0.477 | 0.603 | 0.608 | 0.617 | 0.593 | 0.641 | 0.634 |

Liveliness jumps in the two bull markets that spent old coins (2013 and 2017). It has drifted
between 0.59 and 0.64 since, and peaked at 0.641 on 2025-12-20. Today:

- **Active supply:** 12.74M BTC.
- **Vaulted supply:** 7.35M BTC.
- **Ageing supply:** coins older than a year were 37.6% of supply at the end of 2013, 42.8% at the
  end of 2017, 56.9% at the end of 2021 and 63.5% today.
- **Ten-year-old coins:** coins untouched for ten years or more are 17.9% of supply but 0.04% of
  realized cap. They were bought for almost nothing.

## 4. The dissertation's three signals

| Signal | Dissertation's reading (§5.5, §6.1, §6.2) | Today | Over 2015-01-01 to 2026-10-16 |
|---|---|---|---|
| **SLRV** (1-day ÷ 6m–1y realized-cap HODL wave) | Below 0.04 in bear markets | 0.057 | Below 0.04 on 611 days: 181 in 2018, 95 in 2022, 92 in 2015 and 133 in 2026. Peak 5.42 on 2017-12-07 |
| **RPV** (realized profit ÷ market cap) | ≥ 2% at tops, ≤ 0.001% at bottoms | 0.023% | Never reaches 2%; maximum 1.68% on 2017-12-07. At 0.001% on one day (2015) |
| **STHPL** (price ÷ realized price of coins < 155 days) | Unity separates gain from loss; below unity, oversold | 1.094 | Below 1 on 42% of days, concentrated in 2018 (326), 2022 (356) and 2015 (229). 135 crossings of 1. Peak 2.35 on 2017-12-07, trough 0.52 on 2015-01-14 |

- **SLRV and STHPL** behave as the dissertation describes.
  - Both peak on 2017-12-07, ten days before that cycle's price top.
  - Both sit in their "oversold" zones through the 2018 and 2022 bear markets.
  - The 133 SLRV-below-0.04 days in 2026 and STHPL's 218 below-unity days in 2026 put this year
    in the same state.
- **RPV** does not reproduce its thresholds on this construction.
  - Its highest reading is 1.68%, and the second and third highest come at the December 2018
    bottom (1.16%) and in July 2019 (1.09%), not at tops.
  - Two reasons are on the method card. Profit here is measured between daily closes. And every
    spend counts as a realisation, including exchange reshuffles and change outputs, which inflate
    "profit" in capitulation as well as in euphoria.
  - The dissertation's 2% and 0.001% thresholds presumably come from a construction that filters
    those spends. This one does not.

STHPL is computed as short-term-holder MVRV (Plan 3b Ruling 3). The dissertation's literal ratio
of short-term to long-term profit is undefined whenever long-term profit crosses zero. Its own
reading of the number ("unity marks breakeven") is this one's.

## 5. Cointime prices

| Price | 2026-10-16 | BTC price ÷ it, today | BTC below it |
|---|---|---|---|
| Active price (realized cap ÷ active supply) | $84,948 | 0.960 | 53% of days since 2015; 254 days in 2026 |
| True market mean (investor cap ÷ active supply) | $77,239 | 1.056 | 41% of days since 2015; 185 days in 2026 |

BTC closed at $81,538 on 2026-10-16, which is between the two. AVIV, the ratio of active cap to
investor cap, is the same number as price ÷ true market mean, by construction, not a second finding
(market cap × Liveliness ÷ investor cap = price × active supply ÷ investor cap). It is 1.056 today,
against a median of 1.095 since 2015, and ranged from 0.28 (2015-01-14) to 2.97 (2017-12-07).

Both prices read Coin Metrics' realized cap, so both carry the MVRV circularity flag (spec §6.2).
They are a cost basis, not an independent valuation.

## 6. What is absent

- **Cointime stock-to-flow and reserve risk** (§10.6(b)): each has several published variants and
  no canonical formula, and the dissertation computes neither. They are left out rather than
  approximated.
- **Other UTXO chains:** LTC, DOGE and BCH are in BigQuery too. Every E(b) method refuses them
  with the measurement "0.0% dense".
- **BTC on 2026-10-17 and 2026-10-18:** these days are past the extraction cutoff, so each method
  refuses them. The cutoff is today minus two days, because a just-mined day can still reorg.
- **Glassnode comparability:** the "1-day wave" is coins created on the day and still unspent at
  its end (Plan 3b Ruling 2). SLRV's level is not directly comparable with Glassnode's 24-hour band.
