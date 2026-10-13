# 2022 dissertation findings vs what holds in 2026

**These are valuation estimates under stated assumptions, not investment advice. These assets are
volatile and these methods disagree by design.**

Run of 2026-10-13 against `data/cryptoval.duckdb`. Panel: `fct_asset_metrics_daily`, 69,239
symbol-days, 24 symbols, 2015-01-01 to 2026-10-13. Estimates: `fct_valuation_estimate`, 14 methods
over the same panel, written by `python -m cryptoval.cli estimate`. Every figure below is a query
against that table and is reproducible with `make cards && make estimate && make dbt`.

---

## 1. What the dissertation publishes — and what it does not

The plan for this chapter assumed a published figure per method to compare against. There is none,
and that is the first finding.

- The dissertation's empirical dataset is **Bitcoin alone**: daily Yahoo Finance OHLCV, 1 January
  2015 to 30 November 2022, seven attributes (§8.2). No on-chain series, no second asset.
- Its numeric results are a **price regression** (§8.3: linear, SVR, feed-forward NN at shifts of
  30/100/200/365 rows) reported qualitatively — "the linear regression model proves the more
  accurate of the three" — with no R² in the text, and **five Pearson correlation windows**
  (Tables 3-10).
- Chapters 3 to 6 — QTM, J-curve, Metcalfe, NVT, P/U, cost of production, MVRV, MVTV, investor
  capitalisation, SLRV — **state formulas and never compute them.** There is no fair value, no
  fitted β, no NVT level, no cost-per-coin anywhere in the document.

So this is not a reproduction of published numbers. It is the first execution of the
dissertation's own specifications, and the comparison it supports is formula-level: given the
formula as written, does it yield a defensible number on live data, and where does the formula
itself fail? The regression and correlation chapters — the only parts with published results — are
Plan 4's business, not this one's.

## 2. What reproduced

Twelve of the fourteen configured methods produce estimates. BTC on 2026-10-12, the last day with
a complete Coin Metrics panel (price $81,805.16):

| Method | Fair value | vs price | Circular |
|---|---|---|---|
| `crypto_qtm_panel_velocity` | $81,805.16 | 1.000 | **yes** |
| `mvrv_reversion` | $88,483.08 | 1.082 | **yes** |
| `mvtv_reversion` | $79,121.65 | 0.967 | **yes** |
| `nvt_reversion` | $56,332.00 | 0.689 | no |
| `realized_price` | $53,841.35 | 0.658 | **yes** |
| `hayes_cost_of_production` | $50,667.63 | 0.619 | no |
| `investor_cap_price` | $48,960.90 | 0.599 | **yes** |
| `crypto_qtm` | $33,172.57 | 0.406 | no |
| `j_curve` | $33,172.57 | 0.406 | no |
| `metcalfe_estimated` | $8,255.33 | 0.101 | no |
| `thermocap_floor` | $4,880.45 | 0.060 | no |
| `holding_time_correction` | (a correction factor, not a price) | — | **yes** |

Coverage across the 24-symbol universe:

| Method | Rows | Applicable | Symbols with estimates |
|---|---|---|---|
| `thermocap_floor` | 59,499 | 59,448 | 20 |
| `realized_price` / `investor_cap_price` | 57,525 | 57,315 | 20 |
| `metcalfe_estimated` | 57,537 | 53,928 | 20 |
| `mvrv_reversion` | 57,525 | 53,715 | 20 |
| `mvtv_reversion` | 30,473 | 28,833 | 9 |
| `hayes_cost_of_production` | 16,293 | 16,269 | 4 (BTC, BCH, DOGE, LTC) |
| `crypto_qtm` / `j_curve` | 7,289 | 7,279 / 7,285 | 20 |
| `crypto_qtm_panel_velocity` | 6,926 | 6,910 | 19 |
| `nvt_reversion` | 1,844 | 920 | 5 |
| `holding_time_correction` | 1,844 | 1,680 | 5 |
| `nvt_golden_ratio` | 24 | **0** | 0 |
| `price_to_utility` | 24 | **0** | 0 |

The spread is the result. On one day, for one asset, with every method applied as specified, the
range is $4,880 to $88,483 — a factor of 18. Any single number quoted from this family is a choice
of method, not a measurement of value.

## 3. What did not reproduce, and why

**(a) A formula that is wrong as printed.** The dissertation's CUV (§3.2) reads "token's current
supply / (token supply × token velocity)", which is dimensionally incoherent: a stock divided by a
rate, with no holding period. `j_curve` implements the economically intended form instead, which is
why its number coincides with `crypto_qtm` here. This is a correction, not a reproduction, and the
method card says so.

**(b) A formula that measures its own input.** With velocity taken from the panel — the
dissertation's own reading of §3.1, since the only velocity available is volume/market cap —
QTM collapses to `price / f`. See §5.

**(c) A relationship the dissertation relies on that does not hold.** §3.1 and §3.6 both invert
velocity to get holding time (`H = 1/V`). Measured per asset on the trailing 365 rows:

| Symbol | Naive `1/V` (days) | Log-normal mean (days) | Ratio | Anderson-Darling p | Log-normality |
|---|---|---|---|---|---|
| BTC | 37.97 | 51.98 | 1.369 | 3.4e-14 | **rejected** |
| ETH | 15.77 | 24.77 | 1.571 | 1.2e-16 | **rejected** |
| LTC | 13.40 | 15.48 | 1.155 | 9.2e-12 | **rejected** |
| BCH | 33.51 | 44.56 | 1.330 | 0.073 | not rejected |
| DOGE | 14.71 | 18.44 | 1.253 | 0.618 | not rejected |

The inversion understates mean holding time by 16% to 57%, so every QTM and P/U level that depends
on it is biased by that much. Note also that the closed form's own premise — log-normal velocity —
is rejected for three of the five assets, so for BTC, ETH and LTC the correction is itself only an
approximation. Both facts belong in the same sentence, and neither is in the dissertation.

**(d) A different denominator.** NVT as published is network value over *on-chain transaction
value*; the free tier serves *exchange volume*. The two volume series we do carry differ by a
median factor of **3.78x** on BTC over the last 365 days (range 0.04x to 16.22x) — the earlier
working estimate of "~5x" was a level comparison, and the measured median is 3.78x. NVT reversion
is invariant to a constant rescaling, so the level difference is harmless; the *shape* difference
is not. Every NVT row records `inputs_used.volume_column` for this reason.

**(e) A fit whose diagnostics invalidate it.** `metcalfe_estimated` returns β with a confidence
interval and Durbin-Watson on every row, and the verdict is uniform: **residuals are
autocorrelated for all 20 assets**, DW between 0.005 and 0.13 against a floor of 1.5. BTC's latest
β is 3.892 (CI 3.773-4.010, R² 0.490) — neither Metcalfe's 2.0 nor the published ≈1.69, and the CI
is meaningless at that DW. The method's fair value for BTC is 10% of price. Metcalfe's law on this
data is a regression of one trending series on another; the dissertation's §3.3 presentation of it
as a valuation tool does not survive its own diagnostics.

**(f) Market conditions, not method.** `mvrv_reversion` and `mvtv_reversion` both land within 8% of
price in October 2026 — but both are circular (§5), so that agreement is not evidence.

## 4. What cannot be reproduced at all, with the measurement that says so

Verbatim from `fct_valuation_estimate.reason`:

| Asset / method | Reason |
|---|---|
| TRX, `realized_price` | `realized_cap is 0.0% dense, below this method's floor 50.0%` |
| TRX, `mvrv_reversion` | `realized_cap is 0.0% dense, below this method's floor 50.0%` |
| BNB, `realized_price` | `realized_cap is 19.2% dense, below this method's floor 50.0%` |
| BNB, `metcalfe_estimated` | `market_cap_usd is 19.2% dense, below this method's floor 60.0%` |
| SOL, everything on-chain | `supply is 0.0% dense, below this method's floor 60.0%` |
| BTC, `nvt_golden_ratio` | `only 365 rows, below the method's minimum of 1095` |
| BTC, `price_to_utility` | `only 365 rows, below the method's minimum of 730` |
| BNB/POL/TRX, `nvt_reversion` | `only 364 rows, below the method's minimum of 365` |

Three structural limits, none of them fixable by better code:

1. **CoinGecko's keyless tier serves 365 days.** `volume_total_usd` and `velocity_total` begin
   2025-10-14. `nvt_golden_ratio` needs a 730-row moving average plus a year to use it, so it
   refuses for all 24 symbols and will keep refusing until a paid tier or a BigQuery extraction
   lands. `price_to_utility` needs 730 and refuses for the same reason — note that its null
   `staking_ratio`, the parameter the method card warns about, never even gets the chance to bite.
2. **A 365-row minimum against a 365-day window is a knife edge.** Most assets have 364 rows of
   volume and miss `nvt_reversion` and `holding_time_correction` by one row. Only BCH, BTC, DOGE,
   ETH and LTC clear it. Either threshold is defensible; what matters is that the refusal says
   which it was.
3. **The free tier is not uniform across assets.** TRX has no realized cap at all, BNB's Coin
   Metrics coverage is 19.2% of its history, SOL has none (CoinGecko only). Hash rate exists for
   exactly four proof-of-work assets, which is why Hayes runs on four and refuses the rest at
   `hash_rate is 0.0% dense`.

**A methodology note that affects every number above.** Density is now measured over the span in
which a method's required columns actually exist, not over the whole 2015-2026 history
(`covered_span`, `cryptoval/methods/base.py`). Without it a column that starts in 2025 reads as 8%
dense and every volume-based method refuses for every asset — seven of fourteen methods produced
nothing at all on the first run. The pre-coverage void is absence of history, not sparsity. Interior
gaps are still sparsity and still refuse.

## 5. The circularity audit

Six of the fourteen methods consume a column that is itself computed from market capitalisation,
and are flagged `is_circular` in the mart: `crypto_qtm_panel_velocity` (`velocity_total`),
`price_to_utility` (`velocity_total`), `realized_price` and `mvrv_reversion` (`realized_cap`),
`mvtv_reversion` (`mvtv`), `investor_cap_price` (`investor_cap_usd`), plus
`holding_time_correction`, whose input is circular even though its output is a diagnostic.

The headline demonstration: `crypto_qtm_panel_velocity`'s fair value equals the market price to
floating point, on every applicable row of all 19 symbols.

| | Max absolute deviation from price | Max relative deviation |
|---|---|---|
| Worst symbol (POL, 364 rows) | 8.3e-17 | 6.7e-16 |
| ETH (364 rows) | 9.1e-13 | 4.4e-16 |

That is machine epsilon. The dissertation's QTM, read as its own text specifies and fed the only
velocity series that exists, returns the price it was given. Its residuals against price are zero
by construction, and so are those of every other method in the list above to the extent its input
is a market-cap identity. None of them is independent evidence of value; they are cost bases and
diagnostics, and the report, the cards and `dim_method` all label them as such.

## 6. The verdict

Of the dissertation's ten valuation methods, **two survive contact with 2026 data as independent
evidence**: Hayes cost of production (for the four proof-of-work assets whose hash rate the free
tier serves) and the thermocap floor — both cost-side lower bounds, neither a target price. **Two
more are computable but not independent**: MVRV and MVTV reversion, which agree with price to
within 8% precisely because they are built from it. **One is computable and falsified by its own
diagnostics**: Metcalfe, whose residuals are autocorrelated for all 20 assets and whose β bears no
relation to the published 1.69. **One is arithmetic**: QTM on panel velocity returns the price to
machine epsilon. **One is dimensionally incoherent as printed** and had to be re-derived before it
could run at all: the J-curve's CUV. **Two cannot be computed on free data at all**: the NVT golden
ratio and price-to-utility. And the inversion `H = 1/V` that underpins QTM and P/U is wrong by 16%
to 57% per asset, on a log-normality premise that is itself rejected for three of the five assets
where it can be tested.

The dissertation's chapters 3 to 6 are a competent survey of the literature's formulas. What this
run establishes is that when those formulas are actually executed, the market approach and much of
the cost approach are measuring price against itself, and the honest output of the exercise is not a
fair value but a spread of 18x and a list of refusals, each carrying the measurement that caused it.

---

### Reproducing this chapter

```bash
make cards                                   # regenerate docs/methods/ from config
python -m cryptoval.cli estimate             # writes gold/valuation_estimate
make dbt                                     # fct_valuation_estimate, dim_method
make validate                                # exits 0; 4 warnings, all unmapped-entity
```

`make validate` passes on this vintage with four warnings, all on entities outside the valuation
universe (`trx_eth` density and freshness, negative DefiLlama fees on 17 unmapped protocols).
