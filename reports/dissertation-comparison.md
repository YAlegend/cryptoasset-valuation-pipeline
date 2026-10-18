# 2022 dissertation findings vs what holds in 2026

**These are valuation estimates under stated assumptions, not investment advice. These assets are
volatile and these methods disagree by design.**

Run of 2026-10-18 against `data/cryptoval.duckdb`, panel vintage `2026-10-18 16:44:17+01:00`.
Panel: `fct_asset_metrics_daily`, 69,379 symbol-days, 24 symbols, 2015-01-01 to 2026-10-18.
Estimates: `fct_valuation_estimate`, 1,439,499 rows over 41 methods. This chapter is the summary.
Each later phase has its own chapter with the full evidence, and every claim below names the
chapter that tested it.

*Revised 2026-10-18. The first version (2026-10-13) covered the 14 methods of Plan 2 only. This one
adds the modern layer, the forecasting, cointegration and volatility tests, and the Bitcoin
coin-age family. Numbers that moved since then are restated from today's run.*

**The result, in one sentence: of the dissertation's valuation formulas, two give independent
evidence of value on 2026 data, both of them cost floors; its regression result (Chapters 7-8) was
an artifact of leakage and, done properly, nothing beats a random walk; its volatility claim holds
in one half and fails in the other; its BTC-ETH correlations reproduce while its crypto-equity
ones do not; and its holder-behaviour signals SLRV and STHPL read the 2017 top and the 2018 and 2022
bottoms as it says, while RPV never reaches its stated thresholds.**

---

## 1. What the dissertation publishes, and what it does not

There is no published fair value to compare against, and that is the first finding.

- **The dataset:** the empirical part is **Bitcoin alone**: daily Yahoo Finance OHLCV from
  1 January 2015 to 30 November 2022, with seven attributes (§8.2), no on-chain series and no
  second asset.
- **The numeric results:**
  - a price regression (§8.3: linear, SVR and a feed-forward network at shifts of 30, 100, 200
    and 365 rows), reported qualitatively;
  - five Pearson correlation windows (Tables 3-10).
- **Chapters 3 to 6** (QTM, J-curve, Metcalfe, NVT, P/U, cost of production, MVRV, MVTV, investor
  capitalisation, SLRV, RPV, STHPL) **state formulas and never compute them**. There is no fair
  value, no fitted β and no cost per coin anywhere in the document.

So the comparison is formula-level for Chapters 3-6: given each formula as written, does it yield
a defensible number on live data? It is result-level for Chapters 7-8: does the published finding
survive a correct protocol?

## 2. The scorecard

| Dissertation claim | 2026 verdict | Evidence |
|---|---|---|
| Ch. 3-6: the valuation formulas price cryptoassets | **Two give independent evidence**, both cost floors: Hayes cost of production (4 PoW assets) and the thermocap floor. MVRV, MVTV and investor cap are circular; QTM as written returns price; Metcalfe is falsified by its own diagnostics; the J-curve is incoherent as printed; the NVT golden ratio and P/U cannot be computed on free data | §3-4 below |
| §3.1, §3.6: holding time is `1/V` | **Wrong at the mean.** It understates mean holding time by 15-56% on all 20 assets with a year of volume; the closed-form correction's own log-normal premise is rejected on 18 of 20 | §4(c) |
| §8.3: linear regression predicts price best | **An artifact.** A shuffled split on a price-from-price target. With a purged walk-forward split, no model beats naive persistence on any asset at any horizon: 0 of 1,069 tests, before or after Holm | §5; `does-anything-beat-a-random-walk.md` |
| (Implied by §3.3-3.5) MVRV, NVT and Metcalfe anchor price | **Partly.** Long-run equilibrium with realized cap on 8 of 20 assets and with volume on 2 of 18; Metcalfe's on none. Where an equilibrium holds, the fundamental does the correcting, not price | `does-price-anchor-to-fundamentals.md` |
| §6.3: high volatility means unpredictable direction, low means trend continuation | **Half.** Volatility clusters on 20 of 21 assets, but the direction claim fails on all 21 (0 after Holm) | §6; `does-volatility-cluster.md` |
| Tables 3-10: crypto correlations by window | **BTC-ETH reproduces** within 0.06 in every window. **Crypto-equity does not:** BTC's equity correlations are over twice the dissertation's ceiling. Correlation is time-varying on 11 of 20 crypto pairs | §6 |
| §8.4.3: assets "part company" at a bull-market peak | **Right in direction, not unusual.** Lower peak correlation on 16 of 19 pairs, but the median pair sits at the 35th percentile of its own history | §6 |
| §5.5, §6.1, §6.2: SLRV, RPV and STHPL thresholds mark tops and bottoms | **SLRV and STHPL yes, RPV no.** Both peak on 2017-12-07 and sit in their oversold zones through 2018 and 2022. RPV never reaches 2% (maximum 1.68%) | §7; `what-do-coin-ages-say.md` |
| Ch. 1: cryptoassets have neither cash flows nor earnings | **No longer true for a large part of the market.** Protocol revenue supports DCFs for 9 assets, and multiples on fees, sales and TVL | `modern-layer.md` |

## 3. The dissertation's valuation formulas on today's data

Coverage across the 24-symbol universe:

| Method (dissertation §) | Assets with estimates | Status |
|---|---|---|
| `thermocap_floor` (§4.4) | 20 | Independent cost floor |
| `hayes_cost_of_production` (§4.1) | 4 (BTC, BCH, DOGE, LTC) | Independent; needs hash rate ≥ 90% dense, so ETH's 63.4% refuses |
| `realized_price`, `mvrv_reversion` (§4.2) | 20 | **Circular**: realized cap = market cap / MVRV |
| `investor_cap_price` (§4.5) | 20 | **Circular** |
| `mvtv_reversion` (§4.3) | 9 | **Circular** |
| `crypto_qtm` (§3.1) | 20 | Exogenous velocity |
| `crypto_qtm_panel_velocity` (§3.1, as written) | 19 | **Returns price**: maximum relative deviation 5.8e-16 (§4b) |
| `j_curve` (§3.2) | 20 | Re-derived; incoherent as printed (§4a) |
| `metcalfe_estimated` (§3.3) | 20 | Falsified by its own diagnostics (§4e) |
| `nvt_reversion` (§3.4) | 19 | Exchange volume, not on-chain value (§4d) |
| `nvt_golden_ratio` (§3.4) | **0** | `only 370 rows, below the method's minimum of 1095` |
| `price_to_utility` (§3.6) | **0** | `only 370 rows, below the method's minimum of 730` |
| `slrv` (§5.5) | 1 (BTC) | Live since Plan 3b's coin-age extraction (§7) |

**The spread is the result.** BTC on its latest day has 22 methods that return a price. Their grid
medians run from $3,706 (`metcalfe_estimated`) to $155,608 (`stock_to_flow`), a factor of 42,
against a market price of $81,020. For ETH, 24 methods span $0.54 to $3,687, a factor of 6,844,
against $2,375. A single number quoted from this registry is a choice of method, not a measurement
of value. The dashboard's football field (`make dashboard`) draws the whole range for any asset.

## 4. What did not reproduce, and why

**(a) A formula that is wrong as printed.** The CUV of §3.2 reads "token's current supply / (token
supply × token velocity)", which divides a stock by a rate with no holding period. `j_curve`
implements the economically intended form, which is why it coincides with `crypto_qtm`.

**(b) A formula that measures its own input.** With velocity taken from the panel, which is the
dissertation's own reading of §3.1 since the only velocity available is volume divided by market
cap, QTM collapses to price. On every applicable row of all 19 assets, `crypto_qtm_panel_velocity`
equals price to a maximum relative deviation of 5.8e-16: machine epsilon.

**(c) A relationship the dissertation relies on that does not hold.** §3.1 and §3.6 invert
velocity to get holding time, `H = 1/V`. It now has a year of volume on 20 non-stablecoin assets,
up from 5 in the first version of this chapter. The inversion understates the log-normal mean
holding time on every one of them:
- the ratio runs from 1.15 (LTC) to 1.56 (ETH), with a median of 1.24;
- BTC's is 1.37, at 38.4 days naive against 52.5 days corrected.

The correction's own premise, log-normal velocity, is rejected (Anderson-Darling p < 0.05) on 18 of
the 20. Only BCH (p = 0.058) and DOGE (0.50) pass. So the inversion is biased everywhere, and the
closed-form fix is itself an approximation almost everywhere. The dissertation states neither.

**(d) A different denominator.** NVT as published divides network value by *on-chain transaction
value*, but the free tier serves *exchange volume*. The two volume series carried here differ by a
median factor of 3.78 on BTC over a year. Reversion is invariant to a constant rescaling but not to
a change of shape, and every NVT row records which volume it used.

**(e) A fit whose diagnostics invalidate it.** `metcalfe_estimated` reports β with a confidence
interval and a Durbin-Watson statistic on every row:
- **Residuals:** autocorrelated on all 20 assets, with DW from 0.004 to 0.277 against a floor of 1.5.
- **BTC's β:** 3.89 (CI 3.77-4.01, R² 0.49). That is neither Metcalfe's 2.0 nor the published
  ≈ 1.69, and its confidence interval means nothing at DW 0.107.
- **Cointegration agrees:** the cointegration chapter finds no asset with the equilibrium the
  exponent would describe.

Metcalfe's law on this data is a regression of one trending series on another.

**(f) What cannot be computed on free data.** Each case carries the refusal it records:
- **TRX:** no realized cap (`realized_cap is 0.0% dense`).
- **BNB:** Coin Metrics covers 19.1% of its history.
- **SOL:** no Coin Metrics series at all, so it has no realized cap and no active addresses.
- **NVT golden ratio and P/U:** both need more history than CoinGecko's keyless 365-day volume
  window (370 rows today) can ever hold. They will keep refusing until a source with a longer
  history is added.

## 5. The Chapter 7-8 correction

The dissertation's regression code sets `Prediction = Close.shift(-N)`, uses `Close` as the only
feature, and splits with `train_test_split`, which shuffles by default. Test rows therefore sit
between training rows, and predicting tomorrow's price from today's on a shuffled split scores
almost perfectly at short shifts, then worse as N grows. "Linear regression beat SVR and the neural
network" measures that leakage plus persistence, not model quality.

`reports/does-anything-beat-a-random-walk.md` rebuilds the question correctly:
- **Target:** the forward h-day log return, not the price level.
- **Features:** every valuation method's signal, lagged one day.
- **Split:** a purged, embargoed walk-forward split, with no shuffling anywhere. A repository test
  forbids `train_test_split` and `shuffle` outright.
- **Models:** 11, from OLS to a GRU.
- **Scoring:** 5 horizons on 22 assets, with Diebold-Mariano tests against naive persistence and
  a Holm correction across 1,069 tests.

**No model beat persistence anywhere.** 457 tests were significant after Holm, and persistence had
the lower error in every one of them. The dissertation's Chapter 7-8 result does not survive a
protocol that cannot see the future.

## 6. Volatility and correlation (§6.3, §8.4)

From `reports/does-volatility-cluster.md` (GARCH, EGARCH, Markov regimes and DCC, full sample,
descriptive, Holm over 97 hypotheses):

- **§6.3, first half: volatility clusters.** ARCH-LM rejects on 20 of 21 testable assets.
- **§6.3, second half: low volatility means trend continuation.** Rejected. A block test of that
  sentence finds nothing on any of the 21, none below 0.05 even before correction. No asset shows
  a significant volatility asymmetry either: EGARCH γ is significant on 0 of 21.
- **Tables 3-10, BTC-ETH:** the Pearson figures reproduce within 0.06 in every window.
- **Tables 3-10, crypto-equity:** these do not reproduce. BTC against equities is between 0.29 and
  0.36 here, against the dissertation's 0.14 or lower.
- **Correlation that moves over time:** the dissertation's conclusion said it could not test this.
  It can now, and the Engle-Sheppard test finds time-varying correlation on 11 of 20 crypto pairs
  and 3 of 14 crypto-equity pairs.
- **§8.4.3, "part company" at a peak:** the peak's correlation is below the 2018-2022 window's on
  16 of 19 pairs. But the median pair's peak sits at the 35th percentile of its own two-month
  correlations, which is a low reading, not an unusual one.

## 7. Holder-behaviour signals (§5.5, §6.1, §6.2)

These were formulas without data until Plan 3b extracted Bitcoin's coin ages from BigQuery's
public dataset. The extraction is checked against Coin Metrics, which computes both series
independently: supply agrees to a median of 0.0004% and realized cap to 0.014%. From
`reports/what-do-coin-ages-say.md`, over 2015-2026:

- **SLRV:** below the dissertation's bear-market threshold of 0.04 on 611 days, mostly 2018,
  2022, 2015 and 2026. It peaked at 5.42 on 2017-12-07, ten days before that cycle's price top.
- **STHPL:** below unity, the "short-term holders at a loss" zone, on 42% of days, concentrated in
  2018 and 2022. It also peaked on 2017-12-07.
  - It is computed as short-term-holder MVRV, because the dissertation's literal ratio of profits
    is undefined whenever long-term profit crosses zero.
- **RPV:** never reaches the 2% "top" threshold (maximum 1.68%, on 2017-12-07), and its second
  highest reading falls at the December 2018 *bottom*. Every spend counts as a realisation here,
  including exchange reshuffles, so capitulation inflates it too.

These readings are descriptive. Whether any of the thresholds would have helped a forecaster is
the question §5 answers, and it answers no.

## 8. The verdict

The dissertation's Chapters 3 to 6 are a competent survey of the literature's formulas. Executed on
2026 data:
- **Independent evidence:** two give it, both cost floors (Hayes, for the four proof-of-work assets
  whose hash rate is served free, and the thermocap floor).
- **Circular:** MVRV, MVTV and investor cap agree with price because they are built from it.
- **Arithmetic:** QTM as written returns the price to machine epsilon.
- **Falsified:** Metcalfe, by its own residuals.
- **Incoherent as printed:** one, the J-curve's CUV.
- **Not computable:** two, on free data.
- **Wrong at the mean:** the holding-time inversion beneath QTM and P/U, on every asset where it can
  be measured.

Its empirical chapters fare as follows:
- **The regression result** was leakage.
- **The volatility claim** is half right.
- **The correlation tables** reproduce for BTC-ETH and fail for crypto-equity.
- **The holder-behaviour signals** behave as described for SLRV and STHPL, and not for RPV.

What the dissertation could not have known, and this project adds, is two things. Cash-flow
valuation now exists for a real part of the market. And cointegration shows that where price and a
fundamental do hold an equilibrium, it is the fundamental that does the adjusting.

The honest output of rebuilding the dissertation is not a fair value. It is a spread of 42x on BTC
and 6,844x on ETH, a list of refusals that each carry the measurement behind them, and a null
forecasting result that every method on the shelf failed to beat.

---

### Reproducing this chapter

```bash
make all          # ingest, dbt, estimate, score, econ, vol, validate (~45 min)
make utxo         # monthly, by hand: the BigQuery coin-age extraction (README)
make dashboard    # the football field and heatmap behind section 3
```

Each linked chapter ends with its own reproduction commands and the query behind every number.
