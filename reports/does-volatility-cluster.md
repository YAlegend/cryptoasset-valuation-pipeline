# Does volatility cluster, and does correlation move? — Phase 4c

**These are valuation estimates under stated assumptions, not investment advice. These assets are
volatile and these methods disagree by design.**

Run of 2026-10-18 against `data/cryptoval.duckdb`, panel vintage `2026-10-18 16:44:17+01:00`.
Crypto panel: `fct_asset_metrics_daily`, 69,379 symbol-days, 24 symbols, 2015-01-01 to 2026-10-18.
Traditional panel: `fct_equity_prices_daily`, 20,756 rows, 7 series (TSLA, AAPL, AMZN, GOOGL, META,
GOLD, DXY), 2015-01-02 to 2026-10-16. Results: `fct_volatility_test`, 1,623 rows, written by
`python -m cryptoval.cli vol` in under two minutes. Every number below is a query against that
table.

**The result, in one sentence: volatility clusters on 20 of 21 testable assets, so the first half
of the dissertation's §6.3 picture holds; the second half does not — whether price keeps its
direction is unrelated to volatility on all 21 (0 of 21 after Holm, none below 0.05 even
uncorrected); no asset shows a significant volatility asymmetry; and the BTC correlation the
dissertation measured in five static windows is genuinely time-varying on 11 of 20 crypto pairs and
3 of 14 crypto-equity pairs, which is what its own conclusion said and could not test.**

The dissertation's BTC-ETH Pearson figures reproduce to within 0.06 in every window. Its
crypto-equity figures do not (§6). Its bull-market-peak claim — that assets "part company" at the
top — is borne out in direction (the peak's correlation is below the 2018-2022 window's on 16 of 19
pairs) but not as anything unusual: the median pair's peak sits at the 35th percentile of its own
two-month correlations.

**Full sample, descriptive.** Every parameter is estimated over the whole history, so every
conditional volatility and correlation path uses future data through its parameters. Nothing here
supports a tradability claim, and nothing here is comparable to Phase 4's out-of-sample null
(`reports/does-anything-beat-a-random-walk.md`).

---

## 1. What this tests

Spec §11.4's second chain, each family against the dissertation claim it tests:

| Family | The dissertation's claim | How it is tested here |
|---|---|---|
| GARCH(1,1) / EGARCH | §6.3: "elevated volatility bespeaks an uncertain and unpredictable market in which the direction of price resists forecasting; subdued volatility bespeaks a steadier market whose price may be expected to continue upon its established course" | ARCH-LM (does volatility cluster), EGARCH γ (is it asymmetric), and a block test of the sentence itself (§4) |
| Markov-switching | §8.4.3: "in the euphoria of a bull-market peak the constituent assets of the market part company" | Two variance regimes per asset; was October-November 2021 a turbulent regime (§5) |
| DCC-GARCH | Discussion: "the relationship between Bitcoin and other assets shifts over time and resists confident prediction" | Bivariate DCC(1,1) per pair, the Engle-Sheppard test of constant correlation, and the dissertation's windows side by side (§6) |

Returns are daily log returns of `price_usd` in percent. Every GARCH and EGARCH fit uses
Student-t innovations; GARCH's fitted degrees of freedom run from 2.9 to 5.8, so a normal likelihood
would have been wrong on every asset.

## 2. The universe, measured

24 crypto assets configured; **21 testable**. Three refusals, each a row in the table:

| Asset | Reason |
|---|---|
| USDC | Stablecoin: a pegged price's variance is not a market quantity |
| SOL | 369 returns, 730 needed (CoinGecko history from 2025-10-14) |
| TRX | 369 returns, 730 needed |

The 21 run from 1,089 returns (POL, from 2023-10-25) to 4,308 (BTC, DOGE, LTC). POL has no
observation inside any of the dissertation's windows, so all of its window rows are NaN with
`0 rows in window …` as the reason; LDO has none inside `nft_2020_2021`.

The dissertation's universe differs: XRP, APE, FLOW, SAND and AXS are not configured, SOL is
refused, and MANA is the only NFT/metaverse asset present. Tables 9-10 therefore have one asset
to compare, and this chapter does not pretend to reproduce them.

## 3. Volatility clusters, and does not lean

**ARCH-LM (7 lags), H0 no clustering: rejected on 20 of 21 after Holm.** Every statistic but one is
above 63; the exception is POL (19.7, raw p 0.0062, Holm 0.37), the shortest series in the panel.

**GARCH(1,1)-t persistence** α+β runs from 0.907 (POL) to the stationarity bound:

| Asset | α | β | α+β | Half life (days) |
|---|---|---|---|---|
| MKR | 0.148 | 0.781 | 0.929 | 9.4 |
| POL | 0.133 | 0.775 | 0.907 | 7.1 |
| MANA | 0.208 | 0.751 | 0.958 | 16.3 |
| AAVE | 0.117 | 0.851 | 0.967 | 20.7 |
| ADA | 0.169 | 0.798 | 0.967 | 20.8 |
| LINK | 0.083 | 0.904 | 0.986 | 49.0 |
| BCH | 0.098 | 0.896 | 0.994 | 109.4 |
| UNI | 0.026 | 0.970 | 0.997 | 196.3 |
| BTC | 0.105 | 0.895 | 1.000 | — |
| ETH | 0.169 | 0.831 | 1.000 | — |

(Ten of 21 shown; the rest are in `fct_volatility_test`, `model_id = 'garch'`.) On **five assets —
BTC, ETH, BNB, DOGE, LTC — persistence sits at 1**: ETH and LTC fit exactly 1, and BTC, BNB and
DOGE fit a value so close to it that the implied half life is 10¹¹ to 10¹⁵ days. The first live run
published those numbers; they are the optimizer resting on the bound, not a measurement, and the
runner now refuses any half life longer than the sample with both numbers in the reason. A
persistence of 1 is an integrated GARCH: a volatility shock never decays, which is the sharpest
possible form of clustering and also a sign that one GARCH over 2015-2026 is averaging volatility
regimes it cannot separate (§5).

**EGARCH(1,1,1)-t asymmetry γ, H0 symmetric response: 0 of 21 after Holm.** Four are below 0.05 raw
(ALGO 0.0009, LTC 0.0078, DOT 0.017, AVAX 0.026); the smallest Holm-adjusted is ALGO at 0.0558. The
sign leans one way — **γ > 0 on 17 of 21**, the inverse of the equity leverage effect, meaning rises
raise volatility at least as much as falls — but no single asset carries it past the correction.

## 4. The §6.3 claim: does low volatility mean price keeps its course?

**No. 0 of 21 after Holm, and the smallest raw p-value is 0.083 (ADA).**

The test cuts each asset's returns into non-overlapping 30-day blocks (overlapping windows would
make the counts mechanically correlated) and asks, for each consecutive pair, whether block k+1
kept block k's sign, split by whether the GARCH conditional volatility on block k's last day was
above or below that asset's median. 35 (POL) to 142 (BTC) block pairs per asset; Fisher's exact
test, two-sided.

| | Mean across 21 assets |
|---|---|
| Continuation after a high-volatility block | 0.516 |
| Continuation after a low-volatility block | 0.484 |

Both are a coin flip, and the small difference runs **against** the dissertation: low volatility is
followed by continuation on only 8 of 21 assets more often than high volatility is. BTC: 0.521 high,
0.479 low, over 142 block pairs, p 0.74.

What this does not test, and cannot here: the other half of §6.3, volatility "taken together with
profit metrics". SOPR and its relatives are UTXO-level series deferred with the §6.3 BigQuery
backfill. A claim that volatility *plus* profit-taking predicts direction is untouched by this
result; a claim that volatility alone does is rejected.

## 5. Regimes, and the bull-market peak

A two-regime Markov-switching model (switching mean and variance) separates a calm and a turbulent
regime on all 21 assets. The turbulent regime's variance is 5.0 to 27.0 times the calm one's; BTC's
calm and turbulent variances are 2.9 and 29.4 (daily standard deviations of 1.7% and 5.4%), with
expected durations of 10.3 and 5.6 days, and BTC spends 31.8% of days in the turbulent regime.
**These are estimates, not tests**: the likelihood-ratio test of two regimes against one has a
non-standard distribution, and no p-value from it is reported.

The dissertation selected October-November 2021 "precisely to examine correlation during the peak
of the bull market". Was the peak turbulent? **For BTC, yes**: its mean turbulent probability over
those 61 days is 0.539 against an unconditional 0.318. Across assets it is a split: the peak was
more turbulent than the asset's own average on **10 of 20** (ALGO, AVAX, BNB, BTC, COMP, DOT, ICP,
LDO, LTC, MANA) and calmer on the other 10 — AAVE (0.067), LINK (0.037), SNX (0.041) and UNI
(0.055) were almost entirely calm. "The peak" was not one regime across the market, which is itself
a version of the dissertation's point that the assets "part company".

## 6. Correlation moves — and the dissertation's windows, reproduced

**Engle-Sheppard, H0 constant conditional correlation: rejected on 14 of 34 pairs after Holm** —
11 of 20 BTC-crypto pairs (AAVE, ADA, AVAX, BCH, BNB, CRV, DOT, ETH, ICP, LDO, SNX) and 3 of 14
crypto-equity pairs (BTC|TSLA, ETH|AAPL, ETH|TSLA). BTC|ETH is the strongest rejection in the
family (statistic 134.4). This is the dissertation's concluding sentence as a hypothesis, and on
half the crypto pairs it holds. On the other half — LTC, DOGE, LINK, MKR, MANA among them — a
constant correlation is not rejected, so a single window would not have misled there.

DCC(1,1) `a` runs from 0.003 (BTC|META) to 0.070 (BTC|LTC): crypto pairs react to a joint shock
about four times faster than crypto-equity pairs (median `a` 0.048 against 0.012). One fit is degenerate: **ETH|META** fits
b = 0.000, a near-constant correlation of 0.14, consistent with its Engle-Sheppard non-rejection.

**BTC|ETH, the dissertation's tables against this panel** (Pearson on log returns; the dissertation
used `pct_change`, which differs negligibly at daily scale):

| Window | Dissertation | Pearson here | DCC mean here |
|---|---|---|---|
| 2018-01 to 2022-11 (Table 3) | 0.82 | 0.833 | 0.816 |
| 2021-01 to 2022-01 (Table 4) | 0.84 | 0.780 | 0.763 |
| 2022-02 to 2022-05 (Table 5) | 0.93 | 0.912 | 0.894 |
| 2022-06 to 2022-11 (Table 6) | 0.87 | 0.900 | 0.863 |
| 2021-10 to 2021-11, the peak (Table 7) | 0.73 | 0.788 | 0.788 |

The static figures reproduce to within 0.06. The DCC path adds what no window can: BTC|ETH's
conditional correlation ranged from −0.52 to 0.97 over the sample.

**The peak claim.** BTC's Pearson correlation with each other asset is lower in the peak window
than in 2018-2022 on **16 of 19** pairs, ranging from 0.07 (MANA) to 0.79 (ETH) — the direction the
dissertation reported. But measured against each pair's own history it is ordinary: the percentile
rank of the peak's mean DCC correlation among all 61-day rolling means has a **median of 0.35**
across the 19 crypto pairs, and only **4 of 19** fall in the bottom decile (ALGO 0.041, AVAX 0.034,
LDO 0.012, MANA 0.095). For the four pairs in the dissertation's own Table 7 that are present here
— ETH, DOT, LINK, ADA — the ranks are 0.447, 0.130, 0.363 and 0.150. The peak was a somewhat
less-correlated stretch, not an exceptional one. The rank is descriptive: rolling windows overlap,
so it has no p-value and is outside the Holm family.

**Crypto against equities does not reproduce.** For 2020-01 to 2022-11 (Table 8) the dissertation
reports ETH-META 0.58, ETH-AAPL 0.42, and BTC against every equity at or below 0.14. This panel
gives ETH|META 0.286, ETH|AAPL 0.358, and BTC between 0.289 (META) and 0.361 (GOOGL). BTC's
equity correlations are more than twice the dissertation's ceiling, and ETH-META is half its
figure. The cause is not identifiable from the dissertation's text; the likeliest is how weekend
crypto prices were aligned to trading days, which the dissertation does not describe. This panel
samples crypto on equity trading dates so both legs span the same interval (a Monday return covers
Friday to Monday for both), and the residual bias from crypto closing at 00:00 UTC against NYSE's
16:00 ET runs toward zero, so these figures if anything understate the true correlation.

## 7. Multiple testing

**One Holm family of 97 hypotheses**: ARCH-LM (21), EGARCH γ (21), the §6.3 exhaustion test (21),
Engle-Sheppard (34). Survivors: 20, 0, 0, 14.

Outside the family, deliberately: GARCH and DCC parameters (estimates); Markov-switching (no valid
test, §5); per-window means (descriptions); the peak percentile rank (overlapping windows). The raw
`p_value` rows are kept beside `p_value_holm`, so a later chapter can re-correct over a wider family.

Engle-Sheppard's size was checked by simulation before the run: it rejected 1 of 100 constant-
correlation series at the 5% level (and the shipped test asserts at most 6 of 60), so if anything
it is conservative here.

## 8. Declared gaps

| Spec | What is not here | Why, and what it costs |
|---|---|---|
| §11.4 | Synchronous crypto-equity returns | Crypto closes at 00:00 UTC, NYSE at 16:00 ET. Correlations are biased toward zero; a one-day-lead adjustment is the named alternative and is not built. |
| §11.4 | Out-of-sample volatility | Full-sample parameters (§2 of the design). The §6.3 test's conditional volatility uses them, so even its block classification has hindsight in its parameters. |
| §11.4 | An order search | GARCH(1,1) and EGARCH(1,1,1) only. |
| §11.4 | A multivariate DCC | Bivariate per pair: the dissertation's tables are pairwise, and a 24-asset DCC biases `a` downward at large N. |
| §6.3 | The profit-metric half of seller exhaustion | UTXO series, deferred with the BigQuery backfill (§4). |
| §11.4 | A regime-count test | Markov-switching's LR test is non-standard; none reported. |
| §11.4 | Structural breaks | Persistence at 1 on five assets (§3) is the symptom: one GARCH over twelve years averages regimes the Markov model separates. The two are reported side by side, not reconciled. |
| §8.4 | Tables 9-10 | Four of the dissertation's five NFT assets are not in this universe. |
| §11.5 | Factor models, Fama-MacBeth, decile portfolios | Still blocked on the point-in-time top-N universe. |

---

### Reproducing this chapter

```bash
python -m cryptoval.cli vol                  # 21 assets, 34 pairs, ~2 min
cd dbt && ../.venv/bin/dbt build --select fct_volatility_test --profiles-dir . && cd ..
```

The windows and anchors are `config/volatility.yml`; every number above is a select against
`fct_volatility_test` filtered by `model_id`, `metric` and `window_name`.
