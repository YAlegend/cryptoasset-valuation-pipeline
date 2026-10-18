# The Volatility Chain — Design (Plan 4c-vol)

Spec §11.4 lists six econometric families. Plan 4b built the first three (ADF/KPSS →
Engle-Granger/Johansen → VECM). This design covers the other three — **GARCH(1,1)/EGARCH,
Markov-switching and DCC-GARCH** — and records the decisions §11.4 does not contain. It answers a
different question from Plan 4b, writes a different table, and adds the one dependency Plan 4b
deliberately did not: `arch==8.0.0`.

Written without a design dialogue at the user's instruction ("keep going"); every decision below
that a reviewer might have made differently is marked **Decision** so it can be overturned.

## 1. The questions

§11.4 assigns each family a question, and each one tests a claim the 2022 dissertation makes in its
own words:

| Family | §11.4's question | The dissertation's claim it tests |
|---|---|---|
| GARCH(1,1) / EGARCH | Volatility clustering and asymmetry | §6.3: "elevated volatility bespeaks an uncertain and unpredictable market in which the direction of price resists forecasting; subdued volatility bespeaks a steadier market whose price may be expected to continue upon its established course" |
| Markov-switching | Regime identification | §8.4.3: "in the euphoria of a bull-market peak the constituent assets of the market part company to a degree not observed in calmer or declining conditions" |
| DCC-GARCH | Time-varying conditional correlation | Discussion: "the relationship between Bitcoin and other assets shifts over time and resists confident prediction"; "conclusions drawn from any single window must be held provisionally" |

The dissertation's correlation evidence is **five static Pearson windows** (Tables 3-7, BTC against
ETH, XRP, DOT, SOL, LINK, ADA), one crypto-equity window (Table 8) and two NFT windows (Tables 9-10).
A static coefficient per window cannot say whether correlation moved *within* a window, nor whether
the differences *between* windows exceed what a constant correlation would produce by chance. DCC
answers both.

## 2. Decision: full sample, descriptive, as in Plan 4b

One fit per asset or pair over the whole available history. Parameters are estimated on the full
sample, so every conditional volatility and correlation path uses future data through its
parameters. The chapter must state, as Plan 4b's does, that the claims are **descriptive about the
sample**, support **no tradability claim**, and are **not comparable** to Plan 4's out-of-sample
null. None of Plan 4's fold machinery is reused.

## 3. Decision: the returns

* **Daily log returns of `price_usd`**, in percent (×100) — `arch` estimates poorly on raw
  fractional returns, and percent is the convention its documentation uses.
* **Crypto-crypto pairs** use the 7-day calendar and Plan 4's contiguity guard
  (`_require_contiguous_days`); a gap is refused, not stitched.
* **Crypto-traditional pairs** use the traditional asset's trading calendar: the crypto price is
  sampled on the equity's trading dates and both returns are taken between consecutive trading
  dates. A crypto price missing on an equity trading date is refused. The **nonsynchronous close**
  (crypto at 00:00 UTC, NYSE at 16:00 ET) biases these correlations toward zero; it is a declared
  gap, not corrected (§10).
* **Stablecoins are refused** (`is_stablecoin: true`, today USDC). A pegged price's variance is not
  a market quantity, and its depeg episodes would enter the Holm family as "clustering".
* **Minimum length 730 returns**, Plan 4b's floor, re-measured in the plan rather than assumed.

## 4. GARCH(1,1) and EGARCH(1,1,1)

Per crypto asset, constant mean, **Student-t innovations** (crypto returns are fat-tailed; a normal
likelihood would be misspecified on every asset):

* **ARCH-LM test** (Engle 1982, `statsmodels.stats.diagnostic.het_arch`) on demeaned returns, 7 lags
  (one week). H0: no ARCH effects. **In the Holm family.** This is "does volatility cluster".
* **GARCH(1,1)**: ω, α, β, ν, persistence α+β, and the **volatility half life**
  `log(0.5)/log(α+β)` days where α+β < 1 (NaN otherwise, with a reason — an integrated process has
  no half life).
* **EGARCH(1,1,1)**: the asymmetry term γ, its robust standard error and p-value. H0: γ = 0
  (symmetric response). **In the Holm family.** Sign matters and is reported: γ < 0 is the equity
  leverage effect (falls raise volatility more than rises); γ > 0 is the inverse effect reported
  for crypto.

A fit that fails to converge (`arch`'s `convergence_flag != 0`) is a refusal row carrying the flag
and the optimizer message, never a parameter row.

## 5. The §6.3 claim, tested

**Decision: test the dissertation's sentence directly rather than leave GARCH as description.**

The claim is that price continues on its course when volatility is low and does not when it is
high. Operationally, per asset:

1. Cut the return series into **non-overlapping 30-day blocks** (non-overlapping so the blocks'
   outcomes are not mechanically autocorrelated; overlapping windows would make any count test
   anti-conservative).
2. For each consecutive block pair, **continuation** = the sign of block k+1's summed return equals
   block k's.
3. Classify each pair by the GARCH(1,1) **conditional volatility on block k's last day**, split at
   that asset's median into high and low.
4. **Fisher's exact test** on the 2×2 (high/low × continued/reversed). H0: the continuation rate is
   the same in both. **In the Holm family.** One-sided would encode the dissertation's direction;
   **Decision: two-sided**, and the reported continuation rates show the direction.

What this does **not** test, and the chapter must say so: the other half of §6.3, "taken together
with profit metrics" — SOPR and similar are UTXO-level series deferred with the §6.3 BigQuery
backfill. And the conditional volatility uses full-sample parameters (§2).

## 6. Markov-switching

Per crypto asset, `statsmodels` `MarkovRegression(k_regimes=2, trend="c",
switching_variance=True)` on percent log returns, `search_reps=20` with a fixed seed.

* Regimes are **labelled by variance**, not by index: `calm` is the lower-variance regime. statsmodels'
  regime index is arbitrary and changes between fits.
* Reported per regime: mean, variance, probability of staying, expected duration `1/(1−p_stay)`; and
  the share of days whose smoothed probability of `turbulent` exceeds 0.5.
* Reported per dissertation window (§8): the **mean smoothed probability of `turbulent`**. The
  peak-window question becomes: was October-November 2021 a turbulent regime for this asset?
* **Not in the Holm family.** The likelihood-ratio test of two regimes against one has a
  non-standard distribution (a nuisance parameter is unidentified under the null; Hansen 1992), and
  reporting its chi-squared p-value would be wrong. Estimates only.

## 7. DCC-GARCH

**Hand-rolled**, because no free package supplies it (Plan 4's scoping note). Engle (2002) two-step:

1. Univariate GARCH(1,1)-t per series (§4's fit, reused) → standardized residuals `z_t`.
2. **Bivariate DCC(1,1) with correlation targeting**:
   `Q_t = (1−a−b)·Q̄ + a·z_{t−1}z_{t−1}' + b·Q_{t−1}`, `R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}`,
   `Q̄` the sample covariance of `z`. (a, b) maximize the correlation part of the Gaussian
   quasi-likelihood, `−½ Σ (log|R_t| + z_t'R_t^{-1}z_t − z_t'z_t)`, under a ≥ 0, b ≥ 0, a + b < 1, by
   `scipy.optimize.minimize` (already installed with statsmodels).

**Decision: bivariate per pair, not one 24-asset system.** The dissertation's tables are pairwise;
a full-system DCC has a known downward bias in `a` at large N (Engle, Ledoit & Sheppard 2008) and a
single `(a, b)` shared by every pair would impose one speed of correlation change on BTC-ETH and
BTC-DOGE alike.

**Pairs.** BTC against every other non-stablecoin crypto; BTC and ETH against every traditional
series in `config/assets.yml` (TSLA, AAPL, AMZN, GOOGL, META, GOLD, DXY). Roughly 22 + 14.

Reported per pair: a, b, a+b; the DCC correlation's mean, min and max; and per dissertation window
the DCC mean **beside the static Pearson on the same returns**, so the chapter can set each
dissertation table next to what the conditional model says about the same dates.

**Engle-Sheppard (2001) test of constant correlation**, per pair. Under H0 (constant conditional
correlation) the whitened standardized residuals' cross-product is unpredictable; regress it on a
constant and 5 of its own lags, statistic `δ'X'Xδ / σ̂²` ~ χ²(6). **In the Holm family.** This is the
dissertation's "shifts over time" claim as a hypothesis.

**The peak claim, descriptively.** For the 2021-10-01 to 2021-11-30 window, the percentile rank of
its mean DCC correlation among all 61-day rolling means of the same pair. Rank 0.05 says the peak
was among the least correlated two-month stretches in the sample. **Not a hypothesis**: rolling
means overlap, so no rank has a valid p-value, and the chapter reports it as a description.

## 8. The dissertation windows

Config, not code (`config/volatility.yml`), named as the dissertation names them:

| Window | Dates | Dissertation table |
|---|---|---|
| `full_2018_2022` | 2018-01-01 to 2022-11-30 | 3 |
| `jan21_jan22` | 2021-01-01 to 2022-01-31 | 4 |
| `feb22_may22` | 2022-02-01 to 2022-05-31 | 5 |
| `jun22_nov22` | 2022-06-01 to 2022-11-30 | 6 |
| `peak_oct21_nov21` | 2021-10-01 to 2021-11-30 | 7 |
| `equities_2020_2022` | 2020-01-01 to 2022-11-30 | 8 |
| `nft_2020_2021` | 2020-01-01 to 2021-06-30 | 9 |
| `nft_2021_2022` | 2021-07-01 to 2022-11-30 | 10 |

The universe differs from the dissertation's and the chapter says how: XRP, APE, FLOW, SAND and AXS
are not configured; SOL has 363 rows (2025-10-14 onward) and is refused in every window; MANA is the
only NFT/metaverse asset present.

## 9. Architecture

A new `cryptoval/volatility/` package, the layering of `cryptoval/econometrics/`:

| Module | Responsibility |
|---|---|
| `returns.py` | Percent log returns per crypto asset (contiguity-guarded) and per crypto-traditional pair (equity calendar). Raises `Unusable` with the measurement. |
| `garch.py` | ARCH-LM, GARCH(1,1)-t, EGARCH(1,1,1)-t; returns the standardized residuals and conditional volatility for reuse. |
| `exhaustion.py` | The §6.3 block test (§5). |
| `regimes.py` | Markov-switching fit with variance-ordered labels. |
| `dcc.py` | Bivariate DCC(1,1) estimation and the Engle-Sheppard test. |
| `runner.py` | One long table, Holm over the family, refusals as rows. |

`Unusable` is reused from `cryptoval.econometrics.base`; `correct_family_wise` from
`cryptoval.scoring.multiplicity` with `p_metric="p_value"` — the family is every row whose metric is
`p_value`, across models, which is exactly one correction over one family.

`cryptoval.cli vol` writes gold `volatility_test`; a `fct_volatility_test` mart reads it; `make vol`
runs it and `make all` runs it after `econ`. **Decision: a new table, not rows in
`fct_econometric_test`**: grain differs (pairs are symbol pairs, windows are a dimension) and a
`--pairs` rerun of `econ` replaces its table wholesale.

## 10. Table shape

`(model_id, symbol, metric, value, n_rows, window_name, reason, params_hash, panel_vintage_ts)`.
`symbol` is an asset (`BTC`) or a pair (`BTC|ETH`); `window_name` is NULL except for per-window rows (`window` is a reserved word in DuckDB's parser).
`model_id` ∈ {`arch_lm`, `garch`, `egarch`, `exhaustion`, `markov`, `dcc`}. Hypothesis rows use the
metric `p_value`; `model_id = '*'` holds the hypothesis count. `TEST_COLUMNS` is asserted against the
mart's select list from the first task, as Plan 4b does.

## 11. Multiple testing

One Holm family: ARCH-LM (per asset) + EGARCH γ (per asset) + §6.3 exhaustion (per asset) +
Engle-Sheppard (per pair) — roughly 22 + 22 + 22 + 36 = 102 hypotheses. Outside it, and the chapter
says so: GARCH and DCC parameters (estimates), Markov-switching (non-standard LR), the peak
percentile rank (overlapping windows), per-window means (descriptions).

## 12. Declared gaps

* **Nonsynchronous trading** for crypto-traditional pairs (§3) — biases those correlations toward
  zero. A one-day-lead adjustment is the named alternative, not built.
* **Full-sample parameters** (§2).
* **No order search.** GARCH(1,1) and EGARCH(1,1,1) only; no (p, q) selection.
* **Bivariate DCC only** (§7).
* **The profit-metric half of §6.3** (§5).
* **Markov-switching has no regime-count test** (§6).
* **Structural breaks.** A GARCH fitted over 2015-2026 averages volatility regimes Markov-switching
  is there to separate; the two are reported side by side, not reconciled.

## 13. Testing strategy

Plan 4b's discipline:

* every estimator **refuses its degenerate case** with the measurement;
* **known-answer tests in both directions**: simulated GARCH must show ARCH effects and iid noise
  must not; a simulated asymmetric EGARCH must recover γ's sign; simulated two-regime data must
  recover the variance ordering; a simulated DCC (a=0.05, b=0.93) must be recovered within
  tolerance and rejected by Engle-Sheppard, a constant-correlation simulation must not be;
  a synthetic §6.3 series where continuation depends on volatility must reject, iid must not;
* **mutation proof per task**: each new test is shown to fail against a deliberately broken
  implementation before it is trusted.

## 14. Out of scope

§11.5 (factor models, Fama-MacBeth, decile portfolios — blocked on the point-in-time top-N
universe); the §6.3 BigQuery UTXO backfill; any revisit of Plan 4's forecasting null or Plan 4b's
anchoring result; a forecasting use of conditional volatility.
