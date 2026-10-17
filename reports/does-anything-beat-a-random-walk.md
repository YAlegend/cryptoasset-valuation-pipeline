# Does anything beat a random walk? — Phase 4

**These are valuation estimates under stated assumptions, not investment advice. These assets are
volatile and these methods disagree by design.**

Run of 2026-10-17 against `data/cryptoval.duckdb`, panel vintage `2026-10-14 18:20:40.522565+01:00`.
Panel: `fct_asset_metrics_daily`, 69,264 symbol-days, 24 symbols,
2015-01-01 to 2026-10-14. Scores: `fct_model_score`, 32,462 rows over **11
models**, written by `python -m cryptoval.cli score`. Every number below is a query against that
table and is reproducible with the commands in the last section.

**The result, in one sentence: no model significantly beat naive persistence on any asset at any
horizon, before or after the multiple-testing correction.** Of 1,069 Diebold-Mariano tests, 724 were
significant at the 5% level before correction and 457 after Holm-Bonferroni; in every one of
those 457 persistence had the lower squared error, and 0 favoured a model.
Spec §11.3 predicted this in advance and called it a result rather than a failure.

---

## 1. What Phase 4 tests, and what it cannot

11 models, scored on a purged expanding window at five horizons, h in
{1, 7, 30, 90, 365} days, against one target: the forward h-day **log return**. No model is fitted
to a price level.

| Family | Models |
|---|---|
| Baseline (3) | `naive_persistence` (forecasts a return of exactly zero), `random_walk_drift` (mean one-day log return, scaled by h), `historical_mean` (training-fold mean of the h-day return) |
| Linear (3) | `ols`, `ridge` (alpha 1.0), `elastic_net` (alpha 0.001, l1_ratio 0.5) |
| Nonlinear (4) | `svr_rbf`, `svr_poly` (degree 2), `feedforward_nn` (64-32 ReLU), `gradient_boosting` (200 trees, depth 3) |
| Recurrent (1) | `gru` (20-day sequences, 16 hidden units, 30 epochs) |

Every feature is the fair value of one of the registered valuation methods, used as a signal. The universe
is 24 assets. **22 can be folded at h of 90 days or less, and 21 at h=365.**
The two that cannot be folded at all:

| Asset | Rows | Why | Refused at |
|---|---|---|---|
| TRX | 365 rows from its first price date, 364 of them with a price (its panel spine runs back to 2017, but no price exists before 2025-10-14) | below the 730-row floor for five purged folds | every horizon, with that count |
| SOL | 363 | below the 730-row floor | every horizon, with that count |

(The brief for this phase put both at 364. The warehouse says TRX has 364 priced days
inside a 365-day span, one day with no price, and SOL has 363. The refusal rows carry
the measured counts, not the brief's.) A third asset, POL, folds at h of 90 or less but is refused
at h=365: `1085 rows at horizon 365 leaves 720 usable rows; the first test row is 740 (365 training + 365 purge + 10 embargo), and 5 test blocks need 745 usable rows`. That is why the h=365 column has 21 assets and not 22.
A scored cell is one (model, asset, horizon): 109 for each of the ten models that run at all
five horizons (22 assets x 4 horizons + 21 at h=365) and 88 for the `gru`, which
refuses h=365 (§6).

**What this chapter does not answer.** Spec §11.4's econometrics (ADF/KPSS, cointegration, VECM,
GARCH, Markov-switching, DCC) are Plan 4b and §11.5's factor models, Fama-MacBeth regressions and
decile portfolios are Plan 4c. So the question "does a valuation signal *earn returns*" is **not**
answered here. This chapter answers a weaker and different question: does a valuation signal
*forecast returns* out of sample, better than forecasting none.

## 2. The protocol, and the three leaks it closes

Every leak this protocol closes is invisible when it reopens: a leaked feature or an unpurged
boundary makes the metrics *better*, so nothing fails and the result looks like a discovery. Each
has a test that asserts it directly.

Folds are sized from each asset's first priced day: panel rows before it have no target and are
dropped before folding (TRX has 3,294 panel rows and 364 priced).

**Leak 1: a feature that already knows the day it predicts.** Every signal is lagged one full day
(`FEATURE_LAG_DAYS = 1`) within its own asset, so a fair value is only usable the day after it
was computed. This matters most for the methods that read the market price or a trailing window
that includes today: `crypto_qtm_panel_velocity` is built from price, and the Mayer and Puell
multiples use inclusive windows. A signal whose variance is below 1e-12 is dropped with its
measured variance. One was: `crypto_qtm_panel_velocity` (variance 2.370e-32). That leaves
27 of the 28 candidate signals.

**Leak 2: a training label that overlaps the test block.** The label at row t is the return from t
to t+h, so a training row within h days of the test block carries a label that reaches into it. The
splitter removes the last **h rows** of training and then a further **10-day embargo**:
`train.stop + h <= test.start`, asserted over a 40-case grid. For h=365 the arithmetic is: 365
minimum training rows + 365 purged + 10 embargo = **740**, the earliest a first test row can start,
and five test blocks need 745 usable rows after that, so an asset needs about 1,110 rows to be
scored at all at h=365 (POL has 1,085 and is refused with exactly that arithmetic).
There are five folds per asset and horizon; the test blocks are disjoint and time-ordered.

**Leak 3: imputation and selection that peek.** A missing signal value is filled with the median of
the **training fold only**, never of the whole series, and a feature is used in a fold only if it
is at least 50% populated on that fold's training rows. That per-fold selection has a visible cost:
for **BNB and DOT** no signal clears the coverage floor on fold 0 at any horizon, so
each of the seven feature-using non-recurrent models records 10 refused fold rows
(2 assets x 5 horizons) with that reason; the `gru` records 8 of them. The three baselines
use no features and are scored on every fold.

The Diebold-Mariano test (spec §11.2) compares squared forecast errors against `naive_persistence`,
pooled over each asset's five test blocks, with sign fixed once: a **negative** statistic means the
model had the lower loss. The long-run variance is a **Bartlett-weighted HAC estimate over lags
0 to h-1 with an n divisor**, with the Harvey-Leybourne-Newbold small-sample correction and a
t reference distribution. *This was corrected during the build.* The plan specified equal weights
on every lag (a rectangular kernel), which is not positive semi-definite: measured on simulated
errors (400 draws per cell) it returned a non-positive variance, and so a refusal instead of a
p-value, in 46% to 72% of draws at the long horizons (45.8% at n=200, h=90; 69.0% at n=500, h=365;
72.0% with autocorrelated errors at the same cell), against 0% to 9% for the Bartlett weights.
Newey and West's contribution was precisely the Bartlett kernel for this reason.
If a model refused some fold (the two coverage-floor assets above), it is compared with persistence
on the folds it scored and the pooled n printed with each test says how many rows that was.
**0 of the 1,069 DM tests in this run refused.**

## 3. The baselines, and the sample-size caveat

The three baselines are scored as ordinary rows, so "did it beat the random walk" is a query and not
a claim. Median MASE per fold, by horizon (MASE is scaled by the **mean absolute return over the
training fold**, the error of forecasting no return):

| model | MASE h=1 | MASE h=7 | MASE h=30 | MASE h=90 | MASE h=365 |
|---|---|---|---|---|---|
| naive_persistence | 0.743 | 0.716 | 0.688 | 0.697 | 0.608 |
| random_walk_drift | 0.748 | 0.734 | 0.732 | 0.744 | 0.834 |
| historical_mean | 0.748 | 0.735 | 0.728 | 0.778 | 0.761 |

Note what that table does *not* say. MASE below 1.0 is not the bar, because persistence itself
scores below 1.0 everywhere (the test blocks are calmer than the training blocks they are scaled
by, most of all at h=365). The bar is persistence's own row. The two other baselines
score above it in every column, and tested against it they never win; after Holm
`historical_mean` loses on 1 test(s) and
`random_walk_drift` on 0:

| model | tests | raw p<0.05 | Holm p<0.05 |
|---|---|---|---|
| historical_mean | 109 | 34 | 1 |
| random_walk_drift | 109 | 34 | 0 |


**The honest sample size.** Daily rows at horizon h overlap h-fold, so the number of independent
observations is about rows / h. Every score row carries it (`non_overlapping_windows`):

| h | assets | min windows | max windows |
|---|---|---|---|
| 1 | 22 | 1,085 | 4,305 |
| 7 | 22 | 155 | 615 |
| 30 | 22 | 36.2 | 143.5 |
| 90 | 22 | 12.1 | 47.8 |
| 365 | 21 | 5.1 | 11.8 |

A h=365 result rests on **at most 11.8 independent annual observations** (BTC, LTC, DOGE) and
as few as 5.1 for the shortest-history asset that is scored, against
thousands at h=1. It is reported, but it is not comparable to a h=1 result and should not be read as
though it were.

**A kernel caveat the p-values at h=365 must carry.** Task 4 measured how the Bartlett statistic
compares with the rectangular one it replaced, on simulated errors: where the pooled test-row count
is close to the horizon (n/h of about 1.4) the Bartlett statistic runs about **1.5x** the
rectangular one. That ratio was measured at n/h of 1.4 and *not* at 2 to 3, so how much of any
given h=365 figure is the kernel is unknown. The pooled n for every DM test is therefore printed
below, and **a borderline h=365 result is partly a kernel artefact.** The pooled n per horizon:

| h | min pooled n | median pooled n | max pooled n |
|---|---|---|---|
| 1 | 708 | 2,348 | 3,928 |
| 7 | 696 | 2,337 | 3,916 |
| 30 | 650 | 2,296 | 3,870 |
| 90 | 530 | 2,188 | 3,750 |
| 365 | 754 | 1,819 | 3,200 |

In this run the smallest pooled n/h at h=365 is 2.07 (LDO, n =
754); 34 of the 189 h=365 tests have n/h below 3.

## 4. What beat persistence, before and after the correction

**1,069 hypotheses were tested** (model x asset x horizon, every model other than persistence itself,
220 each at h of 90 or less and 189 at h=365, all
returned a p-value). At alpha = 0.05, roughly 53 (that is 1,069/20) are expected to look
significant with no skill present at all. **724 were significant before correction. After
Holm-Bonferroni, 457 survive. All 457 are persistence beating the model; zero are a
model beating persistence.**

Read it carefully, because the number 724 is not a discovery: the test is two-sided and the
significant results are all on one side, and it is the side of the zero forecast.

| model | tests | raw p<0.05 | of which beat persistence | Holm p<0.05 | of which beat persistence (Holm) |
|---|---|---|---|---|---|
| svr_rbf | 109 | 83 | 0 | 65 | 0 |
| feedforward_nn | 109 | 91 | 0 | 64 | 0 |
| gradient_boosting | 109 | 87 | 0 | 60 | 0 |
| svr_poly | 109 | 85 | 0 | 58 | 0 |
| gru | 88 | 79 | 0 | 56 | 0 |
| ridge | 109 | 79 | 0 | 54 | 0 |
| ols | 109 | 79 | 0 | 51 | 0 |
| elastic_net | 109 | 73 | 0 | 48 | 0 |
| historical_mean | 109 | 34 | 0 | 1 | 0 |
| random_walk_drift | 109 | 34 | 0 | 0 | 0 |

By horizon:

| h | DM tests | p-value returned | refused | raw p<0.05 | Holm p<0.05 | beat persistence (Holm) | negative statistic, any p | min pooled n/h |
|---|---|---|---|---|---|---|---|---|
| 1 | 220 | 220 | 0 | 179 | 154 | 0 | 5.0000 | 708 |
| 7 | 220 | 220 | 0 | 185 | 156 | 0 | 4.0000 | 99.43 |
| 30 | 220 | 220 | 0 | 179 | 113 | 0 | 7.0000 | 21.67 |
| 90 | 220 | 220 | 0 | 147 | 29 | 0 | 11.0000 | 5.89 |
| 365 | 189 | 189 | 0 | 34 | 5 | 0 | 19.0000 | 2.07 |

And the Holm survivors by model and horizon (all losses to persistence):

| model | h=1 | h=7 | h=30 | h=90 | h=365 |
|---|---|---|---|---|---|
| elastic_net | 13 | 18 | 12 | 4 | 1 |
| feedforward_nn | 22 | 21 | 17 | 4 | 0 |
| gradient_boosting | 21 | 20 | 15 | 4 | 0 |
| gru | 18 | 19 | 14 | 5 | 0 |
| historical_mean | 0 | 0 | 0 | 0 | 1 |
| ols | 19 | 19 | 10 | 2 | 1 |
| random_walk_drift | 0 | 0 | 0 | 0 | 0 |
| ridge | 19 | 19 | 12 | 3 | 1 |
| svr_poly | 21 | 19 | 14 | 3 | 1 |
| svr_rbf | 21 | 21 | 19 | 4 | 0 |

Where the point estimate favoured a model at all (a negative statistic), it was 46 tests of
1,069, and the strongest of them was `random_walk_drift` on LDO at h=365:
statistic -1.575, raw p = 0.116, pooled n = 754 (n/h
2.07). That is a baseline, on the asset-horizon with the least data, and
it is nowhere near significant. **Nothing here beat persistence, so there is nothing to check for
leakage. The leak that would manifest as a model beating persistence is absent from the output;
that is evidence about the code as well as about the market.**

The h=365 survivors, with the pooled n each rests on (see the kernel caveat in §3):

| model | symbol | DM stat | raw p | Holm p | pooled n | n/h |
|---|---|---|---|---|---|---|
| elastic_net | UNI | 7.668 | 3.8e-14 | 3.2e-11 | 1,112 | 3.05 |
| ridge | UNI | 7.558 | 8.5e-14 | 7.2e-11 | 1,112 | 3.05 |
| ols | UNI | 6.661 | 4.3e-11 | 3.4e-08 | 1,112 | 3.05 |
| svr_poly | ALGO | 4.388 | 1.2e-05 | 0.0079 | 1,566 | 4.29 |
| historical_mean | MANA | 4.257 | 2.2e-05 | 0.0137 | 2,232 | 6.12 |

All five are persistence winning. The kernel changes the size of a statistic and not its sign, so
it cannot have manufactured a model win; it can have made a persistence win look more decisive, which
is why the borderline ones (the largest Holm p-values above) are the ones to discount. Full h=365 detail, per asset (pooled n is the largest across models, since a model that refused a
fold is compared on the folds it scored):

| symbol | pooled n | n/h | windows | raw p<0.05 | Holm p<0.05 | smallest raw p |
|---|---|---|---|---|---|---|
| BTC | 3,200 | 8.77 | 11.8 | 3 | 0 | 0.0006 |
| DOGE | 3,200 | 8.77 | 11.8 | 3 | 0 | 0.0038 |
| LTC | 3,200 | 8.77 | 11.8 | 0 | 0 | 0.0679 |
| ETH | 2,981 | 8.17 | 11.2 | 7 | 0 | 0.0002 |
| BNB | 2,273 | 6.23 | 9.3 | 0 | 0 | 0.0845 |
| BCH | 2,257 | 6.18 | 9.2 | 0 | 0 | 0.0650 |
| MANA | 2,232 | 6.12 | 9.1 | 3 | 1 | 0.0e+00 |
| LINK | 2,197 | 6.02 | 9 | 2 | 0 | 0.0269 |
| ADA | 2,135 | 5.85 | 8.9 | 2 | 0 | 0.0004 |
| MKR | 2,108 | 5.78 | 8.8 | 0 | 0 | 0.2339 |
| USDC | 1,833 | 5.02 | 8 | 2 | 0 | 0.0087 |
| ALGO | 1,566 | 4.29 | 7.3 | 2 | 1 | 0.0e+00 |
| SNX | 1,274 | 3.49 | 6.5 | 1 | 0 | 0.0013 |
| COMP | 1,204 | 3.3 | 6.3 | 0 | 0 | 0.0961 |
| CRV | 1,146 | 3.14 | 6.2 | 4 | 0 | 0.0057 |
| DOT | 1,142 | 3.13 | 6.2 | 0 | 0 | 0.1401 |
| UNI | 1,112 | 3.05 | 6.1 | 4 | 3 | 0.0e+00 |
| AVAX | 1,107 | 3.03 | 6.1 | 0 | 0 | 0.0817 |
| AAVE | 1,090 | 2.99 | 6 | 0 | 0 | 0.0689 |
| ICP | 877 | 2.4 | 5.4 | 0 | 0 | 0.1658 |
| LDO | 754 | 2.07 | 5.1 | 1 | 0 | 0.0053 |

## 5. Where the models differ from each other, if anywhere

Median MASE per fold, by model and horizon, the three baselines first:

| model | h=1 | h=7 | h=30 | h=90 | h=365 |
|---|---|---|---|---|---|
| naive_persistence | 0.743 | 0.716 | 0.688 | 0.697 | 0.608 |
| random_walk_drift | 0.748 | 0.734 | 0.732 | 0.744 | 0.834 |
| historical_mean | 0.748 | 0.735 | 0.728 | 0.778 | 0.761 |
| ols | 0.85 | 1.012 | 1.182 | 1.536 | 1.79 |
| ridge | 0.823 | 0.976 | 1.108 | 1.353 | 1.566 |
| elastic_net | 0.773 | 0.93 | 1.116 | 1.336 | 1.509 |
| svr_rbf | 1.076 | 1.199 | 1.136 | 1.079 | 0.879 |
| svr_poly | 1.683 | 1.99 | 2.467 | 1.967 | 1.817 |
| feedforward_nn | 2.319 | 1.474 | 2.115 | 1.717 | 1.387 |
| gradient_boosting | 0.953 | 1.102 | 1.052 | 1.102 | 1.073 |
| gru | 0.817 | 0.989 | 1.008 | 1.065 | refused |

Persistence has the lowest median MASE at every horizon. A model is above 1.0 in
a column when it lost to forecasting no return *by the training fold's own yardstick*:

- Above 1.0 at **every** horizon: `svr_poly`, `feedforward_nn`. At h=1 `feedforward_nn` scores 2.32 and `svr_poly` 1.68, which is 3.1x and 2.3x persistence's 0.74.
- Below 1.0 at some horizons only: `ols` at h=1; `ridge` at h=1, h=7; `elastic_net` at h=1, h=7; `svr_rbf` at h=365; `gradient_boosting` at h=1; `gru` at h=1, h=7. Below 1.0 does not mean it beat persistence: persistence is lower in each of those columns.
- At h=365 the non-baseline medians run from 0.88 (`svr_rbf`) to 1.82 (`svr_poly`), from 100-odd folds over 21 assets whose test blocks overlap in calendar time.

Directional accuracy, the median fold, excluding the persistence row (explained below). A coin flip
is 0.50:

| model | median_direction | folds | usdc_median |
|---|---|---|---|
| ridge | 0.5089 | 535 | 0.5069 |
| elastic_net | 0.5078 | 535 | 0.5098 |
| ols | 0.5077 | 535 | 0.5029 |
| feedforward_nn | 0.5032 | 535 | 0.4961 |
| random_walk_drift | 0.5026 | 545 | 0.4892 |
| svr_rbf | 0.5025 | 535 | 0.5117 |
| gru | 0.5025 | 432 | 0.5105 |
| svr_poly | 0.502 | 535 | 0.502 |
| historical_mean | 0.5019 | 545 | 0.5098 |
| gradient_boosting | 0.5 | 535 | 0.4932 |

No model's median fold is above 0.509. `usdc_median` is each model's
median on USDC, the stablecoin control group: it is also about 0.50, which is the control
behaving correctly (a dollar peg has no direction to find) and not a finding.

**Persistence's directional accuracy is not in that table on purpose.** The metric counts a row
only if the *realised* return has a sign, and a prediction of exactly 0.0 has no sign, so it is
scored as never right: `naive_persistence` carries a directional accuracy of exactly 0.0 on all
545 folds (usable rows 106 to 788),
not a NULL. That is correct for a forecast that makes no directional call, but it is not a score
to rank, and the 0.0 would otherwise sit at the bottom of any pooled table.

## 6. Declared gaps

One row per gap, each with its measurement.

| Spec | What is not here | Why |
|---|---|---|
| §11.3 | The recurrent model at h=365 | **The GRU is built and runs at four horizons (1, 7, 30, 90).** It refuses h=365, naming its measurement: Task 2 measured the assets, of 24, clearing both >=5 folds and >=30 non-overlapping windows as h=1: 22, h=7: 22, h=30: 22, h=90: 11, h=365: 0, against a threshold of 8 stated in advance. The mart carries 105 refused rows for it across 21 assets, each reading: *gru: h=365 lacks the independent observations a recurrent model needs; only 0 of 24 assets cleared >=5 folds and >=30 non-overlapping windows (measured 2026-10-17 on the 2026-10-14 panel vintage; qualifying horizons: [1, 7, 30, 90])* |
| §11.2 | Tuned hyperparameters | No inner cross-validation. An honest tuning loop needs a nested walk-forward inside each training fold, and Task 2's measurement shows the folds are not there to spare. Every model runs at the stated config values, so these are scores for *a* parameterisation, not for the best one. |
| §11.4 | ADF/KPSS, cointegration, VECM, GARCH, Markov-switching, DCC | Plan 4b. Needs `arch` plus a hand-rolled DCC estimator; answers a different question and writes a different table. |
| §11.5 | Factor models, Fama-MacBeth, decile portfolios | Plan 4c, which must first resolve the point-in-time top-N universe §5 and §11.5 both require and no free source serves. That point is **unresolved**. |
| §11.2 | TRX and SOL | TRX has 365 price rows (364 priced) and SOL 363 (CoinGecko only; Coin Metrics serves neither on the free tier). Both are below the 730-row floor for a five-fold purged split, and both are refused with those counts on every horizon (10 refusal rows). |

Two further limits belong here because they bound the result. **First**, the claim is about
*these* parameterisations and *these* 27 signals: a null here does not show that no valuation signal
forecasts returns, only that these did not, scored this way, against a forecast of zero. **Second**,
a single zero-forecast baseline is a hard bar for squared error on returns: any model that predicts
something nonzero pays variance for its prediction, and the tables above are consistent with that
cost exceeding any signal. That reading is a reading, not a test; Phase 4 does not separate bias
from variance.

---

### Appendix: every long-horizon DM p-value, with the pooled n it rests on

Each cell is `raw p (Holm p)` for that model against `naive_persistence`, and a DM statistic is
positive in every cell where the Holm p is below 0.05. Rows are ordered by pooled n/h, largest
first, so the thinnest samples are at the bottom. `pooled n` is a range when a model refused a fold
and was compared on fewer rows.

**h=365** (189 tests; blank = the model does not run at this cell):

| asset | pooled n | n/h | random_walk_drift | historical_mean | ols | ridge | elastic_net | svr_rbf | svr_poly | feedforward_nn | gradient_boosting |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LTC | 3,200 | 8.77 | 0.169 (1.000) | 0.180 (1.000) | 0.068 (1.000) | 0.120 (1.000) | 0.122 (1.000) | 0.570 (1.000) | 0.079 (1.000) | 0.112 (1.000) | 0.134 (1.000) |
| DOGE | 3,200 | 8.77 | 0.927 (1.000) | 0.852 (1.000) | 0.037 (1.000) | 0.076 (1.000) | 0.079 (1.000) | 0.112 (1.000) | 0.004 (1.000) | 0.041 (1.000) | 0.298 (1.000) |
| BTC | 3,200 | 8.77 | 0.889 (1.000) | 0.980 (1.000) | 0.001 (0.354) | 0.005 (1.000) | 0.006 (1.000) | 0.332 (1.000) | 0.087 (1.000) | 0.114 (1.000) | 0.823 (1.000) |
| ETH | 2,981 | 8.17 | 0.151 (1.000) | 0.223 (1.000) | 0.000 (0.124) | 0.000 (0.178) | 0.000 (0.150) | 0.036 (1.000) | 0.014 (1.000) | 0.041 (1.000) | 0.002 (1.000) |
| BNB | 1,819-2,273 | 6.23 | 0.085 (1.000) | 0.768 (1.000) | 0.326 (1.000) | 0.326 (1.000) | 0.326 (1.000) | 0.443 (1.000) | 0.224 (1.000) | 0.137 (1.000) | 0.090 (1.000) |
| BCH | 2,257 | 6.18 | 0.235 (1.000) | 0.323 (1.000) | 0.207 (1.000) | 0.955 (1.000) | 0.977 (1.000) | 0.322 (1.000) | 0.065 (1.000) | 0.167 (1.000) | 0.293 (1.000) |
| MANA | 2,232 | 6.12 | 0.886 (1.000) | 0.000 (0.014) | 0.206 (1.000) | 0.210 (1.000) | 0.213 (1.000) | 0.005 (1.000) | 0.000 (0.081) | 0.116 (1.000) | 0.224 (1.000) |
| LINK | 2,197 | 6.02 | 0.084 (1.000) | 0.342 (1.000) | 0.055 (1.000) | 0.055 (1.000) | 0.055 (1.000) | 0.333 (1.000) | 0.027 (1.000) | 0.196 (1.000) | 0.029 (1.000) |
| ADA | 2,135 | 5.85 | 0.257 (1.000) | 0.242 (1.000) | 0.206 (1.000) | 0.206 (1.000) | 0.206 (1.000) | 0.000 (0.253) | 0.210 (1.000) | 0.206 (1.000) | 0.002 (0.992) |
| MKR | 2,108 | 5.78 | 0.394 (1.000) | 0.906 (1.000) | 0.234 (1.000) | 0.326 (1.000) | 0.323 (1.000) | 0.497 (1.000) | 0.386 (1.000) | 0.358 (1.000) | 0.295 (1.000) |
| USDC | 1,833 | 5.02 | 0.009 (1.000) | 0.868 (1.000) | 0.255 (1.000) | 0.256 (1.000) | 0.868 (1.000) | 0.141 (1.000) | 0.113 (1.000) | 0.026 (1.000) | 0.168 (1.000) |
| ALGO | 1,566 | 4.29 | 0.491 (1.000) | 0.193 (1.000) | 0.156 (1.000) | 0.110 (1.000) | 0.110 (1.000) | 0.229 (1.000) | 0.000 (0.008) | 0.032 (1.000) | 0.087 (1.000) |
| SNX | 1,274 | 3.49 | 0.149 (1.000) | 0.644 (1.000) | 0.215 (1.000) | 0.217 (1.000) | 0.218 (1.000) | 0.959 (1.000) | 0.001 (0.672) | 0.378 (1.000) | 0.664 (1.000) |
| COMP | 1,204 | 3.30 | 0.990 (1.000) | 0.663 (1.000) | 0.256 (1.000) | 0.125 (1.000) | 0.118 (1.000) | 0.096 (1.000) | 0.400 (1.000) | 0.220 (1.000) | 0.117 (1.000) |
| CRV | 1,146 | 3.14 | 0.587 (1.000) | 0.801 (1.000) | 0.038 (1.000) | 0.006 (1.000) | 0.033 (1.000) | 0.490 (1.000) | 0.116 (1.000) | 0.018 (1.000) | 0.944 (1.000) |
| DOT | 914-1,142 | 3.13 | 0.140 (1.000) | 0.504 (1.000) | 0.718 (1.000) | 0.717 (1.000) | 0.717 (1.000) | 0.731 (1.000) | 0.549 (1.000) | 0.795 (1.000) | 0.683 (1.000) |
| UNI | 1,112 | 3.05 | 0.465 (1.000) | 0.512 (1.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.333 (1.000) | 0.273 (1.000) | 0.300 (1.000) | 0.039 (1.000) |
| AVAX | 1,107 | 3.03 | 0.082 (1.000) | 0.549 (1.000) | 0.250 (1.000) | 0.251 (1.000) | 0.253 (1.000) | 0.233 (1.000) | 0.192 (1.000) | 0.288 (1.000) | 0.217 (1.000) |
| AAVE | 1,090 | 2.99 | 0.069 (1.000) | 0.165 (1.000) | 0.178 (1.000) | 0.270 (1.000) | 0.249 (1.000) | 0.201 (1.000) | 0.138 (1.000) | 0.092 (1.000) | 0.134 (1.000) |
| ICP | 877 | 2.40 | 0.354 (1.000) | 0.572 (1.000) | 0.205 (1.000) | 0.261 (1.000) | 0.259 (1.000) | 0.508 (1.000) | 0.466 (1.000) | 0.201 (1.000) | 0.166 (1.000) |
| LDO | 754 | 2.07 | 0.116 (1.000) | 0.121 (1.000) | 0.527 (1.000) | 0.077 (1.000) | 0.080 (1.000) | 0.979 (1.000) | 0.307 (1.000) | 0.423 (1.000) | 0.005 (1.000) |

**h=90** (220 tests):

| asset | pooled n | n/h | random_walk_drift | historical_mean | ols | ridge | elastic_net | svr_rbf | svr_poly | feedforward_nn | gradient_boosting | gru |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTC | 3,750 | 41.67 | 0.853 (1.000) | 0.997 (1.000) | 0.000 (0.008) | 0.000 (0.000) | 0.000 (0.000) | 0.012 (1.000) | 0.001 (0.328) | 0.000 (0.001) | 0.000 (0.278) | 0.001 (0.490) |
| LTC | 3,750 | 41.67 | 0.165 (1.000) | 0.308 (1.000) | 0.001 (0.615) | 0.002 (0.796) | 0.002 (0.817) | 0.000 (0.046) | 0.003 (1.000) | 0.001 (0.451) | 0.001 (0.325) | 0.000 (0.085) |
| DOGE | 3,750 | 41.67 | 0.352 (1.000) | 0.866 (1.000) | 0.003 (1.000) | 0.000 (0.259) | 0.001 (0.305) | 0.001 (0.571) | 0.003 (1.000) | 0.001 (0.492) | 0.000 (0.000) | 0.000 (0.000) |
| ETH | 3,531 | 39.23 | 0.802 (1.000) | 0.736 (1.000) | 0.000 (0.003) | 0.000 (0.000) | 0.000 (0.001) | 0.113 (1.000) | 0.037 (1.000) | 0.000 (0.000) | 0.734 (1.000) | 0.567 (1.000) |
| BNB | 2,259-2,823 | 31.37 | 0.056 (1.000) | 0.380 (1.000) | 0.716 (1.000) | 0.716 (1.000) | 0.717 (1.000) | 0.033 (1.000) | 0.054 (1.000) | 0.002 (1.000) | 0.001 (0.473) | 0.700 (1.000) |
| BCH | 2,807 | 31.19 | 0.674 (1.000) | 0.455 (1.000) | 0.007 (1.000) | 0.000 (0.007) | 0.000 (0.004) | 0.008 (1.000) | 0.014 (1.000) | 0.018 (1.000) | 0.000 (0.001) | 0.000 (0.022) |
| MANA | 2,782 | 30.91 | 0.023 (1.000) | 0.016 (1.000) | 0.003 (1.000) | 0.003 (1.000) | 0.003 (1.000) | 0.077 (1.000) | 0.003 (1.000) | 0.009 (1.000) | 0.034 (1.000) | 0.003 (1.000) |
| LINK | 2,747 | 30.52 | 0.153 (1.000) | 0.277 (1.000) | 0.000 (0.060) | 0.000 (0.063) | 0.000 (0.069) | 0.036 (1.000) | 0.004 (1.000) | 0.002 (0.822) | 0.000 (0.020) | 0.003 (1.000) |
| ADA | 2,685 | 29.83 | 0.022 (1.000) | 0.026 (1.000) | 0.012 (1.000) | 0.012 (1.000) | 0.012 (1.000) | 0.000 (0.006) | 0.024 (1.000) | 0.012 (1.000) | 0.000 (0.266) | 0.000 (0.000) |
| MKR | 2,658 | 29.53 | 0.139 (1.000) | 0.233 (1.000) | 0.165 (1.000) | 0.016 (1.000) | 0.019 (1.000) | 0.056 (1.000) | 0.129 (1.000) | 0.052 (1.000) | 0.006 (1.000) | 0.007 (1.000) |
| USDC | 2,383 | 26.48 | 0.492 (1.000) | 0.840 (1.000) | 0.014 (1.000) | 0.015 (1.000) | 0.840 (1.000) | 0.000 (0.005) | 0.073 (1.000) | 0.002 (1.000) | 0.009 (1.000) | 0.004 (1.000) |
| ALGO | 2,116 | 23.51 | 0.140 (1.000) | 0.056 (1.000) | 0.004 (1.000) | 0.022 (1.000) | 0.021 (1.000) | 0.001 (0.439) | 0.000 (0.000) | 0.000 (0.001) | 0.018 (1.000) | 0.000 (0.009) |
| SNX | 1,824 | 20.27 | 0.016 (1.000) | 0.025 (1.000) | 0.211 (1.000) | 0.115 (1.000) | 0.107 (1.000) | 0.000 (0.079) | 0.000 (0.011) | 0.000 (0.112) | 0.087 (1.000) | 0.001 (0.311) |
| COMP | 1,754 | 19.49 | 0.074 (1.000) | 0.062 (1.000) | 0.088 (1.000) | 0.093 (1.000) | 0.094 (1.000) | 0.002 (0.858) | 0.011 (1.000) | 0.024 (1.000) | 0.207 (1.000) | 0.018 (1.000) |
| CRV | 1,696 | 18.84 | 0.497 (1.000) | 0.041 (1.000) | 0.001 (0.599) | 0.000 (0.145) | 0.000 (0.008) | 0.000 (0.039) | 0.000 (0.027) | 0.000 (0.000) | 0.001 (0.532) | 0.012 (1.000) |
| DOT | 1,354-1,692 | 18.80 | 0.031 (1.000) | 0.044 (1.000) | 0.981 (1.000) | 0.982 (1.000) | 0.979 (1.000) | 0.020 (1.000) | 0.171 (1.000) | 0.027 (1.000) | 0.018 (1.000) | 0.314 (1.000) |
| UNI | 1,662 | 18.47 | 0.098 (1.000) | 0.065 (1.000) | 0.002 (1.000) | 0.002 (1.000) | 0.003 (1.000) | 0.005 (1.000) | 0.002 (1.000) | 0.000 (0.274) | 0.001 (0.466) | 0.003 (1.000) |
| AVAX | 1,657 | 18.41 | 0.051 (1.000) | 0.054 (1.000) | 0.021 (1.000) | 0.019 (1.000) | 0.020 (1.000) | 0.001 (0.774) | 0.054 (1.000) | 0.013 (1.000) | 0.002 (0.792) | 0.000 (0.072) |
| AAVE | 1,640 | 18.22 | 0.052 (1.000) | 0.057 (1.000) | 0.032 (1.000) | 0.000 (0.160) | 0.000 (0.123) | 0.010 (1.000) | 0.000 (0.077) | 0.000 (0.083) | 0.000 (0.052) | 0.000 (0.017) |
| ICP | 1,427 | 15.86 | 0.033 (1.000) | 0.152 (1.000) | 0.009 (1.000) | 0.026 (1.000) | 0.027 (1.000) | 0.002 (0.888) | 0.000 (0.153) | 0.002 (1.000) | 0.000 (0.007) | 0.132 (1.000) |
| LDO | 1,304 | 14.49 | 0.885 (1.000) | 0.996 (1.000) | 0.108 (1.000) | 0.000 (0.193) | 0.000 (0.160) | 0.004 (1.000) | 0.006 (1.000) | 0.000 (0.071) | 0.003 (1.000) | 0.006 (1.000) |
| POL | 530 | 5.89 | 0.527 (1.000) | 0.616 (1.000) | 0.443 (1.000) | 0.439 (1.000) | 0.438 (1.000) | 0.581 (1.000) | 0.209 (1.000) | 0.798 (1.000) | 0.977 (1.000) | 0.818 (1.000) |

---

### Reproducing this chapter

```bash
python -m cryptoval.cli score                # 11 models x 22 foldable assets x 5 horizons, ~8 min
make dbt                                     # fct_model_score and its uniqueness test
```

`make all` now runs `score` between the two `dbt` builds, as a recipe, because `make` runs a
prerequisite listed twice only once.
