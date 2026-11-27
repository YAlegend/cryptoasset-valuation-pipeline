# The Anchoring Chain — Design (Plan 4b)

Spec §11.4 lists six econometric families. This design covers the first three only —
**ADF/KPSS → Engle-Granger/Johansen → VECM** — and records the four decisions §11.4 does not
contain. GARCH(1,1)/EGARCH, Markov-switching and DCC-GARCH are a later plan: they answer a
different question, write a different table, and need the `arch` package this one does not.

## 1. The question

**Does market cap hold a long-run equilibrium with a fundamental?**

§11.4 calls this "the correct test for 'does MVRV/NVT/Metcalfe anchor price' — far stronger than
the correlation evidence", and it is the question the dissertation's correlation chapter could not
reach. Plan 4 answered a different one — *does a valuation signal forecast returns out of sample*
— and answered it null. This chapter does not revisit that.

## 2. Decision: the fundamental form, not the ratio form

**The three series §11.4 names are all price-derived.** In this codebase
`mvrv = market_cap_usd / realized_cap` and `market_cap_usd = price_usd × supply`, so
`log(MVRV) = log(price) + log(supply) − log(realized_cap)`. NVT is
`market_cap_usd / volume_total_usd`, the same shape. And `metcalfe_estimated` is **fitted to
price** by `fit_metcalfe`. Testing whether price holds a long-run equilibrium with a series that
algebraically contains price, or was regressed on it, is close to circular.

This project has already been bitten by this shape. Plan 3 found `velocity_total` was identically
`volume_total_usd / market_cap_usd`, which made the Kampakis log-linear design matrix
rank-deficient and recovered `price = market cap / supply` — a result that looked like an estimate
and was an identity.

So the test is run in the **fundamental form**, against the fundamental itself:

| §11.4's question | Pair tested |
|---|---|
| Does MVRV anchor price? | `log(market_cap_usd)` ~ `log(realized_cap)` |
| Does NVT anchor price? | `log(market_cap_usd)` ~ `log(volume_total_usd)` |
| Does Metcalfe's law hold? | `log(market_cap_usd)` ~ `log(active_addresses)` |

The cointegrating coefficient is **estimated, not assumed**. For the Metcalfe pair this is the
substantive gain: Metcalfe's law predicts an exponent of 2, so the coefficient's confidence
interval is a falsifiable test of the law rather than a goodness-of-fit statistic for a line
already fitted to price.

**The ratio tests are retained as the constrained case.** A stationary `log` ratio *is* a
cointegrating relationship with the coefficient forced to the value the ratio assumes, so ADF/KPSS
on the ratios and Engle-Granger on the levels are the constrained and unconstrained versions of one
question. The constraint differs per pair, and the ratio must match the law it encodes:

| Pair | Ratio tested | Coefficient it forces |
|---|---|---|
| MVRV | `market_cap_usd / realized_cap` | 1 |
| Liquidity | `market_cap_usd / volume_total_usd` | 1 |
| Metcalfe | `market_cap_usd / active_addresses²` | **2**, the exponent the law states |

Reporting both says whether an equilibrium exists *and* whether the coefficient the practitioner
literature assumes is the one the data supports. For the Metcalfe pair the estimated coefficient's
confidence interval is checked against 2 explicitly — that comparison is the falsifiable test, and
`market_cap_usd / active_addresses` (the coefficient-1 ratio) is **not** Metcalfe's law and is not
tested as though it were.

## 3. Decision: full sample, and labelled as such

One ADF/KPSS, one Engle-Granger/Johansen and one VECM per pair over the whole available history.
This is how the literature reports cointegration and it is what the dissertation lacked.

It also uses future data by construction. The chapter must therefore state plainly that:

* the claim is **descriptive, about the sample** — not a forecast;
* it supports **no tradability claim** whatsoever;
* it is **not comparable** to Plan 4's out-of-sample null, and a finding here does not soften that
  null.

None of Plan 4's splitting machinery is reused. There are no folds, no purge, no embargo, because
there is no out-of-sample claim to protect.

## 4. Decision: the universe is measured, not assumed

`realized_cap` and `mvrv` bind the universe, not price history. Measured on the 2026-10-14 panel
vintage (24 configured assets):

* 20 assets carry 1,000+ days of both price and MVRV;
* **DOT** (653) and **BNB** (647) are marginal;
* **TRX** has no MVRV at all; **SOL** has 363 rows and nothing usable.

The plan must **re-measure this** rather than hard-code 20: a re-ingest moves it. Every exclusion
is a refusal row carrying its own count, as in Plan 4. A minimum-row floor for a cointegration test
is set in the plan with the measurement behind it, not inherited from Plan 4's 730.

## 5. Architecture

A new `cryptoval/econometrics/` package, mirroring the layering of `cryptoval/scoring/`:

| Module | Responsibility |
|---|---|
| `series.py` | Build the levels panel: `log(market_cap_usd)` and the three fundamentals, per symbol. Reuses Plan 4's contiguity guard rather than carrying a second copy. |
| `stationarity.py` | ADF and KPSS, each refusing its degenerate case with the measurement. |
| `cointegration.py` | Engle-Granger and Johansen, returning the estimated coefficient with its interval. |
| `vecm.py` | VECM fit and the speed-of-adjustment term — the §11.4 column that says how fast reversion happens. |
| registry | Pairs are config-driven in `config/econometrics/*.yml`, following `scoring/registry.py`. A pair is data, not a branch in code. |

`cryptoval.cli econ` writes gold; a `fct_econometric_test` mart reads it. `correct_family_wise` is
reused from `cryptoval/scoring/multiplicity.py`, **generalized by three keyword arguments whose
defaults are Plan 4's exact behaviour** (`p_metric`, `columns`, `count_keys`) — it hardcodes
`dm_p_value` and `SCORE_COLUMNS` today, so it cannot be reused literally unchanged. One
implementation rather than two matters because the guard that refuses a second pass over adjusted
p-values lives in it, and `tests/test_cli.py` passing untouched is the proof the generalization
changed nothing for Plan 4.

**No new dependency.** `statsmodels==0.15.0` is already pinned and carries `adfuller`, `kpss`,
`coint`, `coint_johansen` and `VECM`.

## 6. Table shape

One long table in the shape of `fct_model_score`: one row per (pair, symbol, test, statistic),
with refusals as rows whose `reason` names the measurement that caused them, and
`panel_vintage_ts` on every row.

One `TEST_COLUMNS` constant, and — unlike Plan 4 — **asserted against the mart's SQL select list**
from the first task. The whole-branch review of Plan 4 found that nothing tied `SCORE_COLUMNS` to
`fct_model_score.sql`, so a column added to the constant would reach gold and never reach the mart.
This plan does not repeat that.

## 7. Multiple testing

Per §11.6, the headline claims enter one family-wise correction. **The family is the cointegration
tests**: three pairs × the measured asset count, so roughly 60 hypotheses — small enough that Holm
leaves real power, which is why the design tests three fundamentals rather than all 28 registry
series.

Two things are deliberately **outside** the family, and the chapter must say so rather than leave it
to be inferred:

* **ADF/KPSS are pre-tests, not claims.** They determine which regressions are valid at all
  (§11.4's own framing). Counting specification checks as hypotheses would inflate the correction
  and make the claims harder to reject than the evidence warrants — the same error
  `correct_family_wise` already refuses when it excludes a refused DM test from the count.
* **VECM coefficients are estimates, not hypotheses.** A VECM is fitted only where cointegration
  survives the correction, so its speed-of-adjustment term describes a relationship already
  established; it is reported with its interval, not with a corrected p-value.

The hypothesis count is written to the table as a row, as Plan 4 does, so a later plan can
re-correct over a wider family from the raw p-values.

## 8. Plan 4 carry-overs

The Plan 4 review triaged three items into this plan's scope:

1. **`param_set_hash` is dead code.** `fct_model_score` records no parameterisation, so re-running
   `score` after editing a model's YAML overwrites the old rows indistinguishably. Decision: give
   the new mart a `params_hash` from the start, and **delete** `param_set_hash` from
   `cryptoval/scoring/registry.py` rather than retrofitting it. Deleting dead code beats wiring it
   on speculation; if Plan 4's mart needs provenance later, this plan will have proved the pattern.
2. **`SCORE_COLUMNS` is not tested against the mart SQL.** Fixed for both marts — see §6.
3. **`features_used` is computed per fold and discarded** by `pd.concat`. Deferred again, named:
   it belongs to Plan 4's table, and persisting it does not serve this chapter.

## 9. Declared gaps

Stated here so the plan carries them into the chapter rather than discovering them at review:

* **Structural breaks.** A full-sample Engle-Granger over 2014–2026 can miss a regime shift, and no
  rolling or recursive re-test is in scope. Gregory-Hansen (cointegration with an endogenous break)
  is the named alternative, not built.
* **`volume_total_usd` is exchange trading volume, not on-chain transfer value.** The method card
  for `nvt_golden_ratio` already declares this objection. That pair therefore tests market cap
  against market liquidity, and the chapter must not call it NVT without saying so.
* **Bivariate pairs only.** A joint Johansen over market cap and all three fundamentals at once is
  a reasonable test and is not built; interpreting a three-vector equilibrium is a different
  chapter's work.
* **No tuned lag selection beyond an information criterion.** The lag order comes from AIC/BIC with
  the choice recorded, not from a search.

## 10. Testing strategy

Plan 4's discipline, unchanged:

* every statistic **refuses its degenerate case** with the measurement, rather than returning a
  meaningless number;
* **known-answer tests**: a synthetic cointegrated pair the test must find, and two independent
  random walks it must not. Both directions, so the tests can fail;
* **mutation proof per task** — a test that cannot fail is not coverage. The Plan 4 review found
  five vacuous tests and proved each with a surviving mutation; this plan assumes it will generate
  its own and checks for them per task rather than at the end.

## 11. Out of scope

GARCH(1,1)/EGARCH, Markov-switching, DCC-GARCH (§11.4's second chain, a later plan); everything in
§11.5 (factor models, Fama-MacBeth, decile portfolios — still blocked on the point-in-time top-N
universe); any revisit of Plan 4's forecasting null.
