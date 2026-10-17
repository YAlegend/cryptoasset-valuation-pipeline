# Does price anchor to fundamentals? — Phase 4b

**These are valuation estimates under stated assumptions, not investment advice. These assets are
volatile and these methods disagree by design.**

Run of 2026-10-17 against `data/cryptoval.duckdb`, panel vintage `2026-10-14 18:20:40.522565+01:00`.
Panel: `fct_asset_metrics_daily`, 69,264 symbol-days, 24 symbols, 2015-01-01 to 2026-10-14.
Results: `fct_econometric_test`, 1,215 rows, written by `python -m cryptoval.cli econ`. Every
number below is a query against that table and is reproducible with the commands in the last
section.

**The result, in one sentence: market cap holds a long-run equilibrium with realized cap on 9 of
20 testable assets and with traded volume on 2 of 18 after a Holm correction over 56 tests, and
with active addresses on none — but only 6 of those 11 rest on series that pass both unit-root
pre-tests, and in every one of the 6 the equilibrium is slow, with a half life of two months or
more where it can be measured at all.** Metcalfe's law could not be tested as a law, because
no asset has the equilibrium its exponent would describe.

**This is a full-sample, descriptive result. It uses future data by construction, supports no
tradability claim, and is not comparable to Phase 4's out-of-sample null** (`reports/does-anything-beat-a-random-walk.md`).
Nothing here softens that null: an equilibrium visible over nearly twelve years of hindsight is not a
forecast anyone could have traded.

---

## 1. What this tests, and what it cannot

Spec §11.4 asks whether MVRV, NVT and Metcalfe "anchor price" and calls cointegration the correct
test, "far stronger than the correlation evidence". Taken literally the question is close to
circular. `mvrv = market_cap_usd / realized_cap` and `market_cap_usd = price_usd × supply`, so
`log(MVRV)` contains `log(price)`; NVT is the same shape; and `metcalfe_estimated` is fitted *to*
price. Testing whether price holds an equilibrium with a series built from price tests an identity.
Phase 3 shipped exactly that once: `velocity_total` was identically
`volume_total_usd / market_cap_usd`, and the Kampakis regression recovered
`price = market cap / supply` while looking like an estimate.

So each question is asked in the **fundamental form**, against the fundamental itself, with the
coefficient **estimated rather than assumed**:

| §11.4's question | Pair | Ratio form, and the coefficient it forces |
|---|---|---|
| Does MVRV anchor price? | `log(market_cap_usd)` ~ `log(realized_cap)` | `market_cap / realized_cap`, 1 |
| Does NVT anchor price? | `log(market_cap_usd)` ~ `log(volume_reported_spot_usd)` | `market_cap / volume`, 1 |
| Does Metcalfe's law hold? | `log(market_cap_usd)` ~ `log(active_addresses)` | `market_cap / active_addresses²`, **2** |

A stationary log ratio *is* a cointegrating relationship with the coefficient forced, so the ratio
tests (§3) and Engle-Granger on the levels (§4) are the constrained and unconstrained versions of
one hypothesis. Reporting both says whether an equilibrium exists and whether the coefficient the
practitioner literature assumes is the one the data supports.

**The liquidity pair is not NVT** (§7), and it reads Coin Metrics' volume, not CoinGecko's.
CoinGecko's `volume_total_usd` covers 363 to 366 days on every asset, all starting 2025-10-14: the
free tier's one-year window. That is below the 730-row floor, so the first run refused the pair on
all 24 assets. Coin Metrics' `volume_reported_spot_usd` is the same quantity under a different
construction and clears 730 rows on 23 assets. The two are never coalesced.

Each pair gets one ADF/KPSS, one Engle-Granger, one Johansen and, where it survives, one VECM over
its whole available history. There are no folds, no purge and no embargo, because there is no
out-of-sample claim to protect.

## 2. The universe, measured

A pair is tested on the dates where **both** of its series are finite. The intersection must be at
least 730 contiguous daily rows, and it is refused, with its count, if it has an interior gap.
Gaps are refused rather than stitched: a stitched series makes a one-day lag mean one thing before
the hole and another after it.

| Pair | Assets | Tested | Refused |
|---|---|---|---|
| `mvrv_anchor` | 24 | 20 | 4 |
| `liquidity_anchor` | 24 | 18 | 6 |
| `metcalfe_anchor` | 24 | 18 | 6 |

Every refusal, with the measurement it carries:

| Pair | Asset | Reason |
|---|---|---|
| all three | BNB | 647 rows where both series are finite; 730 needed |
| all three | DOT | 653 rows; 730 needed |
| all three | SOL | 0 rows (no Coin Metrics series on the free tier) |
| `mvrv_anchor` | TRX | 0 rows: no `realized_cap` |
| `liquidity_anchor`, `metcalfe_anchor` | TRX | 364 rows; 730 needed |
| `liquidity_anchor` | MKR | a 5-day step into 2026-01-19 (4 missing days) |
| `liquidity_anchor` | USDC | a 2-day step into 2026-05-06 (1 missing day) |
| `metcalfe_anchor` | AVAX | a 2-day step into 2020-09-27 (1 missing day) |
| `metcalfe_anchor` | MANA | a 9-day step into 2017-09-15 (8 missing days) |

Four of those refusals discard thousands of good rows over one to eight missing days. That is the
stated rule, and a longest-contiguous-segment variant would recover them; it is not built (§7).

The tested intersections run from 812 rows (POL, liquidity) to 4,304 (BTC, DOGE, LTC).

## 3. Stationarity, as a pre-test

Cointegration is a statement about two **I(1)** series, non-stationary in levels and stationary in
differences, that share a stochastic trend. A cointegration p-value computed where either series
is not I(1) does not mean what it says. So both series of every pair are pre-tested, and so is the
constrained ratio.

ADF and KPSS have **opposite nulls**. ADF's null is a unit root; KPSS's is stationarity. Read
together they give four verdicts:

| ADF rejects? | KPSS rejects? | Verdict |
|---|---|---|
| no | yes | **I(1)**: evidence of a unit root from both sides |
| yes | no | **I(0)**: stationary |
| yes | yes | **conflict**: each test rejects its own null |
| no | no | inconclusive |

A conflict is the typical signature of a near-unit-root, trend-stationary or long-memory series.
It is not evidence for I(1), and it is reported as what it is.

| Series | I(1) | conflict | I(0) |
|---|---|---|---|
| `log(market_cap_usd)`, per asset (20 assets in some pair) | 14 | 6 | 0 |
| `log(realized_cap)`, the 20 MVRV assets | 12 | 6 | 2 |
| `log(volume)`, the 18 liquidity assets | 5 | 12 | 1 |
| `log(active_addresses)`, the 18 Metcalfe assets | 9 | 9 | 0 |
| MVRV ratio | 2 | 18 | 0 |
| Liquidity ratio | 1 | 15 | 2 |
| Metcalfe ratio, `market_cap / addresses²` | 14 | 4 | 0 |

**The constrained ratios are almost never cleanly stationary.** The Metcalfe ratio is I(1) on 14 of
18 assets: `market_cap / active_addresses²` wanders without bound, which is the ratio-form evidence
against Metcalfe's law. The MVRV ratio is a conflict on 18 of 20: ADF rejects a unit root while
KPSS rejects stationarity, so the practitioner's "MVRV mean-reverts" is neither confirmed nor
refuted by these tests.

Two cautions on the numbers themselves. **KPSS p-values are clipped** to [0.01, 0.10], the range of
statsmodels' lookup table: a KPSS p of 0.01 in the table means "at most 0.01", and 0.10 means "at
least 0.10". And these pre-tests are **not in the Holm family**. They decide which regressions are
valid; counting a specification check as a hypothesis would inflate the correction and make the
claims harder to reject than the evidence warrants.

## 4. Cointegration

Engle-Granger on the levels, `log(market_cap) = c + β·log(fundamental) + u`, testing whether `u`
is stationary. **The Holm-Bonferroni family is these 56 tests and only these** (20 + 18 + 18). The
hypothesis count is written to the table as the row with `pair_id = '*'`.

| Pair | Tested | Raw p < 0.05 | Holm p < 0.05 |
|---|---|---|---|
| `mvrv_anchor` | 20 | 14 | **9** |
| `liquidity_anchor` | 18 | 7 | **2** |
| `metcalfe_anchor` | 18 | 2 | **0** |
| total | 56 | 23 | **11** |

The 11 survivors, each with its pre-test verdicts. β is the estimated coefficient with its standard
error, and Johansen is the trace test's verdict at rank 0:

| Pair | Asset | n | Holm p | β (se) | `log mcap` | fundamental | Johansen |
|---|---|---|---|---|---|---|---|
| MVRV | AAVE | 2,195 | 4.4e-26 | 1.245 (0.083) | I(1) | conflict | rank 1 |
| MVRV | LDO | 1,859 | 4.1e-08 | 1.532 (0.187) | I(1) | I(1) | **rank 0** |
| MVRV | ICP | 1,982 | 6.2e-06 | 0.891 (0.138) | conflict | I(1) | rank 1 |
| MVRV | COMP | 2,309 | 7.1e-05 | 1.334 (0.022) | I(1) | I(1) | rank 1 |
| MVRV | USDC | 2,938 | 2.6e-04 | 1.026 (0.001) | conflict | conflict | rank 1 |
| MVRV | AVAX | 2,212 | 7.0e-04 | 0.929 (0.008) | conflict | conflict | rank 1 |
| MVRV | MKR | 3,012 | 1.1e-03 | 0.820 (0.033) | I(1) | I(1) | rank 1 |
| MVRV | ADA | 3,239 | 3.2e-03 | 1.272 (0.024) | I(1) | I(1) | rank 1 |
| liquidity | LINK | 3,302 | 9.6e-03 | 0.735 (0.012) | I(1) | I(1) | rank 1 |
| MVRV | CRV | 2,251 | 1.2e-02 | 1.110 (0.017) | conflict | conflict | rank 1 |
| liquidity | LTC | 4,304 | 2.8e-02 | 0.532 (0.006) | I(1) | I(1) | rank 1 |

**Only 6 of the 11 rest on two cleanly I(1) series**: LDO, COMP, MKR and ADA (MVRV), and LINK and
LTC (liquidity). The other five are reported with that caveat attached. A conflict verdict on
either series means the cointegration p-value is computed on a premise the data does not clearly
support.

**USDC's survival is the circularity this design exists to catch, not a finding.** A $1 peg makes
`market_cap ≈ supply` and `realized_cap ≈ supply` by construction, so their log ratio is pinned
near zero and β is 1.026 with a standard error of 0.001. It is reported because it was tested, and
it should not be counted.

**BTC and ETH do not survive** on any pair. BTC's MVRV raw p is 0.121; ETH's is 0.025 raw and 0.909
after Holm.

### Johansen, beside Engle-Granger

Johansen's trace test rejects rank 0 on **51 of 56** pairs, against Engle-Granger's 11. The two
disagree on 42 pairs, and the agreeable reading, that Johansen confirms everything, is wrong.

Johansen's rank counts *any* stationary linear combination of the two series, including one that
puts all its weight on a single stationary series. Where the fundamental is I(0) or a conflict,
which §3 shows is most of the volume and half the address series, Johansen finds "rank 1" from that
series alone. It is also not exactly sized here. With the constant unrestricted
(`det_order=0`), its critical values assume the walks drift. Measured on independent random walks
(n = 2,500), it rejects 11.5% at drift/sd 0, 9.5% at 0.02 and about 6% at 0.05 and above, and
this panel's median drift/sd for log market cap is 0.018. **Engle-Granger, measured at 5% to 7%
over the same range, carries the claim**, and a Johansen "rank 1" here is not confirmation.

The one survivor Johansen rejects is LDO on MVRV, rank 0 with a trace statistic of 13.95 against a
95% critical value of 15.49.

### The coefficient against the ratio's assumption

The MVRV ratio assumes β = 1. The median estimated β over the 20 MVRV assets is **0.992**, but the
95% interval contains 1 on only **2** of them (ICP and UNI). The liquidity ratio also assumes 1;
the median β is **0.595**, and no interval contains 1. On the liquidity pair, then, the ratio
practitioners use is the wrong constraint even where an equilibrium exists. LTC is the clean
example: cointegrated at β = 0.532, while its forced ratio is I(1).

**The intervals are somewhat narrow.** β and its standard error come from Stock-Watson dynamic OLS
(±2 leads and lags of the differenced regressor) with a Newey-West HAC covariance, not from the
static regression's OLS standard error, which is invalid here. But the HAC bandwidth is a rule of
thumb, 4(n/100)^(2/9). On a simulated equilibrium error with autocorrelation 0.95 it recovered a
2.4× inflation of the naive standard error against a true ≈6×. On persistent equilibrium errors,
then, an interval excluding 1 is weaker evidence than its width suggests.

## 5. Metcalfe's exponent

This was the one place the design predicted a number in advance: Metcalfe's law says market cap is
proportional to the square of the users, so β on log active addresses should be **2**.

**No Metcalfe pair cointegrates** (0 of 18 after Holm; the smallest Holm p is ICP's 0.067). Without
cointegration the levels regression is a **spurious regression**, and its β and interval do not
estimate an exponent of anything. So the observation that 0 of 18 intervals contain 2 is **not** a
falsification of Metcalfe's law, and is not reported as one. For the record, the β's run from
−0.46 (POL) to 4.16 (BTC), with a median near 0.8.

The evidence against the law here is the ratio form in §3: `market_cap / active_addresses²` is I(1)
on 14 of 18 assets and cleanly stationary on none. The constrained version of the law finds no
equilibrium, and the unconstrained version finds no equilibrium to put an exponent on.

## 6. Speed of adjustment

A VECM is fitted **only where Engle-Granger survived Holm**. On the 45 pairs that did not, the table
holds one `vecm_alpha` row with a NULL value and a reason naming the adjusted p-value. A speed of
return to an equilibrium the correction rejected would read as a finding.

α is market cap's share of yesterday's disequilibrium corrected today. Negative is reversion,
positive divergence. The half life, log(0.5) / log|1 + α| days, exists only for −2 < α < 0, and is
reported only there. The VAR lag order is the AIC's choice.

| Pair | Asset | α | se | α / se | Half life (days) |
|---|---|---|---|---|---|
| liquidity | LTC | −0.0098 | 0.0021 | −4.6 | 71 |
| MVRV | ICP | −0.0052 | 0.0015 | −3.4 | 132 |
| MVRV | MKR | −0.0057 | 0.0030 | −1.9 | 122 |
| MVRV | COMP | −0.0065 | 0.0043 | −1.5 | 106 |
| MVRV | LDO | −0.0037 | 0.0027 | −1.4 | 185 |
| MVRV | CRV | −0.0073 | 0.0078 | −0.9 | 94 |
| MVRV | ADA | −0.0023 | 0.0027 | −0.9 | 297 |
| liquidity | LINK | −0.0022 | 0.0026 | −0.9 | 309 |
| MVRV | USDC | −0.0015 | 0.0069 | −0.2 | 457 |
| MVRV | AVAX | +0.0021 | 0.0038 | +0.5 | none: positive |
| MVRV | AAVE | **+0.0127** | 0.0041 | **+3.1** | none: **diverges** |

**Where market cap reverts at all, it reverts slowly.** Only two α's are both negative and more than
two standard errors from zero: LTC on liquidity (half life 71 days) and ICP on MVRV (132 days).
Every other half life rests on an α that is not distinguishable from zero, and should be read as
"no measurable adjustment by market cap". AAVE's market cap moves *away* from its realized-cap
equilibrium, significantly so: whatever restores AAVE's equilibrium, it is not AAVE's price.

These α's describe **market cap's** adjustment only. In a cointegrated pair the fundamental can do
the adjusting instead, and an α near zero on market cap is consistent with that. The fundamental's
own α is not reported (§7).

## 7. Declared gaps

One row per gap, each with its measurement.

| Spec | What is not here | Why, and what it costs |
|---|---|---|
| §11.4 | Structural breaks | One full-sample Engle-Granger over up to 4,304 days (2015 to 2026) can miss a regime shift entirely, or report one equilibrium where there were two. No rolling or recursive re-test is in scope. Gregory-Hansen, cointegration with one endogenous break, is the named alternative and is not built. |
| §11.4 | NVT, as the literature means it | `volume_reported_spot_usd` is **exchange trading volume**, not on-chain transfer value; the `nvt_golden_ratio` method card already declares this objection. The liquidity pair tests market cap against market liquidity. USD volume is also units traded × price, so this pair, like realized cap, is not wholly free of price. |
| §11.4 | A joint system | Bivariate pairs only. A joint Johansen over market cap and all three fundamentals at once is a reasonable test and is not built; interpreting a three-vector equilibrium is a different chapter's work. |
| §11.4 | A searched lag order | The VAR lag order is AIC's choice up to a cap of 14 days, and **AIC chose the cap on 25 of 56 pairs**, so on those it wanted more lags than it was allowed. ADF's lag is also AIC's choice. Nothing is tuned beyond that. |
| §11.4 | The fundamental's adjustment | Only market cap's α is reported (§6). A pair where the fundamental does all the adjusting shows α ≈ 0 here and looks like "no reversion". |
| §11.4 | Four testable assets, discarded over short gaps | AVAX and MANA (Metcalfe), MKR and USDC (liquidity) are refused over gaps of 1 to 8 days (§2). A test on the longest contiguous segment would recover them. Not built: the rule is "refuse, never stitch", and choosing a segment is a decision the design did not make. |
| §11.4 | GARCH, Markov-switching, DCC | §11.4's second chain. It answers a different question, writes a different table and needs `arch`; a later plan. |
| §11.5 | Factor models, Fama-MacBeth, decile portfolios | Still blocked on the point-in-time top-N universe that §5 and §11.5 both require and no free source serves. |

Two further limits belong here because they bound the result. **First**, every number is in-sample
over the whole history. An equilibrium found this way says the two series did not drift apart *over
2015 to 2026*, not that anyone standing in 2018 could have known they would not. **Second**, the
claims rest on Engle-Granger's choice of market cap as the dependent variable. Johansen is the
symmetric check, and §4 shows why it cannot do that job on this panel.

---

### Reproducing this chapter

```bash
python -m cryptoval.cli econ                 # 3 pairs x 24 assets, ~20 s
cd dbt && ../.venv/bin/dbt build --select fct_econometric_test --profiles-dir . && cd ..
```

`make all` now runs `econ` after `score` and before the last `dbt` build.

```sql
-- the hypothesis count the correction was taken over (56)
select value, reason from fct_econometric_test where metric = 'hypotheses_tested';

-- tested and refused per pair
select pair_id, count(distinct symbol) as assets,
       count(distinct case when metric = 'refused' then symbol end) as refused,
       count(distinct case when metric = 'cointegration_p_value' then symbol end) as tested
from fct_econometric_test where pair_id <> '*' group by 1 order by 1;

-- every refusal and its measurement
select pair_id, symbol, reason from fct_econometric_test where metric = 'refused' order by 1, 2;

-- the survivors
select pair_id, symbol, value as holm_p from fct_econometric_test
where metric = 'cointegration_p_value_holm' and value < 0.05 order by value;

-- every statistic for one pair and asset, one row per metric
select metric, value, n_rows, lags, reason from fct_econometric_test
where pair_id = 'mvrv_anchor' and symbol = 'COMP' order by metric;

-- the speed of adjustment, survivors only
select pair_id, symbol, metric, value, lags from fct_econometric_test
where metric like 'vecm_%' and value is not null order by 1, 2, 3;
```

The pre-test verdicts in §3 classify each series from its `adf_p_value` and `kpss_p_value` rows
(`independent_*` for the fundamental, `ratio_*` for the constrained ratio) at 0.05, by the table in
§3. The β intervals in §4 are `cointegration_coefficient ± 1.96 × cointegration_coefficient_stderr`.
The Johansen and Engle-Granger size figures in §4 are simulations, not warehouse queries. They are
recorded in the docstring of `cryptoval/econometrics/cointegration.py::johansen` and were measured
on 300 to 400 independent-walk pairs of 2,500 days each.
