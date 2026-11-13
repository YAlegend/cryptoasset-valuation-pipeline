# Does price anchor to fundamentals? — Phase 4b

**These are valuation estimates under stated assumptions, not investment advice. These assets are
volatile and these methods disagree by design.**

Run of 2026-10-18 against `data/cryptoval.duckdb`, panel vintage `2026-10-14 18:20:40.522565+01:00`.
Panel: `fct_asset_metrics_daily`, 69,264 symbol-days, 24 symbols, 2015-01-01 to 2026-10-14.
Results: `fct_econometric_test`, 1,248 rows, written by `python -m cryptoval.cli econ`. Every
number below is a query against that table and is reproducible with the commands in the last
section.

**The result, in one sentence: after a Holm correction over 56 tests and a check of each survivor
against its own simulated null, market cap holds a long-run equilibrium with realized cap on 8 of
20 testable assets (7 without USDC, whose peg makes it an identity) and with traded volume on 2 of
18, and with active addresses on none. Where an equilibrium exists, it is the fundamental that does
the correcting, not price.** Holm passed 11; ICP is withdrawn because its market cap is stationary
on its own and Engle-Granger rejects for it with no fundamental at all (§4). Only 6 of the 10 rest
on series that pass both unit-root pre-tests. On those 6 the gap between market cap and its
fundamental halves in 6 to 137 days. On all 10 the fundamental's adjustment is significant, while
market cap's own is significant on one, and on a second it runs the wrong way. Metcalfe's law could
not be tested as a law, because no asset has the equilibrium its exponent would describe.

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

| Pair | Tested | Raw p < 0.05 | Holm p < 0.05 | And past its own null |
|---|---|---|---|---|
| `mvrv_anchor` | 20 | 14 | 9 | **8** |
| `liquidity_anchor` | 18 | 7 | 2 | **2** |
| `metcalfe_anchor` | 18 | 2 | 0 | **0** |
| total | 56 | 23 | 11 | **10** |

The last column is the simulation in "Each survivor against its own null" below; it is not a
warehouse query. The 11 Holm survivors, each with its pre-test verdicts. β is the estimated
coefficient with its standard error, and Johansen is the trace test's verdict at rank 0:

| Pair | Asset | n | Holm p | β (se) | `log mcap` | fundamental | Johansen |
|---|---|---|---|---|---|---|---|
| MVRV | AAVE | 2,195 | 4.4e-26 | 1.245 (0.083) | I(1) | conflict | rank 1 |
| MVRV | LDO | 1,859 | 4.1e-08 | 1.532 (0.187) | I(1) | I(1) | **rank 0** |
| ~~MVRV~~ | ~~ICP~~ | 1,982 | 6.2e-06 | 0.891 (0.138) | conflict | I(1) | rank 1 — **withdrawn**, see below |
| MVRV | COMP | 2,309 | 7.1e-05 | 1.334 (0.022) | I(1) | I(1) | rank 1 |
| MVRV | USDC | 2,938 | 2.6e-04 | 1.026 (0.001) | conflict | conflict | rank 1 |
| MVRV | AVAX | 2,212 | 7.0e-04 | 0.929 (0.008) | conflict | conflict | rank 1 |
| MVRV | MKR | 3,012 | 1.1e-03 | 0.820 (0.033) | I(1) | I(1) | rank 1 |
| MVRV | ADA | 3,239 | 3.2e-03 | 1.272 (0.024) | I(1) | I(1) | rank 1 |
| liquidity | LINK | 3,302 | 9.6e-03 | 0.735 (0.012) | I(1) | I(1) | rank 1 |
| MVRV | CRV | 2,251 | 1.2e-02 | 1.110 (0.017) | conflict | conflict | rank 1 |
| liquidity | LTC | 4,304 | 2.8e-02 | 0.532 (0.006) | I(1) | I(1) | rank 1 |

**Only 6 of the 10 rest on two cleanly I(1) series**: LDO, COMP, MKR and ADA (MVRV), and LINK and
LTC (liquidity). The other four are reported with that caveat attached. A conflict verdict on
either series means the cointegration p-value is computed on a premise the data does not clearly
support.

### Each survivor against its own null

A conflict verdict is a warning; the simulation below measures what it costs.
`python -m cryptoval.econometrics.null_size` rebuilds every tested pair with no equilibrium in
it: the fundamental's daily differences are circularly shifted against market cap's and
re-integrated, so each series keeps its own autocorrelation, fat tails and volatility clustering
while the link between them is gone. The runner's own Engle-Granger, AIC lag rule and Johansen then
run on 100 such draws per pair, 5,600 in all. On the unshifted pairs the script reproduces the
mart's p-values exactly, and refuses to report otherwise.

| Pair | Asset | Live p | Null p < 0.05 | Null p < Holm bar (8.9e-04) | Smallest null p |
|---|---|---|---|---|---|
| MVRV | **ICP** | 1.2e-07 | 100 of 100 | **51 of 100** | 3.1e-04 |
| MVRV | AVAX | 1.4e-05 | 100 of 100 | 0 | 0.012 |
| MVRV | LDO | 7.4e-10 | 44 of 100 | 0 | 0.020 |
| MVRV | AAVE | 7.9e-28 | 0 | 0 | 0.070 |
| MVRV | CRV | 2.5e-04 | 0 | 0 | 0.070 |
| MVRV | ADA | 6.4e-05 | 0 | 0 | 0.28 |
| MVRV | MKR | 2.2e-05 | 0 | 0 | 0.37 |
| MVRV | COMP | 1.3e-06 | 0 | 0 | 0.61 |
| MVRV | USDC | 4.9e-06 | 0 | 0 | 0.66 |
| liquidity | LINK | 2.0e-04 | 0 | 0 | 0.17 |
| liquidity | LTC | 6.0e-04 | 0 | 0 | 0.22 |

The live p is Engle-Granger's raw p; the Holm bar is the strictest step, 0.05 / 56.

**ICP is withdrawn.** Its market cap is stationary by ADF alone (p = 0.0002), so a regression of it
on *any* series leaves a stationary residual: under the null, Engle-Granger clears the Holm bar for
ICP on 51 of 100 draws. ICP's survival says its market cap mean-reverts, not that it is anchored
to realized cap. It is also the only false survivor the null produces: over 100 null families of
56 tests, Holm passes at least one in 51, and every one is ICP.

**AVAX and LDO are kept, with a caveat.** Their null rejects at 0.05 on 100 and 44 of 100 draws,
so a raw p near 0.05 would mean nothing for either. But no null draw comes near the Holm bar (the
smallest null p is 0.012 for AVAX and 0.020 for LDO), while their live p-values are 1.4e-05 and
7.4e-10. With 100 draws that bounds the null's chance of clearing the bar at roughly 3%, not at
zero. The other eight survivors' nulls never reject even at 0.05.

The pre-tests in §3 flagged this and did not act on it: they are reported beside the test and do
not gate it. Making a stationary market cap a refusal would shrink the Holm family and move every
adjusted p-value in this chapter, so it is declared in §7 rather than done here.

**USDC's survival is the circularity this design exists to catch, not a finding.** A $1 peg makes
`market_cap ≈ supply` and `realized_cap ≈ supply` by construction, so their log ratio is pinned
near zero and β is 1.026 with a standard error of 0.001. It is reported because it was tested, and
it should not be counted.

**BTC and ETH do not survive** on any pair. BTC's MVRV raw p is 0.121; ETH's is 0.025 raw and 0.909
after Holm.

### Johansen, beside Engle-Granger

Johansen's trace test rejects rank 0 on **51 of 56** pairs, against Engle-Granger's 11. The two
disagree on 42 pairs, and the agreeable reading, that Johansen confirms everything, is wrong.

Part of the gap has a known cause. Johansen's rank counts *any* stationary linear combination of
the two series, including one that puts all its weight on a single stationary series. Where the
fundamental is I(0) or a conflict, which §3 shows is most of the volume and half the address
series, Johansen can find "rank 1" from that series alone.

**Independent walks do not explain the rest.** On the 24 pairs where *both* series are cleanly
I(1), Johansen still finds rank 1 on 19, and Engle-Granger fails after Holm on 14 of those 19. The
size distortion measured on independent walks does not account for that. With the constant unrestricted (`det_order=0`), Johansen's
critical values assume the walks drift. Measured on independent random walks (n = 2,500, 300 pairs
each) under this runner's AIC lag rule, it rejects 13.7% at drift/sd 0, 9.0% at 0.02 and 6.3% at
0.05, and this panel's median drift/sd for log market cap is 0.018. That is up to roughly 2.7× its
nominal 5%, but nowhere near 14 of 24.

**At the lags that actually ran, it is far worse.** AIC picks 0 lags on iid walks, while the live
pairs chose up to 14, at the cap on 25 of 56. The null simulation above runs at those lags, on each
pair's own dynamics, and there Johansen rejects a pair with no equilibrium in it 22.5% of the time:
4.7% where AIC picks 0 to 2 lags, 18.5% at 3 to 13, and **32.5% at the cap**. By pair it is 7% for
MVRV, but 32% for liquidity and 30% for Metcalfe, the pairs whose fundamentals are mostly conflicts
or I(0) in §3. That is several times its nominal 5% and explains much of the 51 of 56, though even
32.5% falls well short of 19 of 24 (79%) on the cleanly I(1) pairs; whether Engle-Granger also
lacks power there is open. Engle-Granger on the same null rejects 9.2% at 0.05, and **about 5.8%
once MVRV's ICP and AVAX, whose market caps are stationary on their own, are set aside** (they
reject on all 100 draws each), in line with the 5% to 7% measured on independent walks. **Engle-Granger carries the claim**, and a Johansen "rank 1" here is not
confirmation.

The one survivor Johansen rejects is LDO on MVRV, rank 0 with a trace statistic of 13.95 against a
95% critical value of 15.49.

### The coefficient against the ratio's assumption

Counted over the **survivors only**: by §5's own reasoning, β from a pair that does not cointegrate
is a spurious-regression coefficient and estimates nothing.

The MVRV ratio assumes β = 1. Over the 8 MVRV survivors the median β is **1.177** (1.245 without
USDC), and the 95% interval contains 1 on **none** of them. The only survivor whose interval did,
ICP at [0.62, 1.16], is the one withdrawn above. The liquidity ratio also
assumes 1. Its 2 survivors have β = 0.735 (LINK) and 0.532 (LTC), and neither interval contains 1.
So where an equilibrium exists, the ratio practitioners use is generally the wrong constraint. LTC
is the clean example: cointegrated at β = 0.532, while its forced ratio is I(1).

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

**Both series can correct the gap, so the VECM reports both loadings.** With the gap
z = log(market cap) + β_x·log(fundamental), α_mcap is market cap's share of yesterday's gap
corrected today and α_fund is the fundamental's. Each corrects by moving *against* the gap:
negative α_mcap, positive α_fund. The gap itself moves by **β′α = α_mcap + β_x·α_fund** a day, and
its half life, log(0.5) / log|1 + β′α| days, exists only for −2 < β′α < 0. β here is the VECM's
own maximum-likelihood estimate, close to but not the DOLS β in §4. The VAR lag order is the AIC's
choice.

An earlier draft of this chapter computed the half life from α_mcap alone, which assumes the
fundamental never moves. That made every equilibrium look months slower than it is.

| Pair | Asset | α_mcap (se) | t | α_fund (se) | t | β′α | Gap half life (days) |
|---|---|---|---|---|---|---|---|
| liquidity | LINK | −0.0022 (0.0026) | −0.9 | +0.126 (0.018) | +7.1 | −0.104 | **6** |
| liquidity | LTC | −0.0098 (0.0021) | **−4.6** | +0.103 (0.019) | +5.6 | −0.066 | **10** |
| MVRV | COMP | −0.0065 (0.0043) | −1.5 | +0.0136 (0.0016) | +8.6 | −0.024 | **28** |
| MVRV | CRV | −0.0073 (0.0078) | −0.9 | +0.0140 (0.0044) | +3.2 | −0.022 | 31 |
| MVRV | USDC | −0.0015 (0.0069) | −0.2 | +0.0186 (0.0069) | +2.7 | −0.021 | 33 |
| MVRV | MKR | −0.0057 (0.0030) | −1.9 | +0.0099 (0.0014) | +6.9 | −0.015 | **47** |
| MVRV | AVAX | +0.0021 (0.0038) | +0.5 | +0.0100 (0.0031) | +3.3 | −0.008 | 84 |
| MVRV | AAVE | **+0.0127** (0.0041) | **+3.1** | +0.0131 (0.0021) | +6.4 | −0.008 | 88 |
| ~~MVRV~~ | ~~ICP~~ (withdrawn, §4) | −0.0052 (0.0015) | −3.4 | +0.0017 (0.0005) | +3.6 | −0.008 | 93 |
| MVRV | LDO | −0.0037 (0.0027) | −1.4 | +0.0021 (0.0008) | +2.6 | −0.007 | **102** |
| MVRV | ADA | −0.0023 (0.0027) | −0.9 | +0.0021 (0.0005) | +4.4 | −0.005 | **137** |

Bold half lives are the 6 survivors resting on two cleanly I(1) series (§4).

The VECM still ran on ICP, because the runner fits every Holm survivor; its row stays in the table
and is not counted.

**The fundamental does the correcting.** α_fund has the correcting sign and is more than 2.6
standard errors from zero on **all 10** survivors. α_mcap is significant with the correcting sign
on only **one**, LTC; the other was ICP, whose market cap reverts on its own. On AAVE it is significant with the **wrong** sign: AAVE's market cap
moves *away* from its realized-cap equilibrium, and realized cap closes the gap faster than price
widens it. On the other 8, market cap's own adjustment is indistinguishable from zero.

Read plainly: where these equilibria exist, the fundamental moves toward price, not price toward
the fundamental. For the MVRV pair that has a mechanical reading. Realized cap revalues every coin
that moves at the day's price, so a rally that turns coins over pulls realized cap up behind
market cap. That is an equilibrium, but it is not the "price is anchored to its cost basis" the
practitioner literature means, and it gives price no pull toward any fair value.

The gap's speed ranges from a half life of 6 days (LINK on liquidity) to 137 (ADA on MVRV).
Liquidity equilibria close in days; realized-cap equilibria take one to four and a half months.

## 7. Declared gaps

One row per gap, each with its measurement.

| Spec | What is not here | Why, and what it costs |
|---|---|---|
| §11.4 | Structural breaks | One full-sample Engle-Granger over up to 4,304 days (2015 to 2026) can miss a regime shift entirely, or report one equilibrium where there were two. No rolling or recursive re-test is in scope. Gregory-Hansen, cointegration with one endogenous break, is the named alternative and is not built. |
| §11.4 | NVT, as the literature means it | `volume_reported_spot_usd` is **exchange trading volume**, not on-chain transfer value; the `nvt_golden_ratio` method card already declares this objection. The liquidity pair tests market cap against market liquidity. USD volume is also units traded × price, so this pair, like realized cap, is not wholly free of price. |
| §11.4 | A joint system | Bivariate pairs only. A joint Johansen over market cap and all three fundamentals at once is a reasonable test and is not built; interpreting a three-vector equilibrium is a different chapter's work. |
| §11.4 | A searched lag order | The VAR lag order is AIC's choice up to a cap of 14 days, and **AIC chose the cap on 25 of 56 pairs**, so on those it wanted more lags than it was allowed. ADF's lag is also AIC's choice. Nothing is tuned beyond that. At the cap Johansen rejects 32.5% of null pairs (§4), so a Johansen verdict at 14 lags carries almost no information here. |
| §11.4 | Pre-tests that gate | ADF/KPSS are reported beside Engle-Granger and never refuse it, so a pair whose market cap is stationary on its own enters the Holm family and can survive it. ICP did, and is withdrawn by the null simulation (§4) rather than by the runner. A refusal rule would shrink the family and move every adjusted p-value; it is not built. |
| §11.4 | Four testable assets, discarded over short gaps | AVAX and MANA (Metcalfe), MKR and USDC (liquidity) are refused over gaps of 1 to 8 days (§2). A test on the longest contiguous segment would recover them. Not built: the rule is "refuse, never stitch", and choosing a segment is a decision the design did not make. |
| §11.4 | GARCH, Markov-switching, DCC | §11.4's second chain, now `reports/does-volatility-cluster.md` (`fct_volatility_test`). |
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
python -m cryptoval.econometrics.null_size   # the null simulation in §4, 5,600 fits, ~11 min
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
The Johansen and Engle-Granger size figures in §4 are simulations, not warehouse queries. The
independent-walk figures were measured on 300 to 400 pairs of 2,500 days each; the figures on this
panel's own pairs, and the null check of every survivor, are printed by `null_size`, which is
seeded and reproduces them exactly. Both are recorded in the docstring of
`cryptoval/econometrics/cointegration.py::johansen`.
