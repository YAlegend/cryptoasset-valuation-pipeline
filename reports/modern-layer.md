# The modern layer on free data — Phase 3a

**These are valuation estimates under stated assumptions, not investment advice. These assets are
volatile and these methods disagree by design.**

Run of 2026-10-16 against `data/cryptoval.duckdb`, panel vintage `2026-10-14 18:20:40+01:00`.
Panel: `fct_asset_metrics_daily`, 69,264 symbol-days, 24 symbols, 2015-01-01 to 2026-10-14.
Estimates: `fct_valuation_estimate`, 1,410,145 rows over **35 methods**, written by
`python -m cryptoval.cli estimate`. Unless stated otherwise every figure is taken on
**2026-10-13**, the last date with a complete Coin Metrics panel — 2026-10-14 carries CoinGecko
price only, and 28 of the 35 methods have nothing to read on it.

Every number below is a query against that table and is reproducible with the commands in the last
section.

---

## 1. What Phase 3a adds, and what it does not

Plan 2 left 14 methods in the registry. This phase adds **21**, for 35 across six families:

| Family | Methods | Added here |
|---|---|---|
| A — monetary | 5 | `kampakis_log_linear_eoe` |
| B — network | 4 | `peterson_metcalfe`, `nvm_ratio`, `lppls_bubble_flag` |
| C — cash flow | 5 | all five: `eth_three_stage_dcf`, `defi_net_revenue_dcf`, `bond_target_yield`, `emissions_adjusted_real_yield`, `supply_effect_counter_model` |
| D — multiples | 9 | `price_to_fees`, `price_to_sales`, `price_to_tvl`, `fdv_to_revenue`, `dfdv_to_revenue`, `sector_cohort_zscore` |
| E — cost | 10 | `puell_multiple`, `mayer_multiple`, `mvrv_z_score`, `energy_value` |
| F — falsified shelf | 2 | both: `stock_to_flow`, `power_law` |

Three of the 21 are not fair values and say so in their own diagnostics:
`emissions_adjusted_real_yield` (`status: yield_not_a_price`), `sector_cohort_zscore`
(`relative_score_not_a_price`) and the Mayer/Puell multiples (`diagnostic_not_fair_value`).
`lppls_bubble_flag` is a regime flag. Counting only the methods that emit a price, the universe's
widest asset — ETH, 23 methods on 2026-10-13 — spans **$2.58 to $3,681.82**, a factor of 1,429,
against a market price of $2,497.35.

**What is absent.** Family E(b) — Cointime, HODL waves, SLRV, STHPL, RPV and reserve risk — is not
here. Spec §6.3 makes it all-or-nothing: those series are cumulative from genesis, so a partial UTXO
history yields *wrong* values rather than partial ones, and the free tier serves no UTXO age
distribution. It needs the BigQuery extraction that Plan 3b owns. The §10.5
market-cap-to-net-assets multiple is absent for a different reason — no free source carries a
protocol balance sheet at all. Both appear in §6 below with the measurement that decided them.

Coverage, applicable rows per method, over the whole panel:

| Family | Method | Applicable rows | Symbols | Param sets |
|---|---|---|---|---|
| A | `crypto_qtm` | 7,286 | 20 | 1 |
| A | `crypto_qtm_panel_velocity` | 6,916 | 19 | 1 |
| A | `holding_time_correction` | 6,726 | 20 | 1 |
| A | `j_curve` | 7,306 | 20 | 1 |
| A | `kampakis_log_linear_eoe` | 3,496 | 19 | 1 |
| B | `lppls_bubble_flag` | 55,149 | 22 | 1 |
| B | `metcalfe_estimated` | 53,948 | 20 | 1 |
| B | `nvm_ratio` | 53,889 | 21 | 1 |
| B | `peterson_metcalfe` | 53,888 | 20 | 1 |
| C | `bond_target_yield` | 18,636 | 9 | 1 |
| C | `defi_net_revenue_dcf` | 42,783 | 9 | 4 |
| C | `emissions_adjusted_real_yield` | 18,434 | 9 | 1 |
| C | `eth_three_stage_dcf` | 108,924 | 9 | 13 |
| C | `supply_effect_counter_model` | 50,046 | 20 | 1 |
| D | `dfdv_to_revenue` | 11,764 | 7 | 1 |
| D | `fdv_to_revenue` | 10,982 | 7 | 1 |
| D | `nvt_golden_ratio` | **0** | 0 | 1 |
| D | `nvt_reversion` | 3,496 | 19 | 1 |
| D | `price_to_fees` | 18,636 | 9 | 1 |
| D | `price_to_sales` | 14,494 | 9 | 1 |
| D | `price_to_tvl` | 15,880 | 7 | 1 |
| D | `price_to_utility` | **0** | 0 | 1 |
| D | `sector_cohort_zscore` | 11,450 | 7 | 1 |
| E | `energy_value` | 14,817 | 4 | 1 |
| E | `hayes_cost_of_production` | 16,273 | 4 | 1 |
| E | `investor_cap_price` | 57,334 | 20 | 1 |
| E | `mayer_multiple` | 58,779 | 22 | 1 |
| E | `mvrv_reversion` | 53,734 | 20 | 1 |
| E | `mvrv_z_score` | 50,034 | 20 | 1 |
| E | `mvtv_reversion` | 28,842 | 9 | 1 |
| E | `puell_multiple` | 29,369 | 13 | 1 |
| E | `realized_price` | 57,334 | 20 | 1 |
| E | `thermocap_floor` | 59,468 | 20 | 1 |
| F | `power_law` | 35,013 | 13 | 1 |
| F | `stock_to_flow` | 27,814 | 14 | 1 |

The thirteenth param set of `eth_three_stage_dcf` is not a grid cell: it is the hash of the
unexpanded config, used by the registry when a symbol is refused before the grid is built (15
symbols, all on density or row-count grounds).

## 2. The DCF chapter

### The ETH three-stage grid

The sensitivity block is `discount_rate ∈ {0.10, 0.15, 0.20, 0.25}` × `terminal_multiple ∈
{10, 15, 25}`, twelve cells, each its own `param_set_hash` and its own row. Fair value per token,
2026-10-13:

| Cell | Terminal share | AAVE | COMP | LINK | UNI | LDO |
|---|---|---|---|---|---|---|
| r=0.10, ×25 | 0.677 | 206.06 | 10.24 | 2.1653 | 2.0842 | 0.3123 |
| r=0.10, ×15 | 0.557 | 150.24 | 7.46 | 1.5788 | 1.5196 | 0.2277 |
| r=0.15, ×25 | 0.629 | 142.22 | 7.07 | 1.4945 | 1.4385 | 0.2156 |
| r=0.10, ×10 | 0.456 | 122.33 | 6.08 | 1.2855 | 1.2373 | 0.1854 |
| r=0.15, ×15 | 0.504 | 106.43 | 5.29 | 1.1184 | 1.0765 | 0.1613 |
| r=0.20, ×25 | 0.577 | 101.30 | 5.03 | 1.0645 | 1.0247 | 0.1536 |
| r=0.15, ×10 | 0.404 | 88.54 | 4.40 | 0.9304 | 0.8955 | 0.1342 |
| r=0.20, ×15 | 0.450 | 77.92 | 3.87 | 0.8188 | 0.7882 | 0.1181 |
| r=0.25, ×25 | 0.522 | 74.40 | 3.70 | 0.7819 | 0.7526 | 0.1128 |
| r=0.20, ×10 | 0.353 | 66.23 | 3.29 | 0.6960 | 0.6699 | 0.1004 |
| r=0.25, ×15 | 0.396 | 58.86 | 2.92 | 0.6185 | 0.5953 | 0.0892 |
| r=0.25, ×10 | 0.304 | 51.09 | 2.54 | 0.5368 | 0.5167 | 0.0774 |

Two parameters a hand sets move the answer by **4.0×** on every asset. That ratio is the method's
honest output; a single cell quoted without its pair is not.

**`terminal_dominated` never fires.** The guard is `max_terminal_share: 0.8` and the grid's highest
terminal share is 0.677, at the cell that most favours it (lowest discount rate, highest multiple).
Across the whole table: **0 of 108,924 applicable rows** are flagged. The unit test that proves the
flag works (`test_eth_dcf_flags_a_cell_whose_value_is_mostly_terminal`) constructs the condition;
live data on this universe never reaches it. The flag is correct and, on this vintage, silent.

**The method refuses ETH.** Despite the name, `eth_three_stage_dcf` is refused on ETH for
**45,060 of 49,032** rows — 91.9% — with the reason `no positive annualised burn_net_of_issuance
over 365 rows`. Post-merge, annualised issuance is $2.49bn against a $0.05bn burn: the cash flow to
the token is negative, so there is nothing to discount. ETH was last valuable to this method on the
331 days since EIP-1559 where the trailing year of burn exceeded issuance. That refusal is the
chapter's central result, and §3 shows the counter-model making it arithmetically.

Refusal reasons across the nine revenue-bearing symbols:

| Symbol | Applicable | Refused | Reason when refused |
|---|---|---|---|
| AAVE | 23,508 | 2,172 | no positive annualised burn_net_of_issuance |
| COMP | 25,536 | 2,172 | as above |
| LDO | 20,136 | 2,172 | as above |
| MKR | 20,832 | 9,432 | as above |
| LINK | 7,140 | 6,828 | as above |
| SNX | 4,068 | 17,496 | as above |
| **ETH** | **3,972** | **45,060** | as above |
| UNI | 3,468 | 20,832 | as above |
| CRV | 264 | 25,812 | as above |

The other 15 symbols are refused once each, before the grid: 14 for `net_revenue_usd is 0.0% dense`
(no DefiLlama revenue series exists for them) and SOL for `only 363 rows, below the method's minimum
of 365` — its history starts two days short of the window.

### The DeFi net-revenue DCF

Of the nine revenue-bearing symbols, this method's refusals split cleanly by cause:

| Reason | Symbols | Rows |
|---|---|---|
| `revenue_regime = fees_fully_distributed` — the protocol does not retain revenue, so there is no cash flow to discount | LINK, UNI, ETH | 10,182 |
| no positive annualised net protocol revenue over 365 rows | MKR, COMP, UNI, LINK, SNX, AAVE, LDO, CRV | 7,257 |
| `net_revenue_usd` 0.0% dense | the other 15 symbols | 15 |

The regime is read per row, not per asset: UNI and ETH appear in both the refused and the valued
sets at different dates, because what share of fees the protocol retains changes over time. On
2026-10-13 the method values eight symbols, ETH among them, at **$2.58–$5.50** per token across its
four param sets — against a $2,497.35 price. Terminal share at AAVE's base cell is 0.387, again
below the dominance guard.

### The bond-style target yield

`bond_target_yield` is a flat perpetuity: current cash-flow yield capitalised at a hand-set target.
Its diagnostics carry only `implied_current_yield` and `status: flat_perpetuity`, and on
2026-10-13 it values AAVE at $431.96 against a $160.25 price — 2.7× — on an implied current yield
of 26.96%. Two hand-set parameters (the target yield and the payout assumption) carry the whole
result; nothing in the data pins either. Read it as a statement about what yield one demands, not
about the asset.

`emissions_adjusted_real_yield` emits **no fair value at all** — `fair_value` is NULL on every
applicable row and `status` is `yield_not_a_price`. Its `dilution` term is 0.0 and
`incentive_share_available` is `false` on every row, because no free source serves per-asset
incentive emissions. Its published `real_yield` therefore equals its `fee_yield` exactly
(AAVE: 0.2696) and is a **lower bound on dilution, i.e. an upper bound on real yield**. See §6.

## 3. Where the DCF and its counter-model disagree

Spec §10.4 asks for the disagreement to be measured rather than argued.
`supply_effect_counter_model` holds network value constant and asks what a change in supply alone
implies — `holds_network_value_constant: true`, `status: null_hypothesis_against_the_dcfs`. On
2026-10-13, against the ETH three-stage grid's full range:

| Symbol | Price | DCF low | DCF high | Counter-model | Counter ÷ DCF high |
|---|---|---|---|---|---|
| LINK | 12.8772 | 0.5368 | 2.1653 | 12.8772 | **5.95×** |
| UNI | 7.5487 | 0.5167 | 2.0842 | 7.5487 | **3.62×** |
| COMP | 22.8978 | 2.5383 | 10.2383 | 22.8978 | **2.24×** |
| LDO | 0.4328 | 0.0774 | 0.3123 | 0.4328 | **1.39×** |
| AAVE | 160.2525 | 51.0854 | 206.0580 | 160.2525 | **0.78×** |
| ETH | 2497.3456 | *refused* | *refused* | 2300.8920 | — (see below) |

**They agree on nothing, and on one asset they agree by accident.** AAVE's 0.78× is the only ratio
inside the grid's own range, and it is inside it only because the grid is 4× wide: the counter-model
sits between the r=0.10,×15 and r=0.10,×25 cells. On LINK, UNI, COMP and LDO the counter-model is
above *every* cell of a twelve-cell grid.

For ETH — the asset the §10.4 critique was aimed at — the three-stage DCF declines to answer, so the
comparison runs against the DeFi DCF instead: **$2.58–$5.50 against the counter-model's $2,300.89**,
a factor of **419 to 893**. The counter-model is within 7.9% of the $2,497.35 price; the cash-flow
models are three orders of magnitude below it.

Note what the counter-model's number *is* on five of these six assets: exactly the market price, to
every printed digit. Its `supply_change_fraction` is 0.0 for them, so holding network value constant
returns the price unchanged. It diverges only where supply is moving — ETH (0.921× price), AVAX
(0.223×), SNX (0.005×), USDC (4.12×). So the "disagreement" on LINK, UNI, COMP and LDO is really the
DCF disagreeing with the market by 2–6×, with the counter-model standing in for the market.

**What this implies about cash-flow arithmetic on an L1 token.** The DCF's answer for ETH is not a
low valuation; it is a *refusal*, and the refusal is structural rather than parametric. Burn minus
issuance has been negative for 91.9% of the rows where the inputs exist, at every discount rate in
the grid. No choice of r or terminal multiple turns a negative cash flow positive. Discounted
cash flow is not a method one can calibrate onto ETH badly — on this definition of the cash flow it
has no object. The narrower reading is deliberate (burn accrues to every holder; priority fees and
MEV accrue to proposers, a subset); counting fees too would turn 457 days positive instead of 331,
which moves the share from 8.1% to 11.2% and changes nothing about the conclusion.

## 4. The falsified shelf, measured

Family F exists to be beaten, not believed. Both methods carry `status: contested` and
`residuals_autocorrelated: true` in every diagnostics blob. Bitcoin, 2026-10-13, price $83,141.42:

| Method | Fair value | ÷ price | β | R² | Durbin-Watson | n |
|---|---|---|---|---|---|---|
| `stock_to_flow` | 155,764.80 | 1.873 | 2.258 | 0.8985 | **0.0044** | 3,939 |
| `power_law` | 147,097.17 | 1.769 | 5.865 | 0.9151 | **0.0041** | 4,303 |

**That is what a spurious fit looks like.** R² near 0.9 with Durbin-Watson at 0.004 means the
residuals are almost perfectly autocorrelated: the regression is tracking a trend in both series,
not a relationship between them. A Durbin-Watson of 2.0 would indicate independence; 0.004 is three
orders of magnitude away from it. The two methods are also the top two fair values for BTC on this
date, and they agree with each other (1.87× and 1.77×) far better than either agrees with anything
else — which is the signature of two trend-followers fitted to the same trend, not corroboration.

It is not confined to BTC. Across every applicable symbol:

| Method | Symbols | Durbin-Watson range | R² range | β range |
|---|---|---|---|---|
| `power_law` | 13 | 0.0031 – 0.0117 (USDC 1.338) | 0.0136 – 0.9151 | −0.780 – 5.865 |
| `stock_to_flow` | 14 | 0.0003 – 0.0410 (POL 0.326) | 0.0002 – 0.8985 | −2.018 – 17.840 |

Every symbol-method pair but two sits below DW = 0.05: the exceptions are `power_law` on USDC
(1.338) and `stock_to_flow` on POL (0.326), and USDC is a dollar peg rather than a trend. β — the exponent the entire claim rests on — ranges from
−2.0 to +17.8 across assets for the same functional form, with `stock_to_flow` giving DOGE a β of
17.84 and ADA a β of −0.843. A law whose exponent changes sign between assets is a curve fit with a
name.

## 5. The circularity audit, extended

Plan 2's comparison report audited 14 methods and found 6 circular. At 35 methods the registry's
automatic flag catches **9**:

`crypto_qtm_panel_velocity` and `price_to_utility` (`velocity_total`); `realized_price`,
`mvrv_reversion` and `mvrv_z_score` (`realized_cap`); `mvtv_reversion` (`mvtv`);
`investor_cap_price` (`investor_cap_usd`); `holding_time_correction`; and
`kampakis_log_linear_eoe` (`velocity_total`).

Two of the new methods need more than the flag.

**`kampakis_log_linear_eoe` — flagged, but the flag understates it.** The registry marks it circular
because it reads `velocity_total`. The real defect is worse and the method declares it itself:
`structurally_circular: true`, and the design matrix is **rank 3 with 4 columns**, condition number
~10¹⁶. M is price × supply and V is T/M, so log(price) is being regressed on two transforms of
log(price); the fit is in part an identity. `coefficients_identified` is therefore `false`. What the
re-estimation returns, 2026-10-13:

| Symbol | β_M | β_T | β_(1/V) | adj. R² | Durbin-Watson |
|---|---|---|---|---|---|
| ADA | 0.678 | 0.337 | 0.341 | 0.99982 | 0.0920 |
| BCH | 0.670 | 0.335 | 0.335 | 0.99999 | 0.0485 |
| BTC | 0.672 | 0.336 | 0.335 | 0.99988 | 0.1457 |
| DOGE | 0.686 | 0.343 | 0.343 | 0.99966 | 0.0399 |
| ETH | 0.670 | 0.336 | 0.335 | 0.99994 | 0.1165 |
| LTC | 0.676 | 0.338 | 0.338 | 0.99983 | 0.0186 |
| *published* | *0.88* | *0.84* | *1.15* | — | — |

Against the published (0.88, 0.84, 1.15) not one coefficient reproduces. What reproduces instead is
the identity: β_M ≈ ⅔ and β_T ≈ β_(1/V) ≈ ⅓ on **every asset**, to three decimals, with adj. R² at
0.9999. Six unrelated assets do not share a regression coefficient to three decimals; they share an
arithmetic constraint. The confidence intervals are correspondingly meaningless — BCH's β_M band is
[0.6697, 0.6702] — because a rank-deficient design produces a least-norm solution, not an estimate.
This is the one place in the project where a published set of coefficients could be checked against
re-estimation, and the answer is that the specification does not admit estimation at all.

**`energy_value` — not flagged, and not independent.** The registry sees no market-cap-derived
column, so it reports `circular_inputs = ''`. But its own diagnostics say
`derived_from: hayes_cost_of_production`, `multiplier: 2.0`,
`status: not_independent_of_hayes`. It is twice a smoothed Hayes cost, so it carries no information
Hayes does not. Measured, 2026-10-13:

| Symbol | `energy_value` | `hayes_cost_of_production` | Ratio |
|---|---|---|---|
| BTC | 105,375.94 | 50,667.63 | 2.080 |
| BCH | 549.37 | 177.81 | 3.090 |
| LTC | 0.037937 | 0.016771 | 2.262 |
| DOGE | 0.0000100 | 0.0000043 | 2.325 |

The ratio is not exactly 2.0 only because the smoothing window differs; the drift from 2.08 to 3.09
is the smoothing, not new evidence. Counting both as independent support for a cost floor
double-counts one measurement.

**Two further methods disclose a fitted constant rather than a circularity.**
`peterson_metcalfe` reports `scale_constant_is_fitted: true` with `fit_rows: 365` — Peterson's
bitcoin-days-destroyed term is not on the free tier, so the functional form is his and the
calibration is ours (§6). `mvrv_z_score` reports `derived_from: realized_price` and
`sigma_basis: prior_only_rolling_sample_stdev_ddof1`: it is circular through realized cap *and* its
σ is estimated only on prior rows, so the score is not comparable across assets with different
history lengths.

**One method is silent because its gates never open.** `lppls_bubble_flag` returns
`bubble_regime: false` on **all 55,149 applicable rows across 22 symbols** — not one bubble flagged
in eleven years of data. A representative fit: 8 restarts attempted, 8 converged,
**0 passing the gates**, with `fit_status: tc_beyond_horizon` and `critical_time: null`. The method
is working as specified (a date appears only when both the `tc` bound and the RMSE floor pass), but
a detector that never fires on a decade containing 2017, 2021 and 2024 has not been shown to
discriminate. Treat it as untested, not as evidence of calm.

## 6. Declared gaps: the methods free data cannot support

A documented refusal with a measurement is this project's output format for an unavailable method.
One row per gap.

| Spec | Method | Why it is not implemented |
|---|---|---|
| §10.5 | Market-cap-to-net-assets | No free source serves protocol treasury or net asset value. DefiLlama's free surface carries fees, revenue and TVL; none of the three is a balance sheet, and a treasury cannot be inferred from a flow. |
| §10.4 | Real yield's incentive term | No free per-asset incentive-emissions series exists. `emissions_adjusted_real_yield` is implemented with the term parameterised to zero: `dilution: 0.0` and `incentive_share_available: false` on **all 18,434 applicable rows**, so its `real_yield` equals its `fee_yield` exactly (AAVE 0.2696) and is an **upper bound**, not a measurement. The method emits a NULL `fair_value` for this reason. |
| §10.6(b) | Cointime, HODL waves, SLRV, STHPL, RPV, reserve risk | Requires the §6.3 BigQuery UTXO extraction. These series are cumulative from genesis, so a partial history yields *wrong* values rather than partial ones — there is no honest subset to ship. Deferred whole to Plan 3b. |
| §10.3 | Peterson's `b_t` (bitcoin-days-destroyed) | Not on the free tier. `peterson_metcalfe` runs with the scale constant **fitted on a trailing 365-row window** and says so on every row (`scale_constant_is_fitted: true`). The functional form is Peterson's; the calibration is not, so it cannot be used to test his published constant. |
| §10.5 | `nvt_golden_ratio`, `price_to_utility` | Implemented and registered, but applicable on **0 rows of 24**. Both need inputs the free tier does not carry at daily frequency. They are in the registry so their refusal is recorded per run rather than forgotten. |

**Not a gap: P/TVL.** The plan held this open pending Task 6's payload measurement
(`GET /protocol/<slug>` returns 30.2 MB per protocol, ~600 MB for one backfill). Task 6 implemented
the size guard instead, and `price_to_tvl` is live on **15,880 rows across 7 symbols** (MKR, SNX,
COMP, CRV, UNI, AAVE, LDO). Note the two distinct zero-TVL cases the method must separate, because
they are not the same fact: LINK is refused for `no positive annualised tvl_usd over 30 rows` — its
TVL is measured and genuinely zero — while ETH, BTC and ten others are refused for
`tvl_usd is 0.0% dense`, meaning DefiLlama publishes no TVL series for them at all.

**Not a gap, but a thin one: `sector_cohort_zscore`.** It runs cross-sectionally and refuses
**57,814 of 69,264 rows (83.5%)** because the cohort is too small on the date, each refusal naming
its measured size — e.g. `cohort 'smart_contract_l1' has 1 member(s) with a usable ratio on this
date, below the minimum of 3`. With 24 assets split across six sectors, a 24-asset universe is
simply too small for cohort-relative scoring most of the time. The method is correct; the universe
is the constraint, and widening it is Plan 4's lever.

---

### The spread, as the one-line summary

Across every method that emits a positive price, on 2026-10-13:

| Symbol | Methods | Low | High | Spread |
|---|---|---|---|---|
| ETH | 23 | 2.58 | 3,681.82 | 1,429× |
| BTC | 20 | 4,882.44 | 155,764.80 | 31.9× |
| BCH | 20 | 85.18 | 549.37 | 6.4× |
| AAVE | 16 | 34.74 | 863.92 | 24.9× |
| UNI | 16 | 0.3514 | 19.0165 | 54.1× |
| LINK | 14 | 0.5368 | 19.4985 | 36.3× |

Plan 2's report measured an 18× spread on BTC over 14 methods. At 35 methods it is 31.9×, and ETH —
where the cash-flow family now has an opinion — is 1,429×. Adding methods did not converge the
answer; it widened it, and it widened it most on the asset with the most methods. The deliverable of
Phase 3a is not a fair value. It is 35 documented methods, a measured spread, and a list of
refusals each carrying the measurement that caused it.

### Reproducing this chapter

```bash
make cards                                   # regenerate docs/methods/ -- 35 cards
python -m cryptoval.cli estimate             # writes gold/valuation_estimate (~17 min)
make dbt                                     # fct_valuation_estimate, dim_method
python -m cryptoval.cli validate             # see note below
```

`validate` exits **1** on this vintage, on two checks, both freshness:
`asset_metrics/freshness: latest 2026-10-13 is 94.3h old` and
`market_chart/freshness: latest 2026-10-14 is 70.3h old`. Neither is a content failure — no
`ingest` has run since 2026-10-14. The 14 warnings are the same classes Plan 2 recorded: unmapped
entities (`trx_eth` density, negative DefiLlama fees on 17 unmapped protocols) and stale-entity
notices on four sources.
