# Do valuation signals earn returns? — Phase 4c, spec §11.5

**These are valuation estimates under stated assumptions, not investment advice. These assets are
volatile and these methods disagree by design.**

Run of 2026-10-21 against the Coin Metrics community universe: 135 listings serving a daily market
cap, fetched whole (389,742 rows, through 2026-10-20) into silver `cm_universe_daily`, vintage
`2026-10-21 06:36 +01:00`. After 43 exclusions (pegged assets, wrappers, the same token on a second
chain), **92 assets** are eligible in at least one week. The sample is **460 weeks, 2017-12-24 to
2026-10-11**, with 50 to 87 assets a week (mean 73). Results: `fct_factor_test` and
`fct_factor_returns`, written by `make factors` in under a minute. Every number below is a query
against those two tables.

**The result, in one sentence: neither valuation signal earns returns. MVRV and active addresses
over market cap show no decile spread, no alpha against the three Liu-Tsyvinski-Wu factors and no
Fama-MacBeth premium, with every adjusted p-value at 1.00. Of 13 hypotheses, one survives Holm:
the value-weighted momentum factor, +1.27% a week.**

## 1. Read this first: what the universe is

This is not the point-in-time top-N universe §11.5 asked for. No free source serves one: CoinGecko,
CoinPaprika and CryptoCompare/CoinDesk all refuse history beyond 365 days without a key or a paid
plan (the 2026-10-18 spike). Three biases follow, declared rather than corrected:

1. **Selection.** Membership is point-in-time *inside* Coin Metrics' catalog: no asset enters
   before its data does, and 27 of the 92 had left the universe before the final week. But Coin
   Metrics chose that catalog ex post. Assets missing from a true top 20 include BNB after
   2019-04-22, SOL (never present), DOT after 2022-06-03, TON, IOTA and Nano. The direction of the
   bias is unknown.
2. **XRP's market cap counts a different supply.** It ranks above BTC on 2018-01-07. Value
   weighting therefore gives XRP more weight than a circulating-supply source would. In the final
   week the weights are BTC 73.5%, ETH 13.5% and XRP 6.2%, so `CMKT` is close to BTC's own return.
3. **Delisting.** An asset with no price in week *t+1* leaves week *t*'s cross-section. It is not
   assigned −100%, because the end of Coin Metrics coverage is not proof of death.

There are also no sector labels: Coin Metrics serves none, and §8.3's labels exist only for the
curated 23. Sector-neutral sorts are not built.

## 2. The method

| Choice | Value | Why |
|---|---|---|
| Week | observation dated each Sunday, no forward fill | Liu-Tsyvinski-Wu (LTW) are weekly |
| Eligible | market cap ≥ $1M at *t*; prices at *t* and *t+1* | LTW's floor |
| Return | `p(t+1)/p(t) − 1`, no risk-free rate | long-short legs cancel it; a T-bill week is ~0.1% against a ~20% weekly sd |
| Signals | `size` = log cap; `momentum` = 3-week return; `value` = active addresses / cap; `mvrv` = Coin Metrics `CapMVRVCur` | §11.5 and LTW |
| Deciles | 10, value-weighted, at least 50 assets carrying the signal; `HL = D10 − D1` | equal weights would hand the result to micro-caps |
| LTW factors | single 30/40/30 sorts, value-weighted legs: `CMKT`, `CSMB` = small − big, `CMOM` = winners − losers | a 2×3 double sort leaves 8-18 assets a leg |
| Fama-MacBeth | weekly OLS on percentile ranks in [0, 1], returns winsorized at 5%/95%, at least 50 assets with all four signals | a slope is the top-minus-bottom spread; winsorizing keeps one micro-cap from setting it |
| Vendor spikes | a Sunday price more than 5x both neighbours (or below 1/5 of both) is nulled | 2 nulled, see below |
| Inference | Newey-West, 4 lags, two-sided | weekly returns are autocorrelated in volatility |
| Family | **13 hypotheses**, one Holm-Bonferroni correction | four HL means, three factor means, two alphas, four FM slopes |

**Two vendor spikes are nulled** (`model_id = 'panel'`, `metric = 'spikes_nulled'`). Both are on
2026-01-18, on BTG and FUN. Coin Metrics prices BTG at about 10x ($0.62 to $6.29) for nine days,
then back to $0.58. BTG is a micro-cap, but inside a decile of seven micro-caps its value weight was
15%. Before the rule, that one spike tripled the MVRV spread, from −0.16% to −0.48% a week.
Nulling the spike Sunday removes the two weekly returns that touch it. A jump that does not revert
is a real move and is kept. The rule nulls the same two prices at a threshold of 3x, and only BTG
at 10x. After it, the largest weekly return is +448% (SWRV, 2022-06-19). The smallest is −93%,
FTT in the week FTX failed (2022-11-06), and it is real.

## 3. Decile portfolios

Mean weekly return, value-weighted, in percent:

| Decile | `size` | `momentum` | `value` | `mvrv` |
|---|---|---|---|---|
| D1 (low) | 0.84 | −0.44 | 0.86 | 0.52 |
| D2 | 1.15 | −0.04 | −0.05 | 1.00 |
| D3 | 0.42 | −0.05 | 0.28 | 1.10 |
| D4 | 0.78 | 1.04 | 0.78 | 0.40 |
| D5 | 0.73 | 0.59 | 0.82 | 0.56 |
| D6 | 0.23 | 0.53 | 0.85 | 0.31 |
| D7 | 0.75 | 1.02 | 0.35 | 0.73 |
| D8 | 0.73 | 0.70 | 0.74 | 0.48 |
| D9 | 0.92 | 1.21 | 0.86 | 0.34 |
| D10 (high) | 0.66 | 0.45 | 0.90 | 0.36 |
| **HL** | **−0.18** | **+0.89** | **+0.05** | **−0.16** |
| HL t (NW) | −0.38 | 1.48 | 0.09 | −0.34 |
| HL p | 0.70 | 0.14 | 0.93 | 0.74 |
| HL p, Holm | 1.00 | 1.00 | 1.00 | 1.00 |
| Weeks | 460 | 457 | 459 | 459 |
| Mean assets a week | 73.2 | 73.0 | 71.3 | 70.1 |

None of the four spreads is distinguishable from zero, even before correction.

- **MVRV** points weakly the way the dissertation implies: low-MVRV ("undervalued") coins earned
  0.16% a week more than high-MVRV ones. But the decile means are not monotone, and the t-statistic
  is −0.34. With the BTG spike left in, this spread was three times larger (§2).
- **Active addresses over market cap**, §11.5's value factor, has the flattest profile of the four.
  Its HL is 0.05% a week.

## 4. The three LTW factors, and alphas against them

| Factor | Mean weekly return | t (NW) | p | p, Holm | Weeks |
|---|---|---|---|---|---|
| `CMKT` | +0.67% | 1.44 | 0.150 | 1.00 | 460 |
| `CSMB` | −0.03% | −0.08 | 0.94 | 1.00 | 460 |
| **`CMOM`** | **+1.27%** | **2.97** | **0.0030** | **0.039** | 457 |

**Momentum is the one result in this chapter that survives the correction.** The size premium LTW
report does not appear. That is unsurprising here: the universe is Coin Metrics' coverage, not the
long tail of small coins in which LTW's size effect lives.

The two valuation spreads, each regressed on the three factors:

| HL portfolio | Alpha a week | t (NW) | p | p, Holm | β CMKT | β CSMB | β CMOM |
|---|---|---|---|---|---|---|---|
| `value` | +0.01% | 0.02 | 0.99 | 1.00 | 0.17 | 0.16 | 0.01 |
| `mvrv` | −0.18% | −0.43 | 0.67 | 1.00 | −0.05 | −0.60 | 0.04 |

Neither signal earns anything the factors do not already price, and neither earned much before
adjustment. MVRV's spread loads at −0.60 on `CSMB`. It behaves partly like a big-minus-small portfolio:
high-MVRV coins tend to be the large ones, so part of what an MVRV sort captures is size.

## 5. Fama-MacBeth

Weekly cross-sectional slopes on the four ranks jointly, 456 weeks:

| Signal | Mean slope a week | t (NW) | p | p, Holm |
|---|---|---|---|---|
| `size` | +0.32% | 0.98 | 0.33 | 1.00 |
| `momentum` | **−0.48%** | −1.99 | 0.046 | 0.55 |
| `value` | +0.11% | 0.48 | 0.63 | 1.00 |
| `mvrv` | +0.17% | 0.68 | 0.49 | 1.00 |

The momentum slope has the **opposite sign** to `CMOM`, and it is the only other raw p-value below
0.05. The two tests weight differently:
- **Fama-MacBeth** weights every coin equally. Across the whole cross-section, last month's winners
  did slightly worse the next week.
- **`CMOM`** is value-weighted. Among the large coins that dominate it, winners kept winning.

That is a description of this sample, not a finding the correction lets stand. The Fama-MacBeth
momentum slope's Holm p is 0.55.

## 6. What this says about the dissertation

The dissertation's correlation chapter could not ask whether a valuation signal earns returns. This
chapter does, and the answer for both valuation signals testable on free data is **no**:
- **MVRV**, the dissertation's §4.2 signal, earns no spread and no alpha.
- **Active addresses per dollar of market cap**, the network-value-to-users reading behind its P/U
  and Metcalfe chapters, earns none either.

This matches the earlier chapters. MVRV is circular as a fair value (`dissertation-comparison.md`
§3), and it fails to forecast any single asset (`does-anything-beat-a-random-walk.md`). Here it also
fails to rank assets against one another.

**Full sample, descriptive.** Week-*t* sorts against week-*t+1* returns have no look-ahead. But the
signals, lookbacks and floors were chosen in 2026 with the literature in hand, and the universe was
chosen by a vendor after the fact. The momentum result is "this factor earned returns in this
sample", not "tradable". Costs, slippage and capacity are not modelled.

## 7. Gaps

| Gap | Effect |
|---|---|
| No point-in-time top-N universe | §1's selection bias; the user chose this universe over a key-based one on 2026-10-21 |
| No sector labels | No sector-neutral sorts |
| NVT, realized-cap signals | `TxTfrValAdjUSD` and `CapRealUSD` are not free; MVRV is the only realized-value signal |
| Multi-chain tokens | One listing kept (`avaxp`, `leo_eth`, `bnb`, `flow_native`); their size is understated |
| LTW's 2×3 momentum sort | Single sorts instead; the legs would hold 8-18 assets |

---

### Reproducing this chapter

```bash
make factors                 # refetch the universe (~1 min), then the 13 tests
cd dbt && ../.venv/bin/dbt build --select fct_factor_test fct_factor_returns --profiles-dir . && cd ..
```

The floors and the exclusion list are `config/factors.yml`. Every number above is a select against
`fct_factor_test` filtered by `model_id`, `signal`, `portfolio` and `metric`. The weekly series are
in `fct_factor_returns`.
