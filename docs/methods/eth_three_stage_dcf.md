# eth_three_stage_dcf

**Family:** C_cashflow
**Citation:** Spec 10.4, ETH three-stage DCF; discount rate decomposed per spec 10.4
**Returns a fair value:** yes

## Data dependencies

- `net_revenue_usd`
- `issuance_usd`
- `supply`
- `price_usd`

Minimum coverage density: **50%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `discount_rate` = `0.15`
- `growth` = `[0.2, 0.1, 0.03]`
- `max_terminal_share` = `0.8`
- `sensitivity` = `{'discount_rate': [0.1, 0.15, 0.2, 0.25], 'terminal_multiple': [10.0, 15.0, 25.0]}`
- `stage_years` = `[3, 3, 4]`
- `terminal_multiple` = `15.0`
- `window_rows` = `365`

## Failure modes

- The output is a GRID, not a number. A single cell quoted without its (discount_rate, terminal_multiple) pair is not this method's answer, and the spread across cells is wider than most method-to-method disagreement in this repo.
- Terminal value dominates: every row carries terminal_share, and a row above max_terminal_share is flagged terminal_dominated. Such a row is a statement about the terminal multiple, not about Ethereum.
- The cash flow is BURN MINUS ISSUANCE, which is negative for most of ETH's post-merge history: annualised issuance is $2.49bn against a $0.05bn burn as of 2026-10-14. Only 331 of the 1,897 days since EIP-1559 carry a positive cash flow, so this method refuses on 83% of them. That refusal IS the finding, and it is the same one supply_effect_counter_model makes arithmetically.
- The narrower of two defensible cash flows. Burn accrues to EVERY holder by reducing supply; priority fees and MEV accrue to PROPOSERS, who are a subset of holders. Counting those too (total_fees_usd - issuance_usd) turns 457 days positive instead of 331. This method takes the narrow reading because the burn is what Step 1 could verify against the vendor's methodology.
- ETH's net_revenue_usd is zero for 53.6% of its history - every day before EIP-1559 on 2021-08-05, when no fee was burned. Those are measured zeros, not gaps, but a trailing window reaching across that date averages a regime that no longer exists.
- A trailing-year mean annualised assumes the last year represents the next ten; ETH's fee revenue is cyclical with onchain activity and L2 migration has moved it structurally.
- MEASURED AGAINST THE COUNTER-MODEL, 2026-10-13: this family values ETH at $2.58-$5.50 per token while supply_effect_counter_model says $2,300 against a $2,497 price -- a factor of 419. That is the disagreement spec 10.4 asks to be measured rather than argued, and it is far wider for ETH than for any DeFi token (LINK 5.9x, UNI 3.6x, COMP 2.2x, AAVE 0.8x). ETH is also exactly the asset the critique was aimed at.
- Despite the method_id, nothing here is ETH-specific: it runs for any asset with a burn-like net_revenue_usd and an issuance series, which on this universe is 9 symbols.

## Assumptions

- Burned fees accrue to holders like a buyback and issuance dilutes like compensation.
- Three stages of 3, 3 and 4 years, and a terminal multiple on the final year's cash flow.
