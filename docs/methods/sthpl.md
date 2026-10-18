# sthpl

**Family:** E_cost
**Citation:** Dissertation 6.2, short-term-holder profit/loss (computed as STH-MVRV, Plan 3b Ruling 3)
**Returns a fair value:** no, it is a correction

## Data dependencies

- `price_usd`
- `sth_realized_cap`
- `sth_supply`

Minimum coverage density: **50%** per column. Minimum observations:
**30**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- (none)

## Failure modes

- The dissertation's literal STH P/L over LTH P/L is undefined when LTH P/L crosses zero; this is price over the realized price of coins younger than 155 days, which matches its own reading that unity is breakeven.

## Assumptions

- A coin's age is days since the output was created; 155 days separates short- from long-term holders. The cutoff is fixed in cryptoval/utxo/derive.py (STH_DAYS), where the cohort is built, not a parameter of this ratio.
