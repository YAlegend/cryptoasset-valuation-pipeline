# realized_price

**Family:** E_cost
**Citation:** Coinmetrics (2018), realized capitalisation; dissertation section 3.5
**Returns a fair value:** yes

## Data dependencies

- `realized_cap`
- `supply`

Minimum coverage density: **50%** per column. Minimum observations:
**30**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- (none)

## Circular inputs

This method consumes **`realized_cap`**, each of which is itself computed from market
capitalisation. Its output is therefore **not independent evidence** of value: it is a
diagnostic or a cost basis, and the report must treat it as one (spec section 14,
"method circularity").

## Failure modes

- Realized cap is derived from market cap via the MVRV identity (spec 6.2), so this is a cost basis, not independent evidence of value.
- If Coin Metrics redefines either side of that identity every number here is silently wrong, which is what the identity assertions guard.
- Lost and never-moved coins keep their original acquisition price in the average, so the level is biased low for the oldest chains.

## Assumptions

- The last on-chain movement of a coin priced an economic transfer, not a wallet shuffle or an exchange's internal sweep.
- supply is the circulating supply (SplyCur), so a dormant or provably burned coin still divides the cap.
