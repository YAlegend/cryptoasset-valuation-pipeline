# true_market_mean

**Family:** E_cost
**Citation:** Cointime Economics (Puell and Check): true market mean price
**Returns a fair value:** yes

## Data dependencies

- `investor_cap_usd`
- `active_supply`

Minimum coverage density: **50%** per column. Minimum observations:
**30**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- (none)

## Circular inputs

This method consumes **`investor_cap_usd`**, each of which is itself computed from market
capitalisation. Its output is therefore **not independent evidence** of value: it is a
diagnostic or a cost basis, and the report must treat it as one (spec section 14,
"method circularity").

## Failure modes

- Inherits the MVRV identity's fragility (spec 6.2) through investor cap.
- Refuses on any day investor cap is not positive.

## Assumptions

- Investor cap (realized cap minus thermocap) is the capital investors, not miners, put in.
