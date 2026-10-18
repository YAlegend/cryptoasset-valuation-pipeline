# aviv

**Family:** E_cost
**Citation:** Cointime Economics (Puell and Check): active value to investor value
**Returns a fair value:** no, it is a correction

## Data dependencies

- `market_cap_usd`
- `liveliness`
- `investor_cap_usd`

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

- Inherits the MVRV identity's fragility (spec 6.2) through investor cap, which is realized cap minus thermocap.
- Liveliness here weights coinblocks by the protocol subsidy schedule, so lost coins never stop earning coinblocks.

## Assumptions

- Active cap is market cap times Liveliness.
