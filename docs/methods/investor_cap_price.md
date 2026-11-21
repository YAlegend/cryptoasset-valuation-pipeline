# investor_cap_price

**Family:** E_cost
**Citation:** dissertation section 3.5, investor capitalisation
**Returns a fair value:** yes

## Data dependencies

- `investor_cap_usd`
- `supply`

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

- Investor cap is realized cap minus thermocap, a difference of two large numbers, so a small relative error in either becomes a large one here.
- It is negative wherever thermocap exceeds realized cap, which is not a price and shows up as a refusal or a nonsense level.
- Inherits the MVRV identity's fragility (spec 6.2) through realized cap.

## Assumptions

- Subtracting miner cost from holder cost isolates non-mining investor capital.
- Both series are denominated and timestamped identically by the vendor.
