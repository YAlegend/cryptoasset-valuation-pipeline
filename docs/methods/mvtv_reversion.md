# mvtv_reversion

**Family:** E_cost
**Citation:** dissertation section 3.5, market-value-to-thermocap
**Returns a fair value:** yes

## Data dependencies

- `thermocap_usd`
- `mvtv`
- `supply`

Minimum coverage density: **50%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `min_history_rows` = `180`

## Circular inputs

This method consumes **`mvtv`**, each of which is itself computed from market
capitalisation. Its output is therefore **not independent evidence** of value: it is a
diagnostic or a cost basis, and the report must treat it as one (spec section 14,
"method circularity").

## Failure modes

- Inherits the MVRV identity's fragility (spec 6.2) through mvtv, which is market cap divided by thermocap.
- thermocap_has_gap understates the denominator, which inflates the ratio and therefore the reversion target.
- Thermocap grows monotonically, so the ratio has a structural downward drift that reversion reads as cheapness.

## Assumptions

- MVTV reverts to its own trailing central tendency within min_history_rows.
- Cumulative issuance value is comparable across the window despite issuance-schedule changes.
