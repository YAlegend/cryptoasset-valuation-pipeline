# active_price

**Family:** E_cost
**Citation:** Cointime Economics (Puell and Check): active price
**Returns a fair value:** yes

## Data dependencies

- `realized_cap`
- `active_supply`

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

- Inherits the MVRV identity's fragility (spec 6.2) through realized cap.
- A cost basis spread over the active supply only, so it rises as Liveliness falls with no change in price.

## Assumptions

- Vaulted coins carry no marginal cost basis.
