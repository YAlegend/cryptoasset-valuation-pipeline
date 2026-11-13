# mvrv_reversion

**Family:** E_cost
**Citation:** Murad Mahmudov & David Puell (2018), MVRV
**Returns a fair value:** yes

## Data dependencies

- `realized_cap`
- `mvrv`
- `supply`

Minimum coverage density: **50%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `min_history_rows` = `180`

## Circular inputs

This method consumes **`realized_cap`, `mvrv`**, each of which is itself computed from market
capitalisation. Its output is therefore **not independent evidence** of value: it is a
diagnostic or a cost basis, and the report must treat it as one (spec section 14,
"method circularity").

## Failure modes

- Inherits the MVRV identity's fragility (spec 6.2): realized_cap and mvrv are two views of one vendor computation.
- Reversion assumes MVRV oscillates around a stable level; a regime change (ETF flows, a halving, the Merge) moves the level instead.
- The trailing median is computed on a history dominated by one cycle, so the target is often 'last cycle's price' restated.

## Assumptions

- MVRV reverts to its own trailing central tendency within min_history_rows.
- Realized cap is measured consistently by the vendor across the whole window.
