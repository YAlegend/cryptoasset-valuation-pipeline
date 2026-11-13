# mayer_multiple

**Family:** E_cost
**Citation:** Trace Mayer, Mayer multiple (200-day moving average)
**Returns a fair value:** yes

## Data dependencies

- `price_usd`

Minimum coverage density: **90%** per column. Minimum observations:
**400**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `target_multiple` = `1.0`
- `window_rows` = `200`

## Failure modes

- The purest form of the circularity this project exists to flag: fair value is a moving average OF PRICE, so the method cannot disagree with price for long and its residual is a momentum signal, not a valuation gap. Reported as a diagnostic.
- The 200-row window is a convention with no derivation behind it; target_multiple 1.0 is likewise a choice, and the estimate scales linearly in it.
- A 200-row window on a gapped series spans more than 200 calendar days, so the average is over a longer stretch than its name claims.

## Assumptions

- Price reverts to its own 200-row moving average.
