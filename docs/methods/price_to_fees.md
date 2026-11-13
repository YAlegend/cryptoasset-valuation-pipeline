# price_to_fees

**Family:** D_multiples
**Citation:** P/F, spec 10.5; denominator discipline per spec 10.4
**Returns a fair value:** yes

## Data dependencies

- `price_usd`
- `supply`
- `total_fees_usd`

Minimum coverage density: **50%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `denominator` = `'total_fees_usd'`
- `target_multiple` = `20.0`
- `window_rows` = `365`

## Failure modes

- GROSS FEES, not revenue: most of this number is routed to liquidity providers and never reaches a token holder. P/F is systematically the most flattering of the three denominators, which is why the row names its denominator.
- target_multiple 20.0 is a CHOICE, not a measurement: the estimate scales linearly in it, and no cross-asset work has yet established what multiple this sector supports.
- An annualised trailing window assumes the last year of fees repeats; for a protocol in a fee regime change it does not.
- The protocol rollup sums a symbol's configured protocols, so a symbol whose slug list is incomplete understates its own fees.

## Assumptions

- Trailing-window fees annualise, and 20x fees is a defensible multiple.
