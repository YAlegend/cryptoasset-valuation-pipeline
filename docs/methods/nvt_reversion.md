# nvt_reversion

**Family:** D_multiples
**Citation:** Willy Woo / Chris Burniske (2017), NVT ratio; dissertation section 3.6
**Returns a fair value:** yes

## Data dependencies

- `market_cap_usd`
- `volume_total_usd`
- `supply`

Minimum coverage density: **60%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `min_history_rows` = `180`
- `volume_column` = `'volume_total_usd'`

## Failure modes

- Exchange volume is not on-chain transaction value, so this denominator measures trading rather than settlement: it is not the dissertation's NVT, and the card says so rather than pretending otherwise.
- The two volume series differ by roughly 5x, so read inputs_used.volume_column before comparing any two numbers.
- NVT reversion is invariant to a constant rescaling of volume, so a level difference between the series changes nothing while a shape difference changes everything.
- Reversion to a trailing median assumes a stable fair multiple, which a liquidity regime change breaks.

## Assumptions

- volume_total_usd is the transaction proxy (set volume_column to switch to reported spot volume).
- NVT mean-reverts within min_history_rows, and the volume it is built on is not materially wash-traded.
