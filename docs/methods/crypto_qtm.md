# crypto_qtm

**Family:** A_monetary
**Citation:** dissertation section 3.1, crypto quantity theory of money
**Returns a fair value:** yes

## Data dependencies

- `volume_total_usd`
- `supply`
- `price_usd`

Minimum coverage density: **60%** per column. Minimum observations:
**180**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `circulating_fraction` = `1.0`
- `velocity_daily` = `0.04`
- `velocity_source` = `'parameter'`

## Failure modes

- The estimate scales as 1/velocity, so it is mostly a restatement of velocity_daily: vary the parameter and read the spread, never the point.
- volume_total_usd is exchange volume, not monetary transaction demand, so the numerator measures speculation rather than use.
- holding_time_correction shows the velocity inversion is biased by exp(sigma^2), so the level is wrong even when the direction is right.
- circulating_fraction 1.0 treats every coin as monetary base, including coins that never move.

## Assumptions

- An exogenous daily velocity of 0.04 (about 14.6 turns a year), supplied rather than measured - which is what keeps this reading non-circular.
- MV = PQ holds at daily frequency on a market whose real output Q is unobserved.
