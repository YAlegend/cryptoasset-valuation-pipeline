# j_curve

**Family:** A_monetary
**Citation:** Burniske (2017), crypto J-curve; dissertation section 3.2
**Returns a fair value:** yes

## Data dependencies

- `volume_total_usd`
- `supply`
- `price_usd`

Minimum coverage density: **60%** per column. Minimum observations:
**180**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `velocity_daily` = `0.04`
- `velocity_source` = `'parameter'`

## Failure modes

- The dissertation's printed CUV formula is dimensionally incoherent - it divides a stock by a rate with no holding period - so this implements the economically intended form. That deviation is the headline, not a detail.
- The split between current utility and speculative value is unobservable, so the decomposition is an assumption dressed as a measurement.
- Inherits crypto_qtm's sensitivity to velocity_daily and the same exchange-volume objection.

## Assumptions

- An exogenous daily velocity of 0.04.
- Today's utility value is the discounted monetary value of today's transaction demand, and everything above it is discounted expectation.
