# nvt_golden_ratio

**Family:** D_multiples
**Citation:** Dmitry Kalichkin (2018), NVT golden ratio
**Returns a fair value:** yes

## Data dependencies

- `market_cap_usd`
- `volume_total_usd`
- `supply`

Minimum coverage density: **60%** per column. Minimum observations:
**1095**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `ma_rows` = `730`
- `volume_column` = `'volume_total_usd'`

## Failure modes

- min_rows is 1,095 - a 730-row moving average plus a year to use it - which exceeds free-tier history for most of the universe, so most assets refuse. That wall of refusals is the measurement, not a bug.
- The 730-day mean carries two years of a different liquidity regime into today's signal.
- Same volume-definition objection as nvt_reversion: read inputs_used.volume_column.

## Assumptions

- A 730-row trailing mean of NVT is the right normaliser (Kalichkin's own choice), and the ratio of today's NVT to it is comparable across assets.
- volume_total_usd is the transaction proxy.
