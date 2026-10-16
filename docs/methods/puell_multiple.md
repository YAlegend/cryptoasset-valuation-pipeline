# puell_multiple

**Family:** E_cost
**Citation:** David Puell (2019), Puell multiple; spec 6.2 derives it from free inputs
**Returns a fair value:** yes

## Data dependencies

- `issuance_usd`
- `price_usd`

Minimum coverage density: **90%** per column. Minimum observations:
**730**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `target_multiple` = `1.0`
- `window_rows` = `365`

## Failure modes

- A miner-revenue oscillator, not a valuation: fair value here means "the price at which daily issuance revenue equals its own trailing-year average", which is a cycle indicator wearing a price's clothes.
- A halving halves issuance overnight, so the multiple steps by construction and the step is not information about price.
- Post-Merge ETH has no mining issuance in the miner sense, so the series describes staking issuance instead and the analogy to Puell's own construction breaks.

## Assumptions

- Issuance revenue mean-reverts to its trailing-year average.
