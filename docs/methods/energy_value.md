# energy_value

**Family:** E_cost
**Citation:** Charles Edwards / Capriole (2019), energy value
**Returns a fair value:** yes

## Data dependencies

- `hash_rate`
- `issuance_usd`
- `price_usd`

Minimum coverage density: **90%** per column. Minimum observations:
**400**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `electricity_usd_per_kwh` = `0.05`
- `hashes_per_unit` = `1000000000000.0`
- `joules_per_hash` = `2e-11`
- `window_rows` = `365`

## Failure modes

- NOT INDEPENDENT OF HAYES. Energy value is 2 x the trailing-year mean of exactly the Hayes cost per coin this repo already computes, and it is implemented by calling that function. Treat the two as one piece of evidence; counting them twice in a composite manufactures agreement.
- The factor of two is Capriole's assumption about the ratio of value to energy input, not a measurement, and the estimate scales linearly in it.
- It inherits every Hayes failure mode - the efficiency and electricity-price parameters, the calibrated hash-rate unit, and proof-of-work only.

## Assumptions

- Value is twice the energy cost of issuance, averaged over a trailing year.
- The same fleet efficiency and electricity price as hayes_cost_of_production.
