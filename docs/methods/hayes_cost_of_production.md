# hayes_cost_of_production

**Family:** E_cost
**Citation:** Hayes (2017), cost of production model; dissertation section 3.4
**Returns a fair value:** yes

## Data dependencies

- `hash_rate`
- `issuance_usd`
- `price_usd`

Minimum coverage density: **90%** per column. Minimum observations:
**180**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `electricity_usd_per_kwh` = `0.05`
- `hashes_per_unit` = `1000000000000.0`
- `joules_per_hash` = `2e-11`

## Failure modes

- A floor, not a target: a miner facing sunk capital sells below marginal cost rather than not at all.
- joules_per_hash and electricity_usd_per_kwh are assumptions, not measurements, and the estimate scales linearly in both.
- The hash-rate unit is calibrated rather than documented - measured as TH/s against BTC on 2026-10-12 ($50,668 implied against a price of $81,805, 1.61x; GH/s gives 1,615x and PH/s 0.0016x). A vendor unit change would rescale every estimate by 1,000 silently.
- Stranded, subsidised or self-generated power breaks the cost argument; ASIC capital cost is excluded, so the floor is understated.
- Post-Merge ETH has no hash rate, so the series ends rather than going to zero, and the density floor then refuses the asset.

## Assumptions

- A fleet-average efficiency of 2.0e-11 J/hash and electricity at $0.05/kWh.
- Coin Metrics' HashRate is in TH/s (hashes_per_unit 1.0e+12), as measured above.
- Energy is the only material marginal cost of production.
