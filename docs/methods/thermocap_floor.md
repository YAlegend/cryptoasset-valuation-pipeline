# thermocap_floor

**Family:** E_cost
**Citation:** Nick Carter / Coinmetrics thermocap; dissertation section 3.5
**Returns a fair value:** yes

## Data dependencies

- `thermocap_usd`
- `supply`

Minimum coverage density: **50%** per column. Minimum observations:
**30**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- (none)

## Failure modes

- thermocap_has_gap means cumulative issuance value is understated, so the floor reads too low and anything built on it reads too high.
- Thermocap counts issuance paid to miners and not fees, so a fee-heavy chain's security spend is understated.
- ETH's issuance changes meaning at the Merge, so a floor spanning that date averages two regimes.

## Assumptions

- Cumulative miner revenue is a lower bound on the capital that had to be spent to bring the supply into existence.
- Security spend is a proxy for value, which is an argument about cost and not about demand.
