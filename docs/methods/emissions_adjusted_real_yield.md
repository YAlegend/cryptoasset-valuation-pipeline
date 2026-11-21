# emissions_adjusted_real_yield

**Family:** C_cashflow
**Citation:** Spec 10.4, emissions-adjusted real yield
**Returns a fair value:** no, it is a correction

## Data dependencies

- `total_fees_usd`
- `issuance_usd`
- `market_cap_usd`

Minimum coverage density: **50%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `incentive_share_usd` = `0.0`
- `window_rows` = `365`

## Failure modes

- INCOMPLETE BY CONSTRUCTION: spec 10.4 defines real yield as fee share + incentive share minus dilution, and no free source serves per-asset incentive emissions. The incentive term is a parameter defaulting to zero, every row carries incentive_share_available=false, and the published number is therefore a LOWER bound on the incentive side and must not be read as the complete figure.
- Returns no fair value by declaration: a yield is not a price.
- Dilution is measured as issuance over market cap at today's price, so a price crash inflates the apparent dilution rate without any change in emissions.
- For ETH the real yield is deeply negative - annualised fees of $0.18bn against issuance of $2.49bn - which is the same arithmetic that makes eth_three_stage_dcf refuse.

## Assumptions

- Fees accrue to holders, and issuance dilutes them at the current market cap.
