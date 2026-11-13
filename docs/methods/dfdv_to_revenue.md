# dfdv_to_revenue

**Family:** D_multiples
**Citation:** dFDV, spec 10.4 - a haircut on locked supply rather than naive full dilution
**Returns a fair value:** yes

## Data dependencies

- `price_usd`
- `supply`
- `total_supply`
- `net_revenue_usd`

Minimum coverage density: **50%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `denominator` = `'net_revenue_usd'`
- `locked_haircut` = `0.5`
- `target_multiple` = `15.0`
- `window_rows` = `365`

## Failure modes

- locked_haircut 0.5 is an ASSUMPTION standing in for an unlock schedule the free tier does not serve, so dFDV here is a sensitivity parameter, not a vesting model.
- total_supply is a point-in-time snapshot captured on supply_as_of, NOT a series, while circulating supply IS a daily series. Locked supply is therefore (today's total - that day's circulating), which overstates the lock the further back the row sits. diagnostics.supply_age_days measures the gap on every row; a large value means the number is an artefact of the join, not of the protocol.
- The two supply figures come from DIFFERENT vendors: circulating is Coin Metrics, total is CoinGecko. They disagree for 7 of 24 assets (BNB by 41%, UNI 13%, ALGO 10%, POL, SNX, MKR, MANA), where circulating exceeds total and the locked portion would be negative. Those are refused rather than haircut upwards. Even where the two agree, the subtraction mixes two vendors' definitions of supply.

## Assumptions

- Half of locked supply is economically present today, and 15x revenue is a defensible multiple.
