# bond_target_yield

**Family:** C_cashflow
**Citation:** Spec 10.4, bond-style target yield = annual fees / (risk free + staking premium)
**Returns a fair value:** yes

## Data dependencies

- `total_fees_usd`
- `supply`
- `price_usd`

Minimum coverage density: **50%** per column. Minimum observations:
**180**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `denominator` = `'total_fees_usd'`
- `risk_free_rate` = `0.04`
- `staking_risk_premium` = `0.06`
- `window_rows` = `365`

## Failure modes

- A perpetuity with no growth term: value = annual fees / target yield assumes this year's fees forever, which is neither a forecast nor a floor.
- Both rate components are parameters set by hand. The risk-free rate is not read from a treasury series (the free tier here serves gold and the dollar index, not the 10-year), so it is as of whenever it was last edited - the value travels on every row for that reason.
- GROSS fees, unlike the two DCFs: this method values the whole fee stream regardless of who receives it, so for a protocol that routes most fees to liquidity providers it answers a question about the protocol rather than about the token.
- The staking risk premium is the entire judgement of the method compressed into one number.

## Assumptions

- Fee revenue is perpetual and flat, and 10% is the yield such a claim should offer.
