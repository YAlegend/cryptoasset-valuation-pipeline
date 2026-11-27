# price_to_tvl

**Family:** D_multiples
**Citation:** P/TVL, spec 10.5
**Returns a fair value:** yes

## Data dependencies

- `price_usd`
- `supply`
- `tvl_usd`

Minimum coverage density: **50%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `annualise` = `False`
- `denominator` = `'tvl_usd'`
- `target_multiple` = `0.3`
- `window_rows` = `30`

## Failure modes

- TVL is not revenue and not a claim on assets: it is deposits, which can leave in a day.
- It is reflexively inflated by the price of the very token being valued, when that token is itself collateral in the protocol being measured.
- TVL is a STOCK, so its denominator is a trailing mean in dollars and is NOT annualised. The target multiple is therefore a price-to-deposits ratio, not a yield.
- DefiLlama's TVL excludes a protocol's own staked governance token. LINK reads 0.0 on all 1,039 days for this reason while $518M sits under the vendor's `staking` key, so LINK is refused here rather than valued on a zero denominator.

## Assumptions

- Deposits support a 0.3x price-to-TVL ratio, and a 30-day mean represents the current balance.
