# fdv_to_revenue

**Family:** D_multiples
**Citation:** FDV/revenue, spec 10.5; FDV assumes full dilution at today's price
**Returns a fair value:** yes

## Data dependencies

- `price_usd`
- `max_supply`
- `net_revenue_usd`

Minimum coverage density: **50%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `denominator` = `'net_revenue_usd'`
- `target_multiple` = `15.0`
- `window_rows` = `365`

## Failure modes

- FDV assumes the entire maximum supply exists TODAY at today's price, which is the critique spec 10.4 raises against it: it values tokens that do not exist yet as though they did.
- An asset with no maximum supply is REFUSED, never given its total supply instead. ETH has no max supply; substituting one would publish a market-cap multiple under an FDV name.
- max_supply is a point-in-time vendor figure captured on supply_as_of and applied to every date in the series. It is a protocol constant in the ordinary case, but a governance change to the cap would be back-applied to history with no trace but this line.
- This is NOT CoinGecko's fully_diluted_valuation, which is price x TOTAL supply: the two differ by a factor of 12 for MKR and 4.3% for BTC.

## Assumptions

- The maximum supply is a constant over the series, and 15x revenue is a defensible multiple.
