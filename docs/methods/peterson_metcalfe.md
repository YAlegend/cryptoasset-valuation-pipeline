# peterson_metcalfe

**Family:** B_network
**Citation:** Timothy Peterson (2018), Metcalfe's law as a model for Bitcoin's value; dissertation 3.3
**Returns a fair value:** yes

## Data dependencies

- `market_cap_usd`
- `active_addresses`
- `supply`

Minimum coverage density: **80%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `fit_window_rows` = `365`
- `min_fit_rows` = `180`

## Failure modes

- The dissertation specifies V = a x (n(n-1)/2 x 1/b_t) where b_t is bitcoin-days-destroyed, which the free tier does not serve. The constant a is fitted on the trailing window instead, so what is implemented is Peterson's FUNCTIONAL FORM with an estimated scale, not his published model, and the card says so.
- An active address is not a user: exchange and contract addresses inflate n, and n enters squared.
- A fitted scale constant absorbs any level error, so the fit can look good while the mechanism is wrong.
- The scale is the mean market cap per pair over the PRIOR window, so the value is the asset's own recent valuation scaled by how much the pair count has moved since: it is a momentum reading of the address series, not an independent price.

## Assumptions

- Network value scales with the number of potential connections among active addresses.
- The scale constant is stable across the trailing window.
