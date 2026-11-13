# stock_to_flow

**Family:** F_shelf
**Citation:** PlanB (2019), Modeling Bitcoin Value with Scarcity; falsified post-2021
**Returns a fair value:** yes

## Data dependencies

- `market_cap_usd`
- `supply`
- `issuance_usd`
- `price_usd`

Minimum coverage density: **90%** per column. Minimum observations:
**730**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `flow_window_rows` = `365`
- `min_fit_rows` = `365`

## Failure modes

- Published here AS A FALSIFIED MODEL (spec 10.7): the 2021-2022 drawdown put price far below the model and it has not recovered the fit. A row from this method is a benchmark to beat, never a fair value to act on.
- The regression is of one trending series on another, so a high R-squared is weak evidence; the Durbin-Watson statistic travels on every row for exactly that reason.
- Flow is a trailing 365-row sum of issuance converted to coins at each day's price, so a price spike inside the window perturbs the stock-to-flow ratio through the denominator.
- A halving is a step change in flow, so the ratio jumps discontinuously and the fit straddles regimes rather than describing one.

## Assumptions

- Scarcity, measured as stock over annual flow, prices the asset.
- issuance_usd divided by price_usd recovers issuance in coins.
