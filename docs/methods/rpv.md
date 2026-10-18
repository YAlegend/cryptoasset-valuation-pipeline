# rpv

**Family:** E_cost
**Citation:** Dissertation 6.1, realised profits-to-value ratio
**Returns a fair value:** no, it is a correction

## Data dependencies

- `realized_profit_usd`
- `market_cap_usd`

Minimum coverage density: **50%** per column. Minimum observations:
**30**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- (none)

## Failure modes

- Profit is measured between daily closes (creation day, spend day), so intraday round trips and same-day moves realise zero profit.
- Exchange-internal and change-output spends count as realisations; the chain cannot tell a sale from a self-transfer.

## Assumptions

- A spent output is a realisation at that day's close price.
