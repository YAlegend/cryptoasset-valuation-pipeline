# defi_net_revenue_dcf

**Family:** C_cashflow
**Citation:** Spec 10.4, DeFi net-revenue DCF - the retained share, never gross fees
**Returns a fair value:** yes

## Data dependencies

- `net_revenue_usd`
- `revenue_regime`
- `supply`
- `price_usd`

Minimum coverage density: **50%** per column. Minimum observations:
**180**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `discount_rate` = `0.2`
- `growth` = `[0.15, 0.08, 0.02]`
- `max_terminal_share` = `0.8`
- `sensitivity` = `{'discount_rate': [0.15, 0.2, 0.3]}`
- `stage_years` = `[3, 3, 4]`
- `terminal_multiple` = `10.0`
- `window_rows` = `365`

## Failure modes

- NET revenue, never gross fees: the gross figure is mostly routed to liquidity providers. A protocol whose revenue_regime is fees_fully_distributed is refused, because discounting a cash flow it does not retain would value money that never reaches a holder.
- Per-token value divides by TODAY'S supply. It does not project emissions forward, so for a token still minting it overstates the per-token value; supply_effect_counter_model is the method that prices the supply path.
- DefiLlama's per-protocol revenue definition is the vendor's, and its methodology hash changing is a silent redefinition of this method's denominator - the hash is stored for that reason.
- A discount rate of 20% is a judgement about a sector with no established cost of capital.

## Assumptions

- The protocol retains the revenue DefiLlama reports as revenue.
