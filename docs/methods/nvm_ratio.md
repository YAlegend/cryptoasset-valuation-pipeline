# nvm_ratio

**Family:** B_network
**Citation:** Network value to Metcalfe ratio (spec 10.3); the deviation series is the signal
**Returns a fair value:** no, it is a correction

## Data dependencies

- `market_cap_usd`
- `active_addresses`

Minimum coverage density: **80%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `fit_window_rows` = `365`
- `min_fit_rows` = `180`

## Failure modes

- Two Metcalfe references live in this family and they do not agree: this method refits a rolling 365-row log-log fit every row, while metcalfe_estimated uses an expanding window refitted every 30 rows. A reader comparing the two is comparing reference lines, not assets.
- Returns NO fair value by declaration: NVM is network value divided by Metcalfe-implied value, so publishing it as a price would be publishing the Metcalfe fit twice.
- It inherits the Metcalfe fit''s autocorrelated residuals wholesale, so a deviation is only as meaningful as that fit.
- The level of NVM is not comparable across assets; only its deviation from its own history is.
- The Metcalfe reference is fitted on the PRIOR window only, so today's market cap never enters the line it is measured against.

## Assumptions

- The Metcalfe fit on the trailing window is the right reference for today's network value.
