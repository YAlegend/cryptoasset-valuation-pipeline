# metcalfe_estimated

**Family:** B_network
**Citation:** Metcalfe (1995); Alabi (2017); Peterson (2018). beta estimated, never assumed to be 2.0
**Returns a fair value:** yes

## Data dependencies

- `market_cap_usd`
- `active_addresses`
- `supply`

Minimum coverage density: **60%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `min_fit_rows` = `180`
- `refit_every_rows` = `30`

## Failure modes

- An active address is not a user: one user holds many addresses, and one exchange address serves many users.
- Exchange consolidation or an L2 migration moves the series with no change in adoption, and the fit reads it as adoption.
- A high R-squared between two trending series is not evidence: read durbin_watson and residuals_autocorrelated first; below the DW floor the beta confidence interval is too narrow to quote.
- beta is refit on a trailing window, so it drifts, and a window spanning a regime break fits the break.

## Assumptions

- Market cap is a power function of active addresses with an exponent that is constant over the fit window - estimated, never assumed to be 2.0.
- Address counts are comparable over time for a given chain.
