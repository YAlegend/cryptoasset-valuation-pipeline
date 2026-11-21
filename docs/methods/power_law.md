# power_law

**Family:** F_shelf
**Citation:** Santostasi power law; contested - see the 2026 nine-objections paper (spec 10.7)
**Returns a fair value:** yes

## Data dependencies

- `price_usd`
- `genesis_date`

Minimum coverage density: **90%** per column. Minimum observations:
**730**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `min_fit_rows` = `365`

## Failure modes

- Published here AS CONTESTED (spec 10.7), not as a model: the claimed mechanistic derivation has nine documented methodological objections against it, and the fit is to time alone.
- Regressing log price on log days since genesis fits ANY long-lived appreciating asset: the explanatory variable is the calendar, so the fit cannot fail for a reason that is about the asset.
- An asset without a configured genesis date is refused rather than given the panel's first observation, which would be the vendor's coverage start, not the asset's birth.
- The residuals are strongly autocorrelated by construction, so the confidence interval on beta is optimistic.

## Assumptions

- A configured genesis date exists for the asset and is its real launch.
- Time since genesis is the explanatory variable, which is the objection as much as the model.
