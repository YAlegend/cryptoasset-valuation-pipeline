# holding_time_correction

**Family:** A_monetary
**Citation:** Kampakis et al. (2024); corrects dissertation sections 3.1 and 3.6
**Returns a fair value:** no, it is a correction

## Data dependencies

- `velocity_total`

Minimum coverage density: **60%** per column. Minimum observations:
**365**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `window_rows` = `365`

## Circular inputs

This method consumes **`velocity_total`**, each of which is itself computed from market
capitalisation. Its output is therefore **not independent evidence** of value: it is a
diagnostic or a cost basis, and the report must treat it as one (spec section 14,
"method circularity").

## Failure modes

- The closed form assumes log-normal velocity: read anderson_darling_p, and treat a rejection as 'this ratio is an approximation for this asset'.
- It returns a correction factor, not a price (returns_fair_value is false); multiplying it into a fair value is a category error.
- velocity_total is market-cap-derived, so the input is circular even though the output is a diagnostic.
- A 365-row window spanning a volatility regime change mixes two distributions and the fitted sigma belongs to neither.

## Assumptions

- Daily velocity is log-normally distributed within the window.
- exp(sigma^2) captures the whole bias in inverting velocity, which is the correction Kampakis et al. (2024) derive.
