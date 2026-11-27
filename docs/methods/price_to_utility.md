# price_to_utility

**Family:** D_multiples
**Citation:** dissertation section 3.6, price-to-utility ratio
**Returns a fair value:** yes

## Data dependencies

- `price_usd`
- `velocity_total`
- `supply`

Minimum coverage density: **60%** per column. Minimum observations:
**730**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `min_history_rows` = `180`
- `staking_ratio` = `None`

## Circular inputs

This method consumes **`velocity_total`**, each of which is itself computed from market
capitalisation. Its output is therefore **not independent evidence** of value: it is a
diagnostic or a cost basis, and the report must treat it as one (spec section 14,
"method circularity").

## Failure modes

- The staking ratio is an unverified input and the free tier serves none, so with staking_ratio null the method refuses for every asset and names that reason - which is the honest answer, not a gap.
- It inverts velocity, the inversion holding_time_correction shows to be biased by exp(sigma^2).
- velocity_total is market-cap-derived, so this is a diagnostic rather than independent evidence of value.

## Assumptions

- A supplied staking_ratio is constant over the window and staked supply is genuinely unavailable for transactional use.
- 1 / velocity is a usable proxy for holding time, which is the assumption the correction method tests.
