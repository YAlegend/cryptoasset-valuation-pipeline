# mvrv_z_score

**Family:** E_cost
**Citation:** Awe and Wonder / Murad Mahmudov, MVRV Z-score; dissertation section 3.5
**Returns a fair value:** yes

## Data dependencies

- `market_cap_usd`
- `realized_cap`
- `supply`

Minimum coverage density: **80%** per column. Minimum observations:
**730**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `target_z` = `0.0`
- `window_rows` = `365`

## Circular inputs

This method consumes **`realized_cap`**, each of which is itself computed from market
capitalisation. Its output is therefore **not independent evidence** of value: it is a
diagnostic or a cost basis, and the report must treat it as one (spec section 14,
"method circularity").

## Failure modes

- Circular by construction: realized cap is derived from market cap through the MVRV identity (spec 6.2), so this scores market cap against a transform of itself.
- The standard deviation is of market cap in dollars, so the Z-score is not comparable across assets of different size, nor across eras of one asset.
- A trailing-window standard deviation on a trending series understates dispersion early and overstates it after a crash.
- At the default target_z of 0 the fair value IS realized_price's, to the last digit: the standard deviation reaches only the z_score diagnostic. The two methods are one piece of evidence, and averaging both into a composite double-counts realized cap. Every row says so in derived_from.
- The standard deviation is a prior-only rolling sample stdev (ddof=1) over window_rows, not the canonical MVRV Z-score's expanding population stdev, so the z-score is not comparable with published charts.

## Assumptions

- Market cap reverts toward realized cap, and the trailing window measures the normal spread.
