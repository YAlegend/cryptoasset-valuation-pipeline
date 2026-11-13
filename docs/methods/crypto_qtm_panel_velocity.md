# crypto_qtm_panel_velocity

**Family:** A_monetary
**Citation:** dissertation section 3.1 as literally specified; retained to measure the circularity
**Returns a fair value:** yes

## Data dependencies

- `volume_total_usd`
- `supply`
- `price_usd`
- `velocity_total`

Minimum coverage density: **60%** per column. Minimum observations:
**180**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `circulating_fraction` = `1.0`
- `velocity_source` = `'panel'`

## Circular inputs

This method consumes **`velocity_total`**, each of which is itself computed from market
capitalisation. Its output is therefore **not independent evidence** of value: it is a
diagnostic or a cost basis, and the report must treat it as one (spec section 14,
"method circularity").

## Failure modes

- With velocity taken from the panel the expression collapses to price / circulating_fraction: every residual against price is zero by construction.
- It is retained only to measure the circularity against crypto_qtm, and must never be quoted as a fair value.
- If its fair value ever differs from price / f beyond floating-point tolerance, the derivation or the panel is wrong - investigate before publishing.

## Assumptions

- The dissertation's section 3.1 formula read literally, with velocity_total (volume / market cap) as the velocity it names.
- circulating_fraction 1.0, which is why the output equals the price exactly rather than approximately.
