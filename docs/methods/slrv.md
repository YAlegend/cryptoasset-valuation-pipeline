# slrv

**Family:** E_cost
**Citation:** Dissertation 5.5, short-to-long-term realised value ratio
**Returns a fair value:** no, it is a correction

## Data dependencies

- `hodl_rc_0d`
- `hodl_rc_6m_1y`

Minimum coverage density: **50%** per column. Minimum observations:
**30**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- (none)

## Failure modes

- Daily grain: the "1-day wave" is coins created on the day and unspent at its end, not a rolling 24 h band (Plan 3b Ruling 2), so the level is not directly comparable with Glassnode's SLRV.
- Returns no fair value by declaration; the dissertation reads thresholds (below 0.04 in bear markets).

## Assumptions

- Realized-cap weighting values each coin at the daily close of the day it was created.
