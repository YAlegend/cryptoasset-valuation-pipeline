# sector_cohort_zscore

**Family:** D_multiples
**Citation:** Sector-cohorted relative valuation (spec 10.5)
**Returns a fair value:** no, it is a correction

## Data dependencies

- `market_cap_usd`
- `net_revenue_usd`
- `sector`

Minimum coverage density: **30%** per column. Minimum observations:
**1**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `min_cohort_size` = `3`
- `ratio` = `'market_cap_to_revenue'`
- `window_rows` = `365`

## Failure modes

- The cohort is dim_asset.sector, which is a hand-assigned label: a wrong label puts an asset in the wrong comparable set, and the z-score then measures the labelling rather than the asset.
- On this universe exactly ONE sector is ever scoreable. defi has 7 members and all 7 have revenue; smart_contract_l1 has 8 members but only ETH has any, so its cohort is one. Five sectors have a single member and seven symbols have no sector at all. Spec 10.5's cohorting is therefore available for one sector here, and nothing in this method is a cross-sector comparison.
- A z-score over three members is arithmetic, not statistics; min_cohort_size 3 is the floor at which the number exists at all, not the size at which it is trustworthy.
- A z-score over n members cannot exceed sqrt(n-1). On 7.3% of scored days one member sits at that ceiling and the rest are compressed 0.124 sigma apart on average, so the score says which asset is the outlier and cannot rank the others. diagnostics.cohort_saturated flags those days; the most recent scored day, 2026-10-13, is one of them (SNX at 4,856x revenue against 12-133x).
- Cross-sectional, so it is the only method here whose answer changes when an unrelated asset's data changes: a revision to one cohort member moves every member's score.
- min_density and min_rows are NOT consulted on the cross-sectional path -- both are statements about one symbol's history. This method does its own per-row refusing instead.

## Assumptions

- Assets within a configured sector are economically comparable.
- Market cap over annualised net revenue is the ratio worth cohorting on.
