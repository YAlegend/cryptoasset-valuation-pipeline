# Runbook: an asset is rebranded or re-ticker'd (e.g. MATIC to POL)

**Symptom:** an asset's series stops updating; `dim_asset.last_observed_on`
goes stale while the asset is plainly still trading.

**Act:**
1. Update `cm_id` / `gecko_id` in `config/assets.yml`; keep the historical
   symbol as the stable `symbol` so the SCD2 history stays joinable.
2. Run `.venv/bin/python -m cryptoval.cli reconcile --start <first-affected-date>` to see how many rows
   changed before writing anything.
3. If the sector also changed, note the date — Plan 4 needs point-in-time
   sector labels and a silent reclassification is look-ahead bias (spec §8.3).
