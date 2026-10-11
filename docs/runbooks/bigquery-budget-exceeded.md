# Runbook: BigQuery byte budget exceeded

**Symptom:** `BudgetExceeded: query would scan N bytes but only M bytes remain`.

**Why it matters:** this is the gate working as designed (spec §6.3), not a
failure. The free-only decision means the query must wait, not be paid for.

**Act:**
1. Check the ledger: `bytes_scanned_in_month` in `gold/run_manifest.jsonl`.
2. Narrow the query — tighter `block_timestamp_month` pruning, fewer columns —
   and re-run the dry run.
3. If it still will not fit, stage the remaining epochs into next calendar month.
   Phase 3b is explicitly allowed to span months.
4. Never enable billing to get past this. That requires a new explicit decision
   recorded in the spec.
