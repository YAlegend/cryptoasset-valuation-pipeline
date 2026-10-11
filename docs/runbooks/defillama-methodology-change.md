# Runbook: a DefiLlama protocol is renamed or its methodology changes

**Symptom:** `methodology_hash` changes for a protocol, or a volume-anomaly
check flags a step change in `total_fees_24h` / `net_revenue_24h`.

**Why it matters:** a methodology revision redefines the revenue series, so
history before and after are not comparable and any DCF built across the break
is wrong (spec §9.2).

**Diagnose:**
```
curl -s "https://api.llama.fi/summary/fees/<protocol>" \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('methodology'))"
```
Compare against the stored `methodology_hash` in `silver/protocol_revenue`.

**Act:**
1. Record the break date in the protocol's method card.
2. Decide explicitly whether to truncate the series at the break or model the
   two regimes separately. Do not silently splice them.
3. For a rename, map the old and new names in config and keep both in the
   SCD2 history so the old vintage is still reconstructable.
