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

## Expected failures that are NOT bugs

`make validate` reports these on healthy live data. Do not "fix" them by
loosening the contract.

**Negative fees fail `protocol_revenue/ranges`.** On 2026-10-11, 12 of 2,991
protocols reported a negative `total_fees_24h` (worst: Lombard LBTC at
-23,915) and 11 a negative `total_fees_30d`. A negative fee is not a possible
quantity, so the `minimum: 0` bound is doing its job: the gate is telling you
the vendor's number is wrong, not that the bound is wrong. Count them before
and after any DefiLlama change:

```
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/silver/protocol_revenue/part-0.parquet')
cur = df[df['is_current']]
for c in ('total_fees_24h','total_fees_30d','net_revenue_24h'):
    s = pd.to_numeric(cur[c], errors='coerce')
    print(c, 'nulls', int(s.isna().sum()), 'negative', int((s<0).sum()))
print(cur.loc[pd.to_numeric(cur['total_fees_24h'],errors='coerce')<0, ['protocol','total_fees_24h']].to_string())
"
```

Act: exclude the named protocols from any multiple built on fees, and record
why in the method card. Do not clamp the value to zero — that invents revenue.

**Absent fees are normal.** 383 protocols are listed with no 24h fee figure, so
`total_fees_24h` is NOT in the contract's `required` list and a null does not
fail `not_null`. Null means "not reported", never zero; the denominator
discipline in `cryptoval/sources/defillama.py` depends on that distinction.

**Net revenue has its own endpoint.** `net_revenue_24h` comes from
`/overview/fees?...&dataType=dailyRevenue`, not from the fees payload — the
fees payload has no revenue field at all. If
`test_defillama_serves_net_revenue_distinctly_from_gross_fees` fails, or
`net_revenue_24h` is suddenly all-null, DefiLlama has changed that dataType:
check `https://api.llama.fi/overview/fees?dataType=dailyRevenue` by hand before
touching the adapter. A failed revenue call degrades the run to fees-only with
`status: partial` in the manifest; it does not silently zero the column.
