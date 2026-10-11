# Runbook: a vendor publishes an impossible value

**Symptom:** `make validate` exits 1 with a `ranges` failure naming a maximum,
e.g.

```
FAIL asset_metrics/ranges: volume_reported_spot_usd_1d > 1000000000000.0
```

**Why it matters:** this is not market volatility. On 2026-10-11 Coin Metrics'
community tier served 14 asset-days of `volume_reported_spot_usd_1d` between
1e47 and 7e55 USD (12 for USDC, 2 for trx_eth) while the largest credible value
in the same panel was 7.5e10. Global daily spot volume is of order 1e11, so the
contract's `maximum: 1.0e+12` separates corruption from a real spike with an
order of magnitude of headroom either side.

**Diagnose** — list the offending rows and confirm they are not merely large:

```
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/silver/asset_metrics/part-0.parquet')
cur = df[df['is_current']].copy()
cur['v'] = pd.to_numeric(cur['volume_reported_spot_usd_1d'], errors='coerce')
bad = cur[cur['v'] > 1e12]
print(bad[['asset','metric_date','v']].to_string(index=False))
print('largest credible value:', cur.loc[cur['v'] <= 1e12, 'v'].max())
"
```

**Act:**
1. Re-fetch the named asset-days. A transient publishing error clears; a stable
   one is in the vendor's history.
2. If it persists, exclude those asset-days from any series that uses the
   metric and say so in the method card. Do NOT widen the bound to make the
   gate green, and do NOT overwrite the value with an interpolation — an
   invented number in the panel is worse than a known hole.
3. The SCD2 history keeps every vintage, so the exclusion is reversible from
   silver. Bronze is NOT a second copy for this purpose: a partition is keyed by
   `ingest_date` and a rewrite on the same date replaces it wholesale
   (`test_rewriting_a_date_replaces_the_partition_wholesale`), so re-fetching the
   named asset-days today overwrites today's raw response record.

**Do not confuse this with the volume-anomaly WARNINGS.** Those are ratio
deviations from the mean of the previous 30 observations (30 days on a dense
daily panel) and they are non-blocking by design:
Bitcoin Cash's fork day (+246x), DOGE in July 2020 (+102x) and MANA in late
2021 (+89x) are genuine market events whose ratios are indistinguishable from
corruption. The ratio warns, the absolute bound blocks.
