# Runbook: a metric leaves the Coin Metrics free tier

**Symptom:** `make test-integration` fails on
`test_coinmetrics_free_tier_still_serves_mvrv`, or `make validate` reports
`asset_metrics/not_null` failures for `CapMVRVCur`.

**Why it matters:** `realized_cap` is derived as `CapMrktCurUSD / CapMVRVCur`
(spec §6.2). Losing `CapMVRVCur` silently removes MVRV, MVRV Z-score,
Thermocap comparisons and Investor Cap — the whole cost approach.

**Diagnose:**
```
curl -s "https://community-api.coinmetrics.io/v4/catalog-v2/asset-metrics?assets=btc" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(sorted(d['data'][0]['metrics']))"
```
A `403` with "not available with supplied credentials" confirms the metric moved to Pro.

**Act:**
1. Do not approximate it and do not substitute another metric. Remove it from
   `FREE_METRICS` and let the affected methods report `applicable=False`.
2. Bronze retains every value already fetched, so the historical series survives;
   only new dates are lost. Say so explicitly in the report.
3. Re-run `make validate` to confirm the remaining checks pass.
