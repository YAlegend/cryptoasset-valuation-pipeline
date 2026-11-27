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
   silver. Bronze is also a genuine second copy: every run lands in its own
   `run_id=` partition and a collision is refused rather than overwritten
   (`test_two_runs_on_the_same_date_both_survive`), so re-fetching the named
   asset-days today leaves today's earlier raw response record intact.

**Do not confuse this with the volume-anomaly WARNINGS.** Those are ratio
deviations from the mean of the previous 30 observations (30 days on a dense
daily panel) and they are non-blocking by design:
Bitcoin Cash's fork day (+246x), DOGE in July 2020 (+102x) and MANA in late
2021 (+89x) are genuine market events whose ratios are indistinguishable from
corruption. The ratio warns, the absolute bound blocks.

## What the volume-anomaly warning now means (2026-10-12)

The warning judges **only the newest observation** of each series against the mean of the
previous 30, and fires at `anomaly_threshold: 5.0`, meaning a deviation above +500%
(`|x - trailing| / trailing`). Both changes were forced by live data, not taste:

- The check used to scan the whole history and report the single worst day. On the 11-year
  Bitcoin series 946 of 4,299 days exceeded the old `0.5`, so it fired on 24 of 24 Coin Metrics
  ids and 24 of 24 CoinGecko ids on every run, re-reporting a 2015 event forever.
- `0.5` is +50%, two to three orders of magnitude below the events this check exists to flag
  (BCH +246x, DOGE +102x, MANA +89x). Latest-observation counts at each candidate
  (asset_metrics 25 ids, market_chart 24 ids): `0.5` -> 13 and 8 warnings; `5.0` -> 0 and 0;
  `10.0` -> 0 and 0. `5.0` is the lowest candidate that leaves normal operation quiet while
  still catching every documented event with a wide margin. DefiLlama's threshold could not be
  measured: `protocol_revenue` holds about three days per protocol, below the check's 5-row minimum,
  so it is inert there and its `5.0` is unvalidated until the history deepens.

**The narrowing is deliberate, not an oversight.** An anomaly in the *middle* of a multi-day
ingest is no longer flagged by this check. That is acceptable because the check is
non-blocking, the contract's absolute range maximum still blocks, and the quarantine step
(`cryptoval/quarantine.py`) now nulls and logs genuinely impossible values regardless of this
warning. The hard protection no longer rests here.

**Known gap: this is a spike-only detector.** The deviation is `|x - trailing| / trailing`, so
a collapse cannot exceed 1.0: a vendor silently returning near-zero volume scores about 1.0,
which warned at `0.5` and does not at `5.0`. On the live panel `trx_eth` (a near-dead
wrapper) scores exactly 1.0 on its latest day. There is a second blind path: `safe_trailing` treats a trailing mean of zero as unknown, so a
series whose previous 30 observations are all zero yields no deviation and the check returns
*passed* with the detail "no valid trailing baseline". A sustained vendor zero therefore does not
merely score below the threshold, it passes under a label an operator would read as "insufficient
data" rather than "collapsed". Whoever builds the deferred collapse signal must cover both paths.
A separate collapse signal is not built here.
