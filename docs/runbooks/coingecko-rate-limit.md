# Runbook: CoinGecko 429 rate limiting

**Symptom:** `make backfill` fails with `HTTP 429` from
`api.coingecko.com`, or the run takes far longer than expected.

**Why it matters:** the Demo tier is capped (roughly 10k calls/month, with a
per-minute ceiling), and a full backfill across 24 assets is call-heavy.

**Act:**
1. The adapter already retries with exponential backoff and sleeps
   `sleep_between_assets_s` between assets. Raise that value first.
2. Backfill in slices: `make backfill` fixes the start date, so call the CLI directly,
   e.g. `.venv/bin/python -m cryptoval.cli backfill --sources coingecko --start 2024-01-01 --end 2024-12-31`,
   a year at a time rather than the whole history in one pass.
3. A rate limit no longer discards the whole pass: the adapter keeps every asset
   already fetched, stops at the failure, and the run is recorded as
   `status: partial` with the failing asset named (fix round 1, Ruling D).
   When the window resets, re-running just the assets that were missed is safe
   for bronze: every run lands in its own `run_id=` partition and nothing is
   overwritten, so the response record for the assets fetched earlier survives.
   Silver keeps both vintages either way, because SCD2 only touches the keys
   present in the incoming frame.
4. Coin Metrics covers price and market cap for everything except SOL, so a
   CoinGecko outage degrades velocity coverage rather than blocking the panel.

## The keyless API only serves 365 days

**Symptom:** `make backfill` fails for `coingecko` with an HTTP 401 whose body is
`error_code: 10012` — "Public API users are limited to querying historical data
within the past 365 days".

This is a hard cap, not a rate limit: without an API key, history before
`today - 365d` is unobtainable from CoinGecko at any pace. Verified 2026-10-11.

**Act:**
1. Do not retry and do not report a 2015 backfill as done. Cap the range:
   `--sources coingecko --start $(python3 -c "import datetime;print(datetime.date.today()-datetime.timedelta(days=364))")`.
2. For full-history volume (the equation-of-exchange `T` term) use Coin Metrics'
   `volume_reported_spot_usd_1d`, which the community tier serves back to
   2015-01-05 for free. It is REPORTED SPOT volume from Coin Metrics' exchange
   set, a different construction from CoinGecko's `total_volume`: the two are
   not interchangeable, so state which one a velocity figure uses.
3. CoinGecko's remaining jobs are the recent window and the assets Coin Metrics'
   free tier does not cover (SOL).

## The keyless ceiling tightened (2026-10-12)

**Symptom:** `make ingest` returned `status: partial` with HTTP 429 on three
consecutive attempts, failing at roughly asset 2, 21 and 21 of 24, the third after
a seven-minute backoff.

The previous default, `sleep_between_assets_s = 2.5` (about 24 calls a minute), is
no longer sustained by the keyless tier, and the adapter's own 62 seconds of
internal backoff cannot clear a limit that the pass itself keeps re-tripping.

**Act:** the default is now `6.0` (about 10 calls a minute), so a 24-asset pass
takes roughly 2.5 minutes before any retry. That is the remedy step 1 above
prescribes. If 429s return, raise it again rather than retrying at the old pace.

## The keyless ceiling tightened (2026-10-12)

**Symptom:** `make ingest` returned `status: partial` with HTTP 429 on three
consecutive attempts, failing at roughly asset 2, 21 and 21 of 24, the third after
a seven-minute backoff.

The previous default, `sleep_between_assets_s = 2.5` (about 24 calls a minute), is
no longer sustained by the keyless tier, and the adapter's own 62 seconds of
internal backoff cannot clear a limit that the pass itself keeps re-tripping.

**Act:** the default is now `6.0` (about 10 calls a minute), so a 24-asset pass
takes roughly 2.5 minutes before any retry. That is the remedy step 1 above
prescribes. If 429s return, raise it again rather than retrying at the old pace.

## Measured: even 6 s pacing clears only part of a 24-asset pass (2026-10-12)

With `sleep_between_assets_s = 6.0`, three full-list `ingest --trailing-days 300 --sources coingecko`
runs (after cool-downs of 0, 3 and 10 minutes) all ended `status: partial` on HTTP 429:

| Run | Assets stored | Rows | Failed at |
|-----|---------------|------|-----------|
| 1   | about 9       | 2709 | `tron` |
| 2   | about 7       | 2107 | `avalanche-2` |
| 3   | about 9       | 2709 | `tron` |

The failure point is deterministic: each pass fetches the same leading assets and stops at the
first 429 (the adapter keeps what it has and stops, by design), so repeating the full list
re-fetches the same head and the tail never arrives. A longer cool-down did not change that.

**What survives:** silver accumulates across passes. SCD2 only touches the keys present in the
incoming frame, so an earlier run's assets are not lost, and bronze lands each run in its own
`run_id=` partition.

**Practical consequence:** `velocity_total` and the SOL/TRX market-cap fill reach completeness over
several runs, not in one. Until the tail assets have been fetched at least once, treat CoinGecko
coverage as incomplete (`market_chart` holds only the assets some pass reached).

**Not done here, deliberately:** a CLI asset filter (to fetch only the missing tail), a change to
the stop-on-429 semantics, and a further rise in pacing. They are real options for a later plan.
