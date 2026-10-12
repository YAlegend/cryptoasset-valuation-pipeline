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
