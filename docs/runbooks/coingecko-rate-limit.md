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
3. Coin Metrics covers price and market cap for everything except SOL, so a
   CoinGecko outage degrades velocity coverage rather than blocking the panel.
