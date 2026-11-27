# Runbook: equity prices are missing or silently incomplete

**Symptom:** `make validate` reports `equity_prices: no data ingested`, or an
equity ticker is absent from `silver/equity_prices` while others loaded fine.

**Why it matters:** the equity comparables (MSTR, COIN, miners, SPY) are the
market-approach anchor. yfinance fails PARTIALLY and without raising, so a
missing ticker looks like a complete run.

**Diagnose:**
1. Check the run manifest for the `equities` row:
   ```
   grep '"source": "equities"' data/gold/run_manifest.jsonl | tail -1
   ```
2. Compare tickers requested against tickers stored:
   ```
   .venv/bin/python -c "
   import pandas as pd
   from cryptoval.config import load_config
   _, assets = load_config()
   want = {a.yahoo_id for a in assets if a.is_equity and a.yahoo_id}
   got = set(pd.read_parquet('data/silver/equity_prices/part-0.parquet')['asset'])
   print('missing:', sorted(want - got))
   "
   ```
3. Re-run one missing ticker alone and read the log:
   ```
   .venv/bin/python -c "import yfinance; print(yfinance.download(['TSLA'], start='2026-09-01', end='2026-10-11', auto_adjust=True, progress=False).tail())"
   ```

**Known causes:**
1. **`database is locked`.** yfinance keeps a sqlite timezone cache; when two
   processes (or a stale lock) touch it, it logs `database is locked` for the
   affected tickers, DROPS them, and still returns a frame for the rest. Seen on
   2026-10-11 for TSLA and META. Act: make sure no other yfinance process is
   running, remove the cache (`rm -rf ~/.cache/py-yfinance`), re-run, then
   re-check with step 2. Never report the run as complete until that list is
   empty.
2. **Unhashable missing values.** Fixed in fix round 1 (Ruling A): the adapter
   writes `np.nan`, not `pd.NA`, for `market_cap_usd` / `volume_usd`, because
   `stable_hash` cannot encode `NAType` and the SCD2 merge therefore raised on
   every equity row. If `cannot encode object of type NAType` reappears
   anywhere, look for a new `pd.NA` assignment rather than relaxing the hash.
3. **Ticker renamed or delisted.** Map the old and new symbols in
   `config/assets.yml` and keep both in the SCD2 history, as in the
   asset-rebrand runbook.
