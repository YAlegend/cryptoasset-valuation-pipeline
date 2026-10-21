# Cryptoasset Valuation

A free-data pipeline that rebuilds and tests the valuation methods of the 2022 dissertation
*Cryptoassets Valuation* (`Cryptoassets-Valuation-Dissertation-Yasir-Arafat.pdf`) against
live data, and adds the methods and statistics it lacked.

**These are valuation estimates under stated assumptions, not investment advice. These assets are
volatile and these methods disagree by design.**

## What it does

- **Ingests** daily data for 24 cryptoassets and 7 traditional series from free sources: Coin
  Metrics community, CoinGecko, DefiLlama and Yahoo Finance. Bitcoin coin ages come from BigQuery's
  public dataset, read through the free sandbox.
- **Stores** it as bronze/silver/gold Parquet and builds dbt marts in DuckDB
  (`data/cryptoval.duckdb`).
- **Estimates** 41 valuation methods in 6 families. Every method has a card in `docs/methods/`
  with its formula, assumptions, failure modes and citation. A method that cannot apply records
  why, rather than a number.
- **Tests** them:
  - walk-forward forecasts against a random walk;
  - cointegration of price with fundamentals;
  - volatility clustering and time-varying correlation;
  - Holm correction across each family of hypotheses.
- **Shows** the results in a local Streamlit dashboard.

## Reproduce

Python 3.14 and `make`:

```bash
make setup                      # .venv, dependencies, the package
ln -s ~/cryptoval-data data     # or: export CRYPTOVAL_DATA_ROOT=/absolute/path/to/data
make all                        # lint, test, ingest, dbt, estimate, score, econ, vol, factors, validate (~45 min)
make dashboard                  # http://localhost:8501
```

Notes:
- **Data location:** `data` is a symlink so that the data lives outside any synced folder. iCloud
  Drive's conflict copies once duplicated every row of the panel. Any **absolute** path works
  through `CRYPTOVAL_DATA_ROOT`, for Python and dbt alike.
- **No API keys:** `make all` needs none.
- **One run per day:** a source that fails partway makes `ingest` exit 1, and `validate` reports
  stale vendor data as a freshness failure.

### Bitcoin coin ages (monthly, by hand)

`make utxo` extracts the next range of Bitcoin's coin-age data from BigQuery. `make all` then
derives Liveliness, HODL waves, SLRV, RPV, STHPL and the Cointime prices from it. It needs:

1. A BigQuery **sandbox** project with no billing account, named as `bigquery_project` in
   `config/settings.yml`.
2. `gcloud auth application-default login`.

Every query is dry-run first and refused if it would breach the 0.9 TB monthly gate. The first
backfill (2009 to now) cost 561 GB, and a month's refresh is about 115 GB. The extraction is kept
out of `make all` because it spends that allowance. See
`docs/runbooks/bigquery-budget-exceeded.md`.

### Docker

```bash
docker build -t cryptoval .
docker run --rm -v ~/cryptoval-data:/data cryptoval                  # make all
docker run --rm -p 8501:8501 -v ~/cryptoval-data:/data cryptoval make dashboard
```

## Reports

| Report | Question |
|---|---|
| `reports/dissertation-comparison.md` | Which of the dissertation's findings hold in 2026 |
| `reports/modern-layer.md` | DCFs, multiples and the benchmark shelf the dissertation lacked |
| `reports/does-anything-beat-a-random-walk.md` | Does any model forecast better than persistence |
| `reports/does-price-anchor-to-fundamentals.md` | Is price cointegrated with its fundamentals |
| `reports/does-volatility-cluster.md` | GARCH, Markov regimes and DCC correlations |
| `reports/what-do-coin-ages-say.md` | Cointime, HODL waves, SLRV, RPV, STHPL |
| `reports/do-valuation-signals-earn-returns.md` | Decile sorts, LTW factors, Fama-MacBeth across 92 assets |

## Layout

| Path | What |
|---|---|
| `cryptoval/` | Sources, storage, contracts, methods, scoring, econometrics, volatility, coin ages, dashboard queries |
| `config/` | Assets, settings, one YAML per method, data contracts |
| `dbt/` | Staging models, marts and their tests |
| `dashboard/app.py` | The Streamlit app |
| `docs/methods/`, `docs/runbooks/` | Method cards and operational runbooks |
| `docs/superpowers/` | Design specs and implementation plans |
