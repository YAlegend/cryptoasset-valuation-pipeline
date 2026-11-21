# Dashboard, README and Dockerfile — Design (Phase 5)

Parent spec: `2026-10-11-cryptoasset-valuation-pipeline-design.md` §12 (deliverables 1 and 3),
§13 phase 5.
Status: written 2026-10-18 without a review round, at the user's standing instruction to proceed
without check-ins. The judgement calls are listed under **Rulings**.

## 1. What this delivers

1. **`dashboard/app.py`**, a local Streamlit app (`make dashboard`). It reads only the dbt marts in
   `data/cryptoval.duckdb`: it writes nothing and computes no estimate of its own, except the DCF
   sliders, which re-run the registered DCF function on the estimate's own recorded inputs.
2. **`README.md`**, with a one-command reproduce path.
3. **`Dockerfile`**, running the same pipeline in a container with the data directory mounted.

Exit criterion (§13): every view renders from `make all`'s output, and the dashboard is served
locally.

## 2. Views (spec §12, deliverable 3)

Every page carries the §12 disclaimer and a freshness banner at the top.

| View | Content | Source |
|---|---|---|
| **Freshness banner** | Latest `metric_date` in the panel; latest coin-age date; panel vintage; a warning when the panel's latest date is more than 3 days old | `fct_asset_metrics_daily`, `fct_btc_cointime_daily` |
| **Football field** (per asset) | One horizontal bar per method: the min-max fair value over that method's grid cells on its latest applicable date, the date shown. A vertical line marks the asset's latest price. Circular methods are coloured apart. Refusing methods are listed below, with their reasons | `fct_valuation_estimate`, `dim_method`, panel price |
| **Cycle indicators** (BTC) | A selectable time series per indicator: MVRV, Mayer multiple, Puell multiple, SLRV, RPV, STHPL, AVIV, Liveliness. The dissertation's stated thresholds are drawn where it states them | Signals' `diagnostics`; panel and coin-age columns |
| **Method agreement** | Heatmap of asset × method on each pair's latest applicable date, coloured by `log10(fair value / price)` (median over grid cells) | `fct_valuation_estimate`, panel price |
| **DCF sliders** | Pick an asset with an applicable `defi_net_revenue_dcf` or `eth_three_stage_dcf` row. Sliders cover the discount rate, three stage growths and the terminal multiple. The page shows the fair value per token from `three_stage_pv` on that row's recorded `annual_cash_flow_usd` and `supply`, beside the latest price and the terminal share | Latest applicable DCF row's `inputs_used` |

**What a signal is:** a method with `returns_fair_value = false`. Signals never appear in the
football field or the heatmap, because they carry no price.

## 3. Architecture

- **`cryptoval/dashboard/queries.py`** holds pure functions taking a DuckDB connection and returning
  DataFrames: `freshness`, `football_field`, `refusals`, `indicator_series`, `agreement`,
  `dcf_inputs`, and `dcf_value`, a thin wrapper over `three_stage_pv`. These are tested against a
  small in-memory DuckDB built in the test.
- **`dashboard/app.py`** is presentation only: Streamlit widgets and Altair charts.
- **Connections:** every read opens a read-only connection and closes it at once, through a
  `cache_data(ttl=600)` wrapper.
  - A long-lived connection would hold DuckDB's file lock and make `make all`'s dbt writes fail.
  - The DB path is `$CRYPTOVAL_DATA_ROOT/cryptoval.duckdb`, defaulting to `data/`.
- **Missing database:** the app shows the instruction to run `make all` instead of a traceback.
- **Smoke test:** `streamlit.testing.v1.AppTest` runs the app against a fixture DB and asserts no
  exception on any tab.

## 4. README and Dockerfile

- **README:**
  - what the project is;
  - the disclaimer;
  - setup (`make setup`, Python 3.14);
  - `make all`, `make dashboard` and the manual monthly `make utxo` with its BigQuery sandbox
    prerequisite;
  - the data symlink note (`ln -s ~/cryptoval-data data`, or `CRYPTOVAL_DATA_ROOT`);
  - a map of the reports and where the method cards live.
- **Dockerfile:**
  - `python:3.14-slim`, installing CPU torch exactly as CI does, then `requirements.txt` and the
    package;
  - `CRYPTOVAL_DATA_ROOT=/data`, with `/data` a volume;
  - default command `make all`, and port 8501 exposed for `make dashboard`;
  - build-verified locally with `docker build`.

## 5. Rulings

1. **Altair, not Plotly.** It ships with Streamlit, so there is no new charting dependency.
2. **Latest applicable date per (asset, method) rather than one global date.** Methods end on
   different days: the coin-age methods stop at the extraction cutoff, and the DCFs need 365 rows.
   One global date would blank half the field. The date is shown on every bar.
3. **Grid min-max as the bar.** The DCF and multiple grids are the method's own stated
   uncertainty. A point-only method draws as a tick. `band_low`/`band_high` are not used: they are
   populated by one method only.
4. **Indicators are BTC only** for now. Coin-age signals exist only for BTC. MVRV, Mayer and Puell
   exist for other assets, and the selector is wired so that adding assets later is a one-line
   change.
5. **No authentication, no hosting.** Local only, as Phase 5 says. Phase 6 hosts it.
6. **The Dockerfile is not run in CI.** Building it means downloading about 1 GB of torch, which
   would slow every push. It is verified by hand, once, in this plan.
