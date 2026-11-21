# Dashboard, README and Dockerfile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Streamlit dashboard over the dbt marts, with a football field, cycle indicators,
a method-agreement heatmap, DCF sliders and a freshness banner, plus a README and a Dockerfile.

**Architecture:** Pure query functions live in `cryptoval/dashboard/queries.py` and are tested
against an in-memory DuckDB. `dashboard/app.py` is presentation only. Every read opens and closes
its own read-only connection, so the app never holds the DuckDB lock that `make all` needs.

**Tech Stack:** Python 3.14, DuckDB, pandas, Streamlit 1.66.0, Altair 6.3 (bundled with Streamlit).

**Spec:** `docs/superpowers/specs/2026-10-18-dashboard-design.md`

## Global Constraints

- The §12 disclaimer is on every page: "These are valuation estimates under stated assumptions,
  not investment advice. These assets are volatile and these methods disagree by design."
- Read-only, connection-per-query. The DB path is
  `os.environ.get("CRYPTOVAL_DATA_ROOT", "data") + "/cryptoval.duckdb"`.
- Signals (`dim_method.returns_fair_value = false`) never appear as prices.
- No new charting dependency; Altair only.
- Test command: `.venv/bin/python -m pytest -q`. Lint: `.venv/bin/ruff check cryptoval tests dashboard`.

## Review Focus

1. **The DB is missing**, as on a fresh clone: the app shows "run `make all`" instead of a
   traceback (Task 2, `test_app_without_a_database_says_how_to_build_one`).
2. **An asset with no applicable fair value at all**, such as a stablecoin: the football field is
   empty and the refusals list explains why (Task 1, `test_refusals_list_methods_that_never_apply`).
3. **A signal method with a non-null diagnostic must not leak into the price views**
   (Task 1, `test_signals_never_enter_the_football_field_or_heatmap`).
4. **DCF sliders set the discount rate at or below terminal growth** with Gordon terminal: the
   page shows the refusal message, not a negative price (Task 1, `test_dcf_value_refuses_an_inverted_gordon`).
5. **A stale panel**, more than 3 days old: the banner warns (Task 1, `test_freshness_flags_a_stale_panel`).

---

### Task 1: Query layer

**Files:**
- Create: `cryptoval/dashboard/__init__.py`, `cryptoval/dashboard/queries.py`
- Modify: `requirements.txt` (`streamlit==1.66.0`)
- Test: `tests/test_dashboard_queries.py`

**Interfaces:**
- Produces:
  - `freshness(con, today) -> dict` with keys `panel_through`, `coin_age_through`,
    `panel_vintage`, `panel_lag_days`, `stale`.
  - `latest_price(con, symbol) -> tuple[date, float] | None`.
  - `football_field(con, symbol) -> DataFrame[method_id, family, metric_date, low, high, mid, n_cells, is_circular]`.
  - `refusals(con, symbol) -> DataFrame[method_id, reason]`.
  - `indicator_series(con, name) -> DataFrame[metric_date, value]`, for BTC.
  - `INDICATORS` and `THRESHOLDS: dict[str, list[float]]`.
  - `agreement(con) -> DataFrame[symbol, method_id, metric_date, log_ratio]`.
  - `dcf_inputs(con) -> DataFrame[symbol, method_id, metric_date, annual_cash_flow_usd, supply, discount_rate, growth, stage_years, terminal_multiple]`.
  - `dcf_value(cash_flow, supply, growth, discount, years, terminal_multiple) -> tuple[float, float]`.

- [ ] **Step 1: Install Streamlit and pin it.** Append `streamlit==1.66.0` to `requirements.txt`
  (already installed in `.venv`).

- [ ] **Step 2: Write the failing tests** in `tests/test_dashboard_queries.py`. They build an
  in-memory DuckDB with:
  - `dim_method`: `realized_price` (price, circular), `mayer_multiple` (price), `slrv` (signal),
    `defi_net_revenue_dcf` (price);
  - `fct_asset_metrics_daily`: BTC and USDC, 400 days, with `price_usd`, `mvrv`, `issuance_usd`
    and `panel_vintage_ts`;
  - `fct_btc_cointime_daily`: liveliness;
  - `fct_valuation_estimate` rows covering a two-cell grid, a signal and a never-applicable method.

  The tests:
  - `test_freshness_flags_a_stale_panel`
  - `test_football_field_is_the_grid_range_on_each_methods_latest_applicable_date`
  - `test_signals_never_enter_the_football_field_or_heatmap`
  - `test_refusals_list_methods_that_never_apply`
  - `test_indicator_series_reads_panel_coin_age_and_signal_diagnostics`
  - `test_mayer_multiple_is_price_over_its_200_day_mean`
  - `test_agreement_is_log10_of_median_fair_value_over_that_days_price`
  - `test_dcf_inputs_come_from_the_latest_applicable_row`
  - `test_dcf_value_matches_three_stage_pv_per_token`
  - `test_dcf_value_refuses_an_inverted_gordon`

- [ ] **Step 3: Run them.** Expected: FAIL (`ModuleNotFoundError: cryptoval.dashboard`).

- [ ] **Step 4: Implement `cryptoval/dashboard/queries.py`** (code in the commit; the SQL is the
  spec §2 table made literal):
  - **`football_field`:** joins `dim_method` on `returns_fair_value`. It filters `applicable` and
    `fair_value > 0`, takes `max(metric_date)` per method, then min/max/median over that date's
    rows.
  - **`agreement`:** the same latest-date rule per (symbol, method). It joins the panel price on
    `metric_date::date` and returns `log10(median / price)`.
  - **`indicator_series`:**
    - `mvrv` comes from the panel;
    - `mayer_multiple` = price / 200-row rolling mean;
    - `puell_multiple` = issuance_usd / 365-row rolling mean;
    - `liveliness` comes from coin age;
    - signals come from `json_extract` of `diagnostics` under the method id.
  - **`dcf_value`:** returns `(pv / supply, terminal_share)` from `three_stage_pv` and lets its
    `ValueError` propagate.

- [ ] **Step 5: Run the tests.** Expected: PASS.

- [ ] **Step 6: Commit.** `feat: dashboard query layer over the marts`

---

### Task 2: The Streamlit app

**Files:**
- Create: `dashboard/app.py`
- Modify: `Makefile` (`dashboard` target), `.github/workflows/ci.yml` (lint covers `dashboard`)
- Test: `tests/test_dashboard_app.py`

**Interfaces:**
- Consumes: every Task 1 function.

- [ ] **Step 1: Write the failing tests:**
  - `test_app_renders_every_tab_against_a_fixture_database`: builds the Task 1 fixture into a
    file DB under `tmp_path`, sets `CRYPTOVAL_DATA_ROOT`, runs
    `AppTest.from_file("dashboard/app.py")`, and asserts `not at.exception`. It also asserts that
    the disclaimer markdown is present and that there are 4 tabs.
  - `test_app_without_a_database_says_how_to_build_one`: with an empty `tmp_path`, the app shows
    an error containing `make all`.
  - `test_app_holds_no_connection_open_between_reads`: after `at.run()`, a fresh **read-write**
    connection to the fixture file succeeds.

- [ ] **Step 2: Run them.** Expected: FAIL (file not found).

- [ ] **Step 3: Implement `dashboard/app.py`:**
  - The disclaimer, then the freshness banner (`st.warning` when stale).
  - Four `st.tabs`: Football field, Cycle indicators, Method agreement, DCF sliders.
  - Charts are Altair: bars with an x-axis on a log scale for the football field, lines with
    threshold rules for the indicators, a rect heatmap for agreement.
  - Reads go through `_read(fn, *args)`, which opens a read-only connection, calls `fn` and closes
    it. `st.cache_data(ttl=600)` sits on per-view loaders.
- [ ] **Step 4: Add the Makefile target**, `dashboard: $(PY) -m streamlit run dashboard/app.py`,
  plus `.PHONY`. In CI, change the lint step to `ruff check cryptoval tests dashboard`.
- [ ] **Step 5: Run the tests.** Expected: PASS. Then run the app for real against the live DB,
  headless for 10 s, and check that it serves HTTP 200.
- [ ] **Step 6: Commit.** `feat: Streamlit dashboard -- football field, indicators, agreement, DCF sliders`

---

### Task 3: README and Dockerfile

**Files:**
- Create: `README.md`, `Dockerfile`, `.dockerignore`
- Test: `tests/test_repo_hygiene.py` (append)

- [ ] **Step 1: Write the failing tests:**
  - `test_readme_carries_the_disclaimer_and_the_reproduce_command`: README contains the
    disclaimer sentence, `make all` and `make dashboard`.
  - `test_dockerfile_installs_cpu_torch_like_ci`: the Dockerfile's torch line equals CI's.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Write `README.md` and the `Dockerfile`** to spec §4. `.dockerignore` excludes
  `data`, `.venv`, `.worktrees`, `.git`, `dbt/target`, `dbt/dbt_packages` and `__pycache__`.
- [ ] **Step 4: Verify.** Run the tests (PASS), then `docker build -t cryptoval .`, and check
  that it succeeds and that `docker run --rm cryptoval .venv/bin/python -c "import cryptoval"`, or
  its equivalent, imports.
- [ ] **Step 5: Commit.** `docs: README with the reproduce path, and a Dockerfile`
