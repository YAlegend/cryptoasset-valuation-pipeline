# Cryptoasset Valuation — Remediation and Method Registry Implementation Plan (Plan 2 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the eight data-integrity defects that Plan 1's live panel revealed, then build the
method registry and reproduce the dissertation's ten valuation methods on the corrected panel, each
with a method card and a comparison against the figures in the PDF.

**Architecture:** Two stages in one plan. **2a (Tasks 1-8)** closes the gap between what the panel
claims and what it contains: an asset's identity becomes a map from metric to vendor id, facts key on
a conformed `symbol`, impossible vendor values go to a quarantine table instead of into velocity
figures, and bronze partitions carry a `run_id` so re-runs stop destroying their own audit trail.
**2b (Tasks 9-16)** adds the method registry — pure functions over the panel behind one `estimate()`
signature, whose applicability is *computed* from declared data dependencies and measured coverage
rather than hand-written per method — then the ten methods, their cards, and the dissertation
comparison.

**Tech Stack:** Python 3.14.5, pandas 3.0.6, DuckDB 1.5.6, dbt-core 1.12.5 with dbt-duckdb 1.11.0,
pyarrow 26.0.0, httpx 0.28.1, statsmodels (new in this plan, for the Metcalfe regression and its
confidence intervals), pytest 9.1.1, ruff 0.17.0.

**Spec:** `docs/superpowers/specs/2026-10-11-cryptoasset-valuation-pipeline-design.md` (revised
2026-10-12; read its revision log first — it names every defect this plan's stage 2a fixes and why
the spec itself was wrong).

## Global Constraints

Every task's requirements implicitly include this section.

- **Free tiers only, no exceptions.** No paid API calls, no API keys. BigQuery stays untouched in
  this plan (it is Plan 3).
- **Python 3.14.5**, dependencies pinned to exact versions in `requirements.txt`. One addition in
  this plan: `statsmodels==0.15.0`. Pin it exactly; do not add anything else.
- **All timestamps UTC.** Every datetime parsed with `utc=True`; naive datetimes are a bug.
- **Parquet is the system of record.** DuckDB files are derived and disposable.
- **Bronze is immutable** — and after Task 6 that is literally true rather than aspirational. Never
  delete or overwrite a bronze partition.
- **No network calls in unit tests.** Unit tests read recorded fixtures. Network tests are marked
  `@pytest.mark.integration` and excluded from the default run.
- **No `train_test_split`, no shuffling** of time-ordered data anywhere in the repository. Enforced
  by `tests/test_repo_hygiene.py`, whose AST detector already scans the whole repo. Every method in
  stage 2b is fitted on a trailing window or on the full series; none of them resamples.
- **Nulls stay null.** A missing vendor value is never coerced to `0.0`. The one sanctioned deviation
  is Task 3's quarantine, which nulls an *impossible* value and records the raw figure.
- **No method invents a number.** A method whose inputs are unavailable returns
  `applicable=False` with a reason. This is enforced by Task 9's registry, not by each method.
- **Every commit ends with** `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Review Focus

Failure modes the spec implies that no obvious task test would exercise. Each has its test assigned
to the task owning the code.

1. **A method that is circular rather than wrong.** QTM with velocity derived as
   `volume / market_cap` algebraically collapses to `price / circulating_fraction` — it returns the
   market price back, dressed as a fair value, and every residual is zero by construction. Expected:
   the registry detects that a method's velocity input derives from market cap and records
   `circular=True` in diagnostics, so the method card and the report cannot present it as
   independent evidence. → Task 10.
2. **A regression fitted on two trending series.** Metcalfe on `log(market_cap) ~ log(active
   addresses)` will report a high R² for almost any pair of growing series. Expected: β reported with
   a confidence interval, plus the Durbin-Watson statistic and the residual series, so spurious
   regression is visible rather than hidden behind R². → Task 14.
3. **An asset whose coverage is dense enough to fit but too sparse to trust.** BNB's market cap is
   19% dense. A regression on 646 scattered days will converge and return a number. Expected: a
   declared minimum coverage density per method, checked before fitting, producing
   `applicable=False` with the measured density in the reason. → Task 10.
4. **A quarantined cell silently changing a method's answer between runs.** Nulling an impossible
   volume changes velocity, and therefore every Family A and D estimate, for that asset-day.
   Expected: `fct_valuation_estimate` rows carry the `panel_vintage_ts` they were computed from, so
   an estimate is always traceable to the panel state that produced it. → Task 11.
5. **A protocol rollup that double-counts.** Summing DefiLlama slugs for an asset will double-count
   if a parent and its child both appear (e.g. a "Uniswap" aggregate alongside "Uniswap V3").
   Expected: the rollup refuses to sum a slug set containing both an aggregate and its components,
   naming the collision. → Task 5.

---

## File Structure

**Stage 2a — remediation**

| File | Responsibility |
|---|---|
| `cryptoval/config.py` *(modify)* | `Asset` gains `metric_sources`, `defillama_slugs`, `coverage_floor_note`. Validation rejects an unknown metric name or an empty slug |
| `config/assets.yml` *(modify)* | The §5.1 vendor-id map; `trx`/`avax` added as metric sources; DefiLlama slug lists for the seven DeFi assets plus ETH and LINK |
| `cryptoval/sources/coinmetrics.py` *(modify)* | Fetch the union of every vendor id any asset names, not just `cm_id`; drop the `-status` sidecar columns |
| `cryptoval/quarantine.py` *(create)* | `quarantine_impossible()` — splits a frame into (clean, quarantined) against a contract's `maximum`/`minimum` bounds |
| `cryptoval/paths.py` *(modify)* | `bronze_partition()` gains `run_id`; `latest_bronze_partition()` added |
| `cryptoval/io/bronze.py` *(modify)* | `write_bronze` takes `run_id` and never deletes; `read_bronze` reads the newest run unless told otherwise |
| `config/contracts/defillama_fees.yml` *(modify)* | Gross-fee floors kept; `net_revenue_24h` floor removed |
| `cryptoval/rollup.py` *(create)* | `rollup_protocols()` — sums a slug list to one symbol, refusing aggregate/component collisions |
| `cryptoval/cli.py` *(modify)* | Wire quarantine into ingest; scope blocking validation to mapped protocols; add `estimate` in Task 16 |
| `dbt/seeds/assets.csv` *(create)* | `dim_asset`'s source of truth, generated from `assets.yml` |
| `dbt/models/staging/stg_asset_metrics.sql` *(modify)* | Resolve vendor id → `symbol` |
| `dbt/models/staging/stg_market_chart.sql` *(create)* | CoinGecko panel, resolved to `symbol` |
| `dbt/models/staging/stg_protocol_revenue.sql` *(create)* | DefiLlama, rolled up to `symbol` |
| `dbt/models/staging/stg_equity_prices.sql` *(create)* | Equities and macro series |
| `dbt/models/marts/dim_asset.sql` *(modify)* | Seeded, not derived |
| `dbt/models/marts/fct_asset_metrics_daily.sql` *(modify)* | Keyed on `symbol`; both volume definitions; CoinGecko market cap filling the TRX gap |
| `dbt/models/marts/fct_equity_prices_daily.sql` *(create)* | The separate equity fact |

**Stage 2b — method registry**

| File | Responsibility |
|---|---|
| `cryptoval/methods/base.py` *(create)* | `Estimate`, `MethodSpec`, the `Method` protocol, and `resolve_applicability()` |
| `cryptoval/methods/registry.py` *(create)* | Discovery, parameter-set hashing, and the run loop producing `fct_valuation_estimate` rows |
| `cryptoval/methods/cost.py` *(create)* | Family E(a): realized price, Thermocap floor, MVTV, Investor Cap, Hayes |
| `cryptoval/methods/monetary.py` *(create)* | Family A: QTM, J-curve, holding-time correction |
| `cryptoval/methods/network.py` *(create)* | Family B: Metcalfe, estimated |
| `cryptoval/methods/multiples.py` *(create)* | Family D: NVT, NVT golden ratio, Price-to-Utility |
| `config/methods/*.yml` *(create)* | One config per method: family, params, data dependencies, coverage floor, citation |
| `dbt/models/marts/fct_valuation_estimate.sql` *(create)* | The estimate fact |
| `dbt/models/marts/dim_method.sql` *(create)* | Seeded from `config/methods/` |
| `docs/methods/*.md` *(create)* | One method card per method, generated from config plus hand-written failure modes |
| `reports/dissertation-comparison.md` *(create)* | Reproduction verdict against the PDF's figures |

---
# Stage 2a — Remediation

## Task 1: An asset's identity becomes a map from metric to vendor id

Spec §5.1. Today `Asset` carries one `cm_id`, which forced a single choice between the right metrics
and the right entity — and for TRX the wrong entity won. This task makes the mapping explicit. It
changes no fetching behaviour yet; Task 2 does that.

**Files:**
- Modify: `cryptoval/config.py`
- Modify: `config/assets.yml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (this is the first).
- Produces:
  - `cryptoval.config.METRIC_GROUPS: tuple[str, ...]` = `("network", "valuation", "volume")`
  - `Asset.metric_sources: dict[str, str]` — group → source, where a source is `"cm:<id>"`,
    `"coingecko"`, or `"none"`
  - `Asset.source_for(group: str) -> str` — the resolved source for a group, defaulting to
    `"cm:<cm_id>"` when the group is unmapped, or `"coingecko"` when the asset has no `cm_id`
  - `Asset.cm_ids: tuple[str, ...]` — every Coin Metrics id this asset draws on, sorted
  - `Asset.defillama_slugs: tuple[str, ...]`
  - `Asset.coverage_note: str`

**Note on hashability:** `Asset` stays `frozen=True` but gains a `dict` field, so it is no longer
hashable. Nothing in the repository hashes an `Asset` (`assert_unique_symbols` builds a set of
symbol *strings*); Step 7 verifies that.

- [ ] **Step 1: Write the failing test for the metric-source map**

Add to `tests/test_config.py`:

```python
def test_trx_draws_network_metrics_from_the_real_tron_series():
    _, assets = load_config()
    trx = next(a for a in assets if a.symbol == "TRX")
    # trx_eth is a near-dead ERC-20 wrapper: market cap NULL, volume 0, $165K credible max
    # against real TRON's $254M/day (spec 5.1).
    assert trx.source_for("network") == "cm:trx"
    assert trx.source_for("volume") == "cm:trx"
    # Coin Metrics serves no free market cap for `trx`, so it comes from CoinGecko.
    assert trx.source_for("valuation") == "coingecko"
    assert trx.cm_ids == ("trx",)


def test_avax_takes_volume_from_a_different_coin_metrics_id():
    _, assets = load_config()
    avax = next(a for a in assets if a.symbol == "AVAX")
    # avaxc is the genuine C-Chain, but serves no reported spot volume (0% of the panel).
    assert avax.source_for("network") == "cm:avaxc"
    assert avax.source_for("valuation") == "cm:avaxc"
    assert avax.source_for("volume") == "cm:avax"
    assert avax.cm_ids == ("avax", "avaxc")


def test_an_unmapped_asset_defaults_every_group_to_its_primary_id():
    _, assets = load_config()
    btc = next(a for a in assets if a.symbol == "BTC")
    assert [btc.source_for(g) for g in METRIC_GROUPS] == ["cm:btc"] * 3
    assert btc.cm_ids == ("btc",)


def test_an_asset_with_no_coin_metrics_id_falls_back_to_coingecko():
    _, assets = load_config()
    sol = next(a for a in assets if a.symbol == "SOL")
    assert sol.source_for("network") == "none"
    assert sol.source_for("valuation") == "coingecko"
    assert sol.cm_ids == ()


def test_an_unknown_metric_group_is_rejected():
    with pytest.raises(ValueError, match="unknown metric group"):
        _validate(Asset(symbol="X", sector="defi", cm_id="x", metric_sources={"nope": "cm:x"}))


def test_a_malformed_metric_source_is_rejected():
    with pytest.raises(ValueError, match="unusable metric source"):
        _validate(Asset(symbol="X", sector="defi", cm_id="x", metric_sources={"volume": "CM:X"}))


def test_a_valuation_group_pointing_at_coingecko_requires_a_gecko_id():
    with pytest.raises(ValueError, match="names coingecko but has no gecko_id"):
        _validate(
            Asset(symbol="X", sector="defi", cm_id="x", metric_sources={"valuation": "coingecko"})
        )
```

Add `METRIC_GROUPS` and `_validate` to the existing import line at the top of
`tests/test_config.py`, which already imports from `cryptoval.config`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -k "metric or tron or avax or coingecko" -v`
Expected: FAIL with `ImportError: cannot import name 'METRIC_GROUPS'`

- [ ] **Step 3: Implement the map in `cryptoval/config.py`**

Add `import re` and `field` to the existing imports, then add after the `CONSENSUS` tuple:

```python
METRIC_GROUPS = ("network", "valuation", "volume")

# A metric source is one Coin Metrics id, CoinGecko, or an explicit refusal. The lowercase
# pattern is deliberate: Coin Metrics ids are lowercase, and silently accepting "CM:X" would
# produce a 404 on the first live fetch instead of a config error here.
_SOURCE_PATTERN = re.compile(r"^(cm:[a-z0-9_]+|coingecko|none)$")
```

Replace the `Asset` dataclass with:

```python
@dataclass(frozen=True)
class Asset:
    symbol: str
    sector: str | None = None
    consensus: str = "na"
    cm_id: str | None = None
    gecko_id: str | None = None
    yahoo_id: str | None = None
    is_stablecoin: bool = False
    is_equity: bool = False
    # Group -> source. Unmapped groups fall back to the primary cm_id (spec 5.1), so only
    # the three assets whose vendor identity is split need an entry here.
    metric_sources: dict[str, str] = field(default_factory=dict)
    defillama_slugs: tuple[str, ...] = ()
    coverage_note: str = ""

    def source_for(self, group: str) -> str:
        if group not in METRIC_GROUPS:
            raise ValueError(f"unknown metric group {group!r}; expected one of {METRIC_GROUPS}")
        explicit = self.metric_sources.get(group)
        if explicit:
            return explicit
        if self.cm_id:
            return f"cm:{self.cm_id}"
        return "none" if group == "network" else "coingecko"

    @property
    def cm_ids(self) -> tuple[str, ...]:
        ids = {s[len("cm:") :] for s in map(self.source_for, METRIC_GROUPS) if s.startswith("cm:")}
        return tuple(sorted(ids))
```

Extend `_validate` — insert before its final `if not asset.cm_id and not asset.gecko_id:` check:

```python
    for group, source in asset.metric_sources.items():
        if group not in METRIC_GROUPS:
            raise ValueError(
                f"{asset.symbol}: unknown metric group {group!r}; expected one of {METRIC_GROUPS}"
            )
        if not _SOURCE_PATTERN.match(str(source)):
            raise ValueError(
                f"{asset.symbol}: unusable metric source {source!r} for group {group!r}; "
                "expected 'cm:<id>', 'coingecko' or 'none'"
            )
        if source == "coingecko" and not asset.gecko_id:
            raise ValueError(f"{asset.symbol}: group {group!r} names coingecko but has no gecko_id")
    if any(not str(slug).strip() for slug in asset.defillama_slugs):
        raise ValueError(f"{asset.symbol}: defillama_slugs contains an empty entry")
```

In `load_config`, normalise the YAML's list into a tuple so the dataclass field type holds. Replace
the two asset-building loops with:

```python
    def _build(row: dict, *, is_equity: bool) -> Asset:
        row = dict(row)
        row["defillama_slugs"] = tuple(row.get("defillama_slugs") or ())
        row["metric_sources"] = dict(row.get("metric_sources") or {})
        return Asset(is_equity=is_equity, **row)

    for row in raw_assets.get("assets", []) or []:
        assets.append(_build(row, is_equity=False))
    for row in raw_assets.get("equities", []) or []:
        assets.append(_build(row, is_equity=True))
```

- [ ] **Step 4: Update `config/assets.yml`**

Replace the TRX, AVAX, POL, BNB and SOL rows with the block below, and add `defillama_slugs` to the
DeFi assets plus ETH and LINK. Leave every other row untouched. The slug lists come from the live
feed measured 2026-10-12 — `Maker` no longer exists, and four protocols span versions (spec §5.1).

```yaml
  - {symbol: ETH,  cm_id: eth,     gecko_id: ethereum,      sector: smart_contract_l1, consensus: pos,
     defillama_slugs: [Ethereum]}
  - {symbol: AVAX, cm_id: avaxc,   gecko_id: avalanche-2,   sector: smart_contract_l1, consensus: pos,
     metric_sources: {volume: "cm:avax"},
     coverage_note: "avaxc is the C-Chain and serves no reported spot volume; volume comes from cm:avax (spec 5.1)"}
  - {symbol: TRX,  cm_id: trx,     gecko_id: tron,          sector: smart_contract_l1, consensus: pos,
     metric_sources: {valuation: coingecko},
     coverage_note: "cm:trx is real TRON but has no free market cap, MVRV or supply, so the cost family does not apply; trx_eth was a near-dead ERC-20 wrapper (spec 5.1)"}
  - {symbol: POL,  cm_id: pol_eth, gecko_id: polygon-ecosystem-token, sector: scaling, consensus: pos,
     coverage_note: "history starts 2023-10-25; the pre-rebrand MATIC series returns no data on the free tier, so no splice is possible (spec 5)"}
  - {symbol: UNI,  cm_id: uni,     gecko_id: uniswap,       sector: defi, consensus: other,
     defillama_slugs: ["Uniswap V4", "Uniswap V3", "Uniswap V2"]}
  - {symbol: AAVE, cm_id: aave,    gecko_id: aave,          sector: defi, consensus: other,
     defillama_slugs: ["Aave V4", "Aave V3", "Aave V2"]}
  - {symbol: CRV,  cm_id: crv,     gecko_id: curve-dao-token, sector: defi, consensus: other,
     defillama_slugs: ["Curve DEX", "Curve LlamaLend V2"]}
  - {symbol: MKR,  cm_id: mkr,     gecko_id: maker,         sector: defi, consensus: other,
     defillama_slugs: ["Sky Lending", "Sky Money"],
     coverage_note: "rebranded: no protocol named Maker exists in the feed as of 2026-10-12"}
  - {symbol: COMP, cm_id: comp,    gecko_id: compound-governance-token, sector: defi, consensus: other,
     defillama_slugs: ["Compound V3", "Compound V2"]}
  - {symbol: SNX,  cm_id: snx,     gecko_id: havven,        sector: defi, consensus: other,
     defillama_slugs: ["Synthetix V4", "Synthetix V3", "Synthetix v1+v2"]}
  - {symbol: LDO,  cm_id: ldo,     gecko_id: lido-dao,      sector: defi, consensus: other,
     defillama_slugs: [Lido]}
  - {symbol: LINK, cm_id: link,    gecko_id: chainlink,     sector: oracle, consensus: other,
     defillama_slugs: ["Chainlink Requests", "Chainlink Staking"]}
  - {symbol: BNB,  cm_id: bnb,     gecko_id: binancecoin,   sector: exchange, consensus: other,
     coverage_note: "market cap, MVRV and supply begin 2019-04-22 and are only 19% dense; density is a method precondition (spec 5)"}
  - {symbol: SOL,  cm_id: null,    gecko_id: solana,        sector: smart_contract_l1, consensus: pos,
     coverage_note: "absent from the Coin Metrics free tier (CapMrktCurUSD returns 403); CoinGecko price and market cap only (spec 5)"}
```

Add the two macro tickers to the `equities:` block (spec §15 Q1, resolved):

```yaml
  - {symbol: GOLD, yahoo_id: "GC=F"}
  - {symbol: DXY,  yahoo_id: "DX-Y.NYB"}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS, all tests including the pre-existing nine.

- [ ] **Step 6: Fix the asset-count test**

`tests/test_config.py` asserts 24 crypto and 5 equities. The equity list is now 7. Update that
assertion to `7` and add a comment naming gold and DXY as the §15 Q1 resolution. Re-run the file.

- [ ] **Step 7: Verify nothing hashes an Asset**

Run: `grep -rn "set(assets\|{a for a in assets\|hash(asset\|Asset)" cryptoval tests --include=*.py`
Expected: no hit that puts an `Asset` into a set or dict key. If one exists, change it to key on
`asset.symbol` before continuing.

- [ ] **Step 8: Run the full suite and lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add cryptoval/config.py config/assets.yml tests/test_config.py
git commit -m "$(cat <<'MSG'
feat: an asset's identity becomes a map from metric to vendor id

Coin Metrics serves the right metrics for the wrong entity on three
assets. One cm_id per asset could not express that, so Plan 1 put
trx_eth -- a near-dead ERC-20 wrapper with a null market cap and $165K
of credible daily volume -- in the panel as TRX.

metric_sources maps network/valuation/volume to a vendor id, defaulting
to the primary cm_id so only the split assets need an entry. Also adds
version-summed, rebrand-aware defillama_slugs (Maker is Sky Lending plus
Sky Money now) and the gold and dollar-index tickers resolving spec 15 Q1.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
## Task 2: Coin Metrics fetches every vendor id, and the dead sidecar columns stop entering the panel

Two independent corrections to one adapter. The fetch must now draw on the union of ids the config
names (`avax` *and* `avaxc`), and the `-status` sidecar columns Coin Metrics returns alongside
`FlowInExUSD` must stop landing in the panel — measured on the live panel, all three are 100% empty
and they are string columns in an otherwise numeric frame.

**Files:**
- Modify: `cryptoval/sources/coinmetrics.py`
- Test: `tests/test_coinmetrics.py`

**Interfaces:**
- Consumes: `Asset.cm_ids` from Task 1.
- Produces: `CoinMetricsSource.fetch` unchanged in signature; its frame's `asset` column now holds
  any of the configured vendor ids, which Task 7's staging layer resolves to `symbol`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_coinmetrics.py`:

```python
def test_fetch_requests_every_configured_vendor_id():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"data": [
            {"asset": "avaxc", "time": "2026-10-01T00:00:00.000000000Z", "PriceUSD": "30.0"},
        ]})

    assets = [
        Asset(symbol="AVAX", sector="smart_contract_l1", consensus="pos", cm_id="avaxc",
              gecko_id="avalanche-2", metric_sources={"volume": "cm:avax"}),
        Asset(symbol="BTC", sector="pow_monetary", consensus="pow", cm_id="btc"),
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        CoinMetricsSource().fetch(dt.date(2026, 10, 1), dt.date(2026, 10, 2), assets, client)

    # Sorted union, so avax rides along with avaxc rather than replacing it.
    assert captured["assets"] == "avax,avaxc,btc"


def test_status_sidecar_columns_are_dropped():
    payload = {"data": [{
        "asset": "btc",
        "time": "2026-10-01T00:00:00.000000000Z",
        "FlowInExUSD": "1000.0",
        "FlowInExUSD-status": "unavailable",
        "FlowInExUSD-status-time": "2026-10-01T00:00:00.000000000Z",
        "FlowOutExUSD-status": "unavailable",
    }]}
    df = parse_timeseries(payload)
    assert [c for c in df.columns if "status" in c] == []
    # The metric itself survives; only the sidecars go.
    assert df["FlowInExUSD"].iloc[0] == 1000.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_coinmetrics.py -k "vendor_id or sidecar" -v`
Expected: FAIL — the first with `assert 'avaxc,btc' == 'avax,avaxc,btc'`, the second with
`assert ['FlowInExUSD-status', ...] == []`

- [ ] **Step 3: Implement the union of ids**

In `cryptoval/sources/coinmetrics.py`, replace the `ids` line in `fetch`:

```python
        # The union, not one id per asset: three assets draw different metric groups from
        # different Coin Metrics series (spec 5.1), so both ids must be requested.
        ids = sorted({cm_id for a in assets for cm_id in a.cm_ids})
```

- [ ] **Step 4: Implement the sidecar drop**

In `parse_timeseries`, after `df = df.drop(columns=["time"])`:

```python
    # Coin Metrics returns "<metric>-status" and "<metric>-status-time" beside some metrics.
    # On the live panel all three are 100% empty, and they are strings in a numeric frame.
    sidecars = [c for c in df.columns if c.endswith(("-status", "-status-time"))]
    if sidecars:
        df = df.drop(columns=sidecars)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_coinmetrics.py -v`
Expected: PASS

- [ ] **Step 6: Run the full suite and lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/sources/coinmetrics.py tests/test_coinmetrics.py
git commit -m "$(cat <<'MSG'
fix: fetch the union of vendor ids and drop the dead status sidecars

AVAX's volume lives under cm:avax while its network metrics live under
cm:avaxc, so requesting one id per asset left velocity uncomputable for
it. The fetch now requests the union the config names.

Separately, the -status and -status-time columns Coin Metrics returns
beside FlowInExUSD were landing in the panel as string columns that are
100% empty on live data.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 3: Quarantine impossible values instead of carrying or dropping them

Spec §9.1. The live panel holds 14 reported-spot-volume values between 1e36 and 1e55 USD against a
credible maximum of 2.3e10 — 12 on `usdc`, 2 on `trx_eth`. There is no economic reading of 7e55, but
the *rest* of those rows is healthy (USDC's market cap on those days is a clean 7.2e10), so dropping
the row discards good data while keeping the value poisons every velocity figure downstream. The
offending **cell** is nulled and the raw value recorded.

**Files:**
- Create: `cryptoval/quarantine.py`
- Create: `tests/test_quarantine.py`
- Modify: `cryptoval/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Contract.ranges` from `cryptoval.contracts.loader` (existing: a
  `dict[str, dict[str, float]]` of `{column: {"minimum": x, "maximum": y}}`).
- Produces:
  - `cryptoval.quarantine.QUARANTINE_COLUMNS: list[str]`
  - `quarantine_impossible(df, contract, source, run_id, key_cols) -> tuple[pd.DataFrame, pd.DataFrame]`
    returning `(clean, quarantined)`. `clean` is `df` with offending cells set to `NaN`;
    `quarantined` has `QUARANTINE_COLUMNS` and one row per offending cell.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quarantine.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.contracts.loader import Contract
from cryptoval.quarantine import QUARANTINE_COLUMNS, quarantine_impossible

CONTRACT = Contract(
    name="t",
    version="1.0.0",
    required_columns=["asset", "metric_date"],
    columns={
        "volume_usd": {"type": "float", "minimum": 0, "maximum": 1.0e12},
        "market_cap_usd": {"type": "float", "minimum": 0},
    },
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset": ["usdc", "usdc", "btc"],
            "metric_date": pd.to_datetime(
                ["2026-05-05", "2026-05-06", "2026-05-05"], utc=True
            ),
            "volume_usd": [8.02e52, 3.4e9, 2.9e9],
            "market_cap_usd": [7.52e10, 7.5e10, 1.6e12],
        }
    )


def test_an_impossible_value_is_nulled_and_the_rest_of_the_row_survives():
    clean, bad = quarantine_impossible(
        _frame(), CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert np.isnan(clean.loc[0, "volume_usd"])
    # The healthy market cap beside it is untouched: this is why the cell, not the row, goes.
    assert clean.loc[0, "market_cap_usd"] == 7.52e10
    assert len(clean) == 3


def test_the_raw_value_and_the_bound_it_broke_are_recorded():
    _, bad = quarantine_impossible(
        _frame(), CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert list(bad.columns) == QUARANTINE_COLUMNS
    assert len(bad) == 1
    row = bad.iloc[0]
    assert row["source"] == "coinmetrics"
    assert row["column"] == "volume_usd"
    assert row["run_id"] == "run-1"
    assert row["raw_value"] == 8.02e52
    assert "maximum 1e+12" in row["reason"]
    assert row["entity"] == "usdc"


def test_values_inside_the_bounds_are_untouched():
    frame = _frame()
    frame.loc[0, "volume_usd"] = 5.0e9
    clean, bad = quarantine_impossible(
        frame, CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert bad.empty
    assert list(bad.columns) == QUARANTINE_COLUMNS
    pd.testing.assert_frame_equal(clean, frame)


def test_a_value_below_a_minimum_is_quarantined_too():
    frame = _frame()
    frame.loc[2, "market_cap_usd"] = -1.0
    _, bad = quarantine_impossible(
        frame, CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert set(bad["column"]) == {"volume_usd", "market_cap_usd"}
    assert "minimum 0" in bad.set_index("column").loc["market_cap_usd", "reason"]


def test_a_null_is_not_impossible():
    frame = _frame()
    frame.loc[0, "volume_usd"] = np.nan
    _, bad = quarantine_impossible(
        frame, CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    # A missing value is a gap, which the null-rate ceiling already governs. Quarantine is
    # only for values the vendor supplied that cannot be true.
    assert bad.empty


def test_a_contract_column_absent_from_the_frame_is_ignored():
    frame = _frame().drop(columns=["volume_usd"])
    clean, bad = quarantine_impossible(
        frame, CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert bad.empty
    assert "volume_usd" not in clean.columns


def test_nulling_keeps_the_column_numeric():
    clean, _ = quarantine_impossible(
        _frame(), CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    # An object-dtype column would break stable_hash and therefore SCD2 change detection.
    assert clean["volume_usd"].dtype.kind == "f"


def test_an_empty_frame_is_handled():
    empty = _frame().iloc[0:0]
    clean, bad = quarantine_impossible(
        empty, CONTRACT, "coinmetrics", "run-1", ["asset", "metric_date"]
    )
    assert clean.empty and bad.empty
    assert list(bad.columns) == QUARANTINE_COLUMNS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_quarantine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptoval.quarantine'`

- [ ] **Step 3: Implement `cryptoval/quarantine.py`**

```python
"""Quarantine for vendor values that are impossible rather than merely extreme.

Spec 9.1. The free tier served reported-spot-volume figures up to 7e55 USD on
2026-10-11 (14 rows) against a largest credible value of 7.5e10 in the same panel.
Carrying such a value poisons every velocity and NVT figure computed from it;
dropping the whole row discards the healthy market cap sitting beside it. So the
offending CELL is nulled and the raw value recorded here, with bronze untouched —
this is the one place silver deviates from the vendor's numbers, and it is logged.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cryptoval.contracts.loader import Contract

QUARANTINE_COLUMNS = [
    "source",
    "entity",
    "metric_date",
    "column",
    "raw_value",
    "reason",
    "run_id",
    "quarantined_at",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=QUARANTINE_COLUMNS)


def quarantine_impossible(
    df: pd.DataFrame,
    contract: Contract,
    source: str,
    run_id: str,
    key_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (clean, quarantined) against the contract's range bounds.

    ``clean`` is ``df`` with every out-of-bounds cell set to NaN. ``quarantined``
    carries one row per offending cell, including the raw value, so the panel can
    always be reconciled against what the vendor actually sent.
    """
    if df.empty:
        return df.copy(), _empty()

    clean = df.copy()
    records: list[dict] = []
    now = pd.Timestamp.now(tz="UTC")
    entity_col = next((c for c in key_cols if c in df.columns), None)

    for column, bounds in contract.ranges.items():
        if column not in clean.columns:
            continue
        numeric = pd.to_numeric(clean[column], errors="coerce")
        # A null is a gap, governed by the null-rate ceiling. Only a value the vendor
        # supplied that cannot be true belongs here.
        offending = pd.Series(False, index=clean.index)
        reasons: dict = {}
        if "minimum" in bounds:
            below = numeric.notna() & (numeric < bounds["minimum"])
            offending |= below
            for idx in clean.index[below]:
                reasons[idx] = f"below contract minimum {bounds['minimum']:g}"
        if "maximum" in bounds:
            above = numeric.notna() & (numeric > bounds["maximum"])
            offending |= above
            for idx in clean.index[above]:
                reasons[idx] = f"above contract maximum {bounds['maximum']:g}"
        if not offending.any():
            continue
        for idx in clean.index[offending]:
            records.append(
                {
                    "source": source,
                    "entity": clean.loc[idx, entity_col] if entity_col else None,
                    "metric_date": clean.loc[idx, "metric_date"]
                    if "metric_date" in clean.columns
                    else pd.NaT,
                    "column": column,
                    "raw_value": float(numeric.loc[idx]),
                    "reason": reasons[idx],
                    "run_id": run_id,
                    "quarantined_at": now,
                }
            )
        # float, not pd.NA: stable_hash cannot encode NAType, so an object-dtype column
        # here would silently break SCD2 change detection (Plan 1, Ruling A).
        clean[column] = numeric.astype("float64").mask(offending, np.nan)

    quarantined = pd.DataFrame(records, columns=QUARANTINE_COLUMNS) if records else _empty()
    return clean, quarantined
```

Note the reason strings use `:g`, which renders `1.0e+12` as `1e+12` — matching the test's
`"maximum 1e+12"` assertion.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_quarantine.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Write the failing CLI wiring test**

Add to `tests/test_cli.py`:

```python
def test_ingest_quarantines_an_impossible_value_before_it_reaches_silver(tmp_path, monkeypatch):
    """The stored vintage must never contain a value the contract calls impossible."""
    settings = Settings(data_root=str(tmp_path))

    class ImpossibleVolumeSource:
        name = "coinmetrics"
        contract_path = "config/contracts/coinmetrics_asset_metrics.yml"

        def fetch(self, start, end, assets, client):
            return FetchResult(
                pd.DataFrame(
                    {
                        "asset": ["usdc"],
                        "metric_date": pd.to_datetime(["2026-05-05"], utc=True),
                        "volume_reported_spot_usd_1d": [8.02e52],
                        "CapMrktCurUSD": [7.52e10],
                    }
                ),
                1,
            )

    monkeypatch.setitem(cli.SOURCES, "coinmetrics", ImpossibleVolumeSource())
    cli.run_ingest(settings, [], ["coinmetrics"], dt.date(2026, 5, 5), dt.date(2026, 5, 5))

    silver = pd.read_parquet(
        pathlib.Path(settings.resolver.table("silver", "asset_metrics")) / "part-0.parquet"
    )
    assert silver["volume_reported_spot_usd_1d"].isna().all()
    # The healthy market cap in the same row survived.
    assert silver["CapMrktCurUSD"].iloc[0] == 7.52e10

    quarantine = pd.read_parquet(
        pathlib.Path(settings.resolver.table("silver", "quarantine")) / "part-0.parquet"
    )
    assert quarantine["raw_value"].iloc[0] == 8.02e52
    assert quarantine["column"].iloc[0] == "volume_reported_spot_usd_1d"
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py -k quarantines -v`
Expected: FAIL — silver holds 8.02e52 and no quarantine table exists.

- [ ] **Step 7: Wire quarantine into `run_ingest`**

In `cryptoval/cli.py`, add the imports:

```python
from cryptoval.quarantine import QUARANTINE_COLUMNS, quarantine_impossible
```

In `run_ingest`, the quarantine must run *after* bronze (which records what the vendor sent) and
*before* `merge_scd2` (so the stored vintage is clean). Replace the body between `write_bronze(...)`
and `table, key = SILVER_TABLE[name], KEYS[name]` with:

```python
                write_bronze(settings.resolver, name, end, result.frame, params)

                table, key = SILVER_TABLE[name], KEYS[name]
                frame, quarantined = quarantine_impossible(
                    result.frame,
                    load_contract(source.contract_path),
                    name,
                    record.run_id,
                    key,
                )
                if not quarantined.empty:
                    _append_quarantine(settings, quarantined)
```

and change the `merge_scd2` call's second argument and the two `result.frame` references after it to
`frame`:

```python
                merged = merge_scd2(
                    _read_silver(settings, table),
                    frame,
                    key,
                    _value_cols(frame, key),
                    as_of,
                )
```

Leave `result.rows_in` and `len(result.frame)` in the manifest call as they are: they describe what
the vendor returned, which is what the manifest is for.

Add the append helper next to `_read_silver`:

```python
def _append_quarantine(settings: Settings, rows: pd.DataFrame) -> None:
    """Quarantine is an append-only log, not an SCD2 table: every entry is a fact about
    one run, and row growth is itself the monitored signal (spec 9.1)."""
    existing = _read_silver(settings, "quarantine")
    combined = (
        pd.concat([existing, rows], ignore_index=True)
        if not existing.empty
        else rows.reindex(columns=QUARANTINE_COLUMNS)
    )
    write_silver(settings.resolver, "quarantine", combined)
```

- [ ] **Step 8: Check whether `RunManifest` exposes `run_id`**

Run: `grep -n "run_id" cryptoval/manifest.py`
If the record returned by `manifest.start()` has no `run_id` attribute, use the attribute it does
expose for the run's identity (read the dataclass and use its id field); do not invent a second id.

- [ ] **Step 9: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py tests/test_quarantine.py -v`
Expected: PASS

- [ ] **Step 10: Run the full suite and lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`
Expected: all green.

- [ ] **Step 11: Commit**

```bash
git add cryptoval/quarantine.py cryptoval/cli.py tests/test_quarantine.py tests/test_cli.py
git commit -m "$(cat <<'MSG'
feat: quarantine impossible vendor values instead of storing them

The free tier served 14 reported-spot-volume figures between 1e36 and
1e55 USD against a credible maximum of 2.3e10. Carrying one poisons every
velocity and NVT figure computed from it; dropping the row discards the
healthy market cap beside it. So the cell is nulled, the raw value is
logged to silver/quarantine with the bound it broke, and bronze keeps the
vendor's original response.

Quarantine is append-only: row growth is the signal that a vendor broke
something new.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
## Task 4: Net revenue may be negative; gross fees may not

Spec §9.2, corrected. The original contract floored all three revenue columns at zero, which blocked
the build on 24 protocols — and for most of them the negative figure is *real*. Measured 2026-10-12:
Ensuro is insurance (claims above premiums), Hegic options (writers lost), WINR and Azuro betting
(the house lost), and Fragment reports $1.05M gross fees against −$852K net revenue, a rebate
structure working as designed. Gross fees cannot be negative, so those 10 rows are vendor errors and
Task 3's quarantine now catches them.

**Files:**
- Modify: `config/contracts/defillama_fees.yml`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Consumes: `check_ranges` and `Contract.ranges` (existing, unchanged).
- Produces: no new code. A contract-data change with a test that pins the asymmetry.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_contracts.py`:

```python
def test_negative_net_revenue_is_legitimate_but_negative_gross_fees_are_not():
    """Spec 9.2: a protocol can pay out more than it takes in; it cannot collect negative fees.

    Measured on the live feed 2026-10-12: Ensuro (insurance, claims above premiums),
    Hegic (options, writers lost) and Fragment ($1.05M fees, -$852K net revenue from
    rebates) all report negative net revenue as a true description of their day.
    """
    contract = load_contract("config/contracts/defillama_fees.yml")

    legitimate = pd.DataFrame(
        {
            "protocol": ["Ensuro", "Fragment"],
            "metric_date": pd.to_datetime(["2026-10-11", "2026-10-11"], utc=True),
            "total_fees_24h": [0.0, 1053729.0],
            "total_fees_30d": [113662.0, 32315485.0],
            "net_revenue_24h": [-1281.0, -852147.0],
        }
    )
    assert check_ranges(legitimate, contract).passed

    vendor_error = legitimate.copy()
    vendor_error.loc[0, "total_fees_24h"] = -23915.0
    result = check_ranges(vendor_error, contract)
    assert not result.passed
    assert "total_fees_24h" in result.detail


def test_net_revenue_has_no_lower_bound_in_the_contract():
    contract = load_contract("config/contracts/defillama_fees.yml")
    assert "minimum" not in contract.ranges.get("net_revenue_24h", {})
    # Gross columns keep theirs.
    assert contract.ranges["total_fees_24h"]["minimum"] == 0
    assert contract.ranges["total_fees_30d"]["minimum"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_contracts.py -k "negative or lower_bound" -v`
Expected: FAIL — `check_ranges(legitimate, ...)` currently fails on `net_revenue_24h < 0`.

- [ ] **Step 3: Correct the contract**

In `config/contracts/defillama_fees.yml`, bump `version` to `"1.1.0"` and replace the
`net_revenue_24h` column line and the comment above `required`:

```yaml
schema:
  # total_fees_24h is NOT required: 383 of 2,991 listed protocols report no 24h fees
  # (fix round 1, Ruling H). A negative GROSS fee still fails the range bound below and
  # is quarantined (spec 9.1): fees collected cannot be negative, so those rows are
  # vendor errors. Negative NET revenue is a different thing entirely -- see below.
  required:
    - protocol
  columns:
    protocol: {type: string}
    total_fees_24h: {type: float, minimum: 0, max_null_fraction: 0.35}   # live 12.8%
    total_fees_30d: {type: float, minimum: 0, max_null_fraction: 0.35}   # live 11.0%
    # NO minimum, deliberately (spec 9.2, corrected 2026-10-12). Net revenue is what the
    # protocol retains after payouts and incentives, and it goes negative for real
    # economic reasons: insurance paying claims above premiums (Ensuro), options writers
    # losing (Hegic), a betting protocol's house losing (WINR, Azuro, DoubleUp), and
    # rebate structures (Fragment: $1.05M fees, -$852K net revenue). The original
    # minimum: 0 was blocking the build on 24 protocols for being honest.
    net_revenue_24h: {type: float, max_null_fraction: 0.45}              # live 21.8%
    methodology_hash: {type: string}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_contracts.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`
Expected: all green. If a pre-existing test asserted that negative net revenue fails, it encoded the
old ruling — update it to assert the new asymmetry and say so in the commit message.

- [ ] **Step 6: Commit**

```bash
git add config/contracts/defillama_fees.yml tests/test_contracts.py
git commit -m "$(cat <<'MSG'
fix: negative net revenue is real; only gross fees are floored at zero

The contract floored all three revenue columns at zero and blocked the
build on 24 protocols. For most of them the negative figure is a true
description of the day: Ensuro pays insurance claims above premiums,
Hegic's options writers lost, WINR's and Azuro's house lost, and
Fragment's rebates turn $1.05M of fees into -$852K of net revenue.

Gross fees cannot be negative, so those rows keep their floor and are
quarantined as vendor errors.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 5: Roll DefiLlama protocols up to one symbol, refusing double-counts

Spec §5.1 and **Review Focus 5**. DefiLlama splits a protocol across versions and renames it on
rebrand, so an asset's revenue is a *list* of slugs that must be summed. The danger is summing an
aggregate together with its own components — a "Uniswap" row alongside "Uniswap V3" would
double-count — so the rollup refuses that configuration by name rather than silently inflating a DCF.

**Files:**
- Create: `cryptoval/rollup.py`
- Create: `tests/test_rollup.py`

**Interfaces:**
- Consumes: `Asset.defillama_slugs` from Task 1.
- Produces:
  - `cryptoval.rollup.REVENUE_COLUMNS: list[str]` =
    `["total_fees_24h", "total_fees_30d", "net_revenue_24h"]`
  - `rollup_protocols(df: pd.DataFrame, assets: list[Asset]) -> pd.DataFrame` with columns
    `["symbol", "metric_date", *REVENUE_COLUMNS, "protocol_count"]`
  - `assert_no_aggregate_collision(slugs: tuple[str, ...], symbol: str) -> None` — raises
    `ValueError` when one slug is a proper word-boundary prefix of another

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rollup.py`:

```python
import numpy as np
import pandas as pd
import pytest

from cryptoval.config import Asset
from cryptoval.rollup import REVENUE_COLUMNS, assert_no_aggregate_collision, rollup_protocols

UNI = Asset(
    symbol="UNI",
    sector="defi",
    consensus="other",
    cm_id="uni",
    gecko_id="uniswap",
    defillama_slugs=("Uniswap V4", "Uniswap V3", "Uniswap V2"),
)
LDO = Asset(
    symbol="LDO",
    sector="defi",
    consensus="other",
    cm_id="ldo",
    gecko_id="lido-dao",
    defillama_slugs=("Lido",),
)


def _feed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protocol": ["Uniswap V4", "Uniswap V3", "Uniswap V2", "Lido", "Hegic"],
            "metric_date": pd.to_datetime(["2026-10-11"] * 5, utc=True),
            "total_fees_24h": [1370164.0, 450744.0, 39866.0, 1628206.0, 1.36],
            "total_fees_30d": [1.0e7, 1.2e7, 1.0e6, 4.0e7, -71888.64],
            "net_revenue_24h": [0.0, 92267.0, 0.0, 99746.0, 1.36],
        }
    )


def test_versions_are_summed_into_one_symbol():
    out = rollup_protocols(_feed(), [UNI, LDO])
    uni = out.set_index("symbol").loc["UNI"]
    # A DCF on Uniswap V3 alone would be wrong and declining: V4 now carries three times
    # V3's fees and the mix is still shifting (spec 5.1).
    assert uni["total_fees_24h"] == pytest.approx(1370164.0 + 450744.0 + 39866.0)
    assert uni["net_revenue_24h"] == pytest.approx(92267.0)
    assert uni["protocol_count"] == 3


def test_an_unmapped_protocol_is_excluded():
    out = rollup_protocols(_feed(), [UNI, LDO])
    # Hegic maps to no asset we value, so it must not appear.
    assert set(out["symbol"]) == {"UNI", "LDO"}


def test_an_asset_with_no_slugs_is_absent_rather_than_zero():
    btc = Asset(symbol="BTC", sector="pow_monetary", consensus="pow", cm_id="btc")
    out = rollup_protocols(_feed(), [btc])
    assert out.empty
    assert list(out.columns) == ["symbol", "metric_date", *REVENUE_COLUMNS, "protocol_count"]


def test_an_all_null_group_stays_null_rather_than_summing_to_zero():
    feed = _feed()
    feed.loc[feed["protocol"].str.startswith("Uniswap"), "net_revenue_24h"] = np.nan
    out = rollup_protocols(feed, [UNI])
    # Spec global constraint: a missing value is never coerced to 0.0. A zero here would
    # read as "this protocol retains nothing", which is a different claim.
    assert np.isnan(out.set_index("symbol").loc["UNI", "net_revenue_24h"])


def test_a_configured_slug_missing_from_the_feed_lowers_the_count():
    feed = _feed()
    feed = feed[feed["protocol"] != "Uniswap V2"]
    out = rollup_protocols(feed, [UNI])
    assert out.set_index("symbol").loc["UNI", "protocol_count"] == 2


def test_an_aggregate_alongside_its_components_is_refused():
    with pytest.raises(ValueError, match="would double-count"):
        assert_no_aggregate_collision(("Uniswap", "Uniswap V3"), "UNI")


def test_sibling_versions_do_not_collide():
    assert_no_aggregate_collision(("Uniswap V4", "Uniswap V3"), "UNI") is None
    assert_no_aggregate_collision(("Curve DEX", "Curve LlamaLend V2"), "CRV") is None
    assert_no_aggregate_collision(("Sky Lending", "Sky Money"), "MKR") is None


def test_a_prefix_that_is_not_a_word_boundary_does_not_collide():
    # "Aave" is a prefix of "Aavegotchi" as a string but not as a protocol family.
    assert_no_aggregate_collision(("Aave", "Aavegotchi"), "AAVE") is None


def test_the_rollup_refuses_a_colliding_config():
    bad = Asset(
        symbol="UNI",
        sector="defi",
        consensus="other",
        cm_id="uni",
        gecko_id="uniswap",
        defillama_slugs=("Uniswap", "Uniswap V3"),
    )
    with pytest.raises(ValueError, match="would double-count"):
        rollup_protocols(_feed(), [bad])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rollup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptoval.rollup'`

- [ ] **Step 3: Implement `cryptoval/rollup.py`**

```python
"""Roll DefiLlama's per-protocol rows up to one row per valuation symbol.

Spec 5.1. DefiLlama splits a protocol across versions and renames it on rebrand, so
an asset's revenue is a LIST of slugs: UNI is Uniswap V4 plus V3 plus V2, and MKR is
Sky Lending plus Sky Money because no protocol named Maker exists in the feed any
more. Summing the wrong set is not a loud failure -- it is a confidently wrong DCF
input -- so the one configuration that silently inflates a sum is refused here.
"""

from __future__ import annotations

import pandas as pd

from cryptoval.config import Asset

REVENUE_COLUMNS = ["total_fees_24h", "total_fees_30d", "net_revenue_24h"]
OUTPUT_COLUMNS = ["symbol", "metric_date", *REVENUE_COLUMNS, "protocol_count"]


def assert_no_aggregate_collision(slugs: tuple[str, ...], symbol: str) -> None:
    """Refuse a slug set holding both an aggregate and its own components.

    Summing "Uniswap" together with "Uniswap V3" double-counts V3. A shared prefix only
    means a family when it ends at a word boundary: "Aave" and "Aavegotchi" are
    unrelated protocols that happen to share four letters.
    """
    for outer in slugs:
        for inner in slugs:
            if outer == inner:
                continue
            if inner.startswith(outer + " "):
                raise ValueError(
                    f"{symbol}: defillama_slugs holds both {outer!r} and {inner!r}, which "
                    f"would double-count {inner!r}; list either the aggregate or its "
                    "components, not both"
                )


def rollup_protocols(df: pd.DataFrame, assets: list[Asset]) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for asset in assets:
        if not asset.defillama_slugs:
            continue
        assert_no_aggregate_collision(tuple(asset.defillama_slugs), asset.symbol)
        for slug in asset.defillama_slugs:
            mapping[slug] = asset.symbol

    if df.empty or not mapping:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    scoped = df[df["protocol"].isin(mapping)].copy()
    if scoped.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    scoped["symbol"] = scoped["protocol"].map(mapping)

    present = [c for c in REVENUE_COLUMNS if c in scoped.columns]
    grouped = scoped.groupby(["symbol", "metric_date"], as_index=False, dropna=False)
    # min_count=1 keeps an all-null group null: pandas would otherwise sum it to 0.0,
    # which asserts "this protocol retained nothing" rather than "we do not know".
    summed = grouped[present].sum(min_count=1)
    counts = grouped.size().rename(columns={"size": "protocol_count"})
    out = summed.merge(counts, on=["symbol", "metric_date"], how="left")
    return out.reindex(columns=OUTPUT_COLUMNS)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rollup.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Verify the shipped config has no collision**

Run:
```bash
.venv/bin/python -I -c "
from cryptoval.config import load_config
from cryptoval.rollup import assert_no_aggregate_collision
_, assets = load_config()
for a in assets:
    if a.defillama_slugs:
        assert_no_aggregate_collision(tuple(a.defillama_slugs), a.symbol)
        print(f'{a.symbol:5s} {len(a.defillama_slugs)} slug(s): {list(a.defillama_slugs)}')
print('no collisions')
"
```
Expected: one line per mapped asset, then `no collisions`.

- [ ] **Step 6: Run the full suite and lint, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`

```bash
git add cryptoval/rollup.py tests/test_rollup.py
git commit -m "$(cat <<'MSG'
feat: roll DefiLlama protocols up to one symbol, refusing double-counts

An asset's revenue is a list of slugs, not one id: UNI is Uniswap V4 plus
V3 plus V2 with the mix shifting, and MKR is Sky Lending plus Sky Money
because no protocol named Maker exists in the feed any more.

The rollup refuses a slug set holding both an aggregate and its own
components, since that inflates a DCF silently rather than failing. A
shared prefix only counts as a family at a word boundary, so Aave and
Aavegotchi stay unrelated. All-null groups stay null instead of summing
to zero.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 6: Bronze partitions carry a run_id, so re-runs stop destroying their own audit trail

Spec §7.4, revised. Plan 1 partitioned bronze on `ingest_date` alone and *replaced* the partition
wholesale on a re-run, which meant an operator re-running a subset of assets after a partial failure
destroyed the response record of everything fetched earlier that day. That contradicted the layer's
one guarantee. Plan 1 documented a workaround ("re-run the FULL asset list"); this fixes the cause.

**Files:**
- Modify: `cryptoval/paths.py`
- Modify: `cryptoval/io/bronze.py`
- Modify: `cryptoval/cli.py`
- Test: `tests/test_paths.py`, `tests/test_bronze.py`
- Modify: `docs/runbooks/coingecko-rate-limit.md`

**Interfaces:**
- Consumes: `RunRecord.run_id` (existing, a uuid4 string from `cryptoval.manifest`).
- Produces:
  - `PathResolver.bronze_partition(source, ingest_date, run_id) -> str` — **signature change**,
    `run_id` is required
  - `PathResolver.bronze_date_prefix(source, ingest_date) -> str` — the directory holding every run
    for that date
  - `write_bronze(resolver, source, ingest_date, df, request_params, run_id) -> str` —
    **signature change**
  - `read_bronze(resolver, source, ingest_date, run_id=None) -> pd.DataFrame` — newest run when
    `run_id` is omitted

- [ ] **Step 1: Write the failing path tests**

Add to `tests/test_paths.py`:

```python
def test_bronze_partition_includes_the_run_id():
    r = PathResolver("/data")
    assert r.bronze_partition("coinmetrics", dt.date(2026, 10, 12), "run-abc") == (
        "/data/bronze/source=coinmetrics/ingest_date=2026-10-12/run_id=run-abc"
    )


def test_bronze_date_prefix_holds_every_run_for_that_date():
    r = PathResolver("/data")
    assert r.bronze_date_prefix("coinmetrics", dt.date(2026, 10, 12)) == (
        "/data/bronze/source=coinmetrics/ingest_date=2026-10-12"
    )
```

- [ ] **Step 2: Write the failing bronze tests**

Add to `tests/test_bronze.py`:

```python
def test_two_runs_on_the_same_date_both_survive(tmp_path):
    """The defect this fixes: re-running a subset used to delete the earlier run's record."""
    r = PathResolver(str(tmp_path))
    first = pd.DataFrame({"asset": ["btc", "eth"], "v": [1.0, 2.0]})
    second = pd.DataFrame({"asset": ["btc"], "v": [1.5]})

    write_bronze(r, "coinmetrics", dt.date(2026, 10, 12), first, {"a": 1}, "run-1")
    write_bronze(r, "coinmetrics", dt.date(2026, 10, 12), second, {"a": 2}, "run-2")

    runs = sorted(p.name for p in pathlib.Path(
        r.bronze_date_prefix("coinmetrics", dt.date(2026, 10, 12))
    ).iterdir())
    assert runs == ["run_id=run-1", "run_id=run-2"]
    # The partial re-run did not take eth's response with it.
    assert set(read_bronze(r, "coinmetrics", dt.date(2026, 10, 12), "run-1")["asset"]) == {
        "btc",
        "eth",
    }


def test_read_bronze_returns_the_newest_run_by_default(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "coinmetrics", dt.date(2026, 10, 12),
                 pd.DataFrame({"v": [1.0]}), {}, "run-1")
    write_bronze(r, "coinmetrics", dt.date(2026, 10, 12),
                 pd.DataFrame({"v": [9.0]}), {}, "run-2")
    assert read_bronze(r, "coinmetrics", dt.date(2026, 10, 12))["v"].iloc[0] == 9.0


def test_rewriting_the_same_run_id_is_refused(tmp_path):
    """A run_id is unique per run, so a collision means a bug, not an intentional rewrite."""
    r = PathResolver(str(tmp_path))
    write_bronze(r, "coinmetrics", dt.date(2026, 10, 12),
                 pd.DataFrame({"v": [1.0]}), {}, "run-1")
    with pytest.raises(FileExistsError, match="run-1"):
        write_bronze(r, "coinmetrics", dt.date(2026, 10, 12),
                     pd.DataFrame({"v": [2.0]}), {}, "run-1")


def test_reading_a_date_with_no_runs_raises(tmp_path):
    r = PathResolver(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        read_bronze(r, "coinmetrics", dt.date(2026, 10, 12))
```

Delete the existing `test_rewriting_a_date_replaces_the_partition_wholesale` test. It pinned the
behaviour this task removes; the commit message must say so explicitly, because deleting a passing
test is otherwise indistinguishable from losing coverage.

- [ ] **Step 3: Run both test files to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_paths.py tests/test_bronze.py -v`
Expected: FAIL — `bronze_partition() missing 1 required positional argument: 'run_id'`

- [ ] **Step 4: Implement the path changes**

In `cryptoval/paths.py`, replace `bronze_partition`:

```python
    def bronze_date_prefix(self, source: str, ingest_date: dt.date) -> str:
        return "/".join(
            [self.layer("bronze"), f"source={source}", f"ingest_date={ingest_date.isoformat()}"]
        )

    def bronze_partition(self, source: str, ingest_date: dt.date, run_id: str) -> str:
        """One partition per RUN, not per date.

        Date-only partitioning made a same-day re-run replace the partition wholesale, so
        re-running a subset of assets after a partial failure destroyed the response record
        of everything fetched earlier that day (spec 7.4, revised).
        """
        if not run_id or not str(run_id).strip():
            raise ValueError("run_id must be a non-empty identifier")
        return "/".join([self.bronze_date_prefix(source, ingest_date), f"run_id={run_id}"])
```

- [ ] **Step 5: Implement the bronze changes**

In `cryptoval/io/bronze.py`, add `run_id: str` as the final parameter of `write_bronze`, then replace
the write block. The old code deleted `final` before renaming; it must now refuse a collision:

```python
    final = pathlib.Path(resolver.bronze_partition(source, ingest_date, run_id))
    if final.exists():
        # A run_id is unique per run, so this is a bug rather than an intentional rewrite.
        # Bronze is append-only: nothing here deletes.
        raise FileExistsError(f"bronze partition already exists for run {run_id}: {final}")
    staging = final.with_name(final.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        enriched.to_parquet(staging / "data.parquet", index=False)
        staging.rename(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return str(final)
```

Also add `enriched["_run_id"] = run_id` beside the other `_`-prefixed audit columns.

Replace `read_bronze`:

```python
def read_bronze(
    resolver: PathResolver,
    source: str,
    ingest_date: dt.date,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Read one run's partition, defaulting to the newest run for that date."""
    if run_id is not None:
        path = pathlib.Path(resolver.bronze_partition(source, ingest_date, run_id)) / "data.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no bronze partition at {path}")
        return pd.read_parquet(path)

    prefix = pathlib.Path(resolver.bronze_date_prefix(source, ingest_date))
    runs = sorted(
        (p for p in prefix.glob("run_id=*") if (p / "data.parquet").exists()),
        key=lambda p: (p / "data.parquet").stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"no bronze runs under {prefix}")
    return pd.read_parquet(runs[-1] / "data.parquet")
```

- [ ] **Step 6: Update the single call site**

In `cryptoval/cli.py`'s `run_ingest`, pass the run id:

```python
                write_bronze(settings.resolver, name, end, result.frame, params, record.run_id)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_paths.py tests/test_bronze.py tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 8: Correct the runbook that documented the workaround**

`docs/runbooks/coingecko-rate-limit.md` tells the operator to re-run the FULL asset list because a
partial re-run would destroy the day's earlier bronze vintage. That hazard is now gone. Replace that
instruction with: re-running the remaining assets is safe — each run lands in its own
`run_id=` partition and nothing is overwritten. Keep the rest of the runbook as it is.

- [ ] **Step 9: Run the full suite and lint, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`

```bash
git add cryptoval/paths.py cryptoval/io/bronze.py cryptoval/cli.py \
        tests/test_paths.py tests/test_bronze.py docs/runbooks/coingecko-rate-limit.md
git commit -m "$(cat <<'MSG'
fix: bronze partitions carry a run_id, so nothing is ever overwritten

Date-only partitioning made a same-day re-run replace the partition
wholesale, so re-running a subset of assets after a partial failure
destroyed the response record of everything fetched earlier that day --
the opposite of what this layer guarantees. Plan 1 documented a workaround
("re-run the full asset list"); this removes the cause.

Each run now lands under run_id=<uuid>, a collision on the same run_id is
refused rather than overwritten, and read_bronze defaults to the newest
run. Deletes test_rewriting_a_date_replaces_the_partition_wholesale, which
pinned the removed behaviour, and corrects the CoinGecko runbook that told
the operator to work around it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
## Task 7: Seed the asset dimension from config, and resolve vendor ids to symbols in staging

Spec §5.1 and §8.1. `dim_asset` is currently derived from "whatever the panel happened to contain",
so its grain is the Coin Metrics vendor id — `trx_eth`, `avaxc`, `usdc`. The four sources therefore
share no key at all: Coin Metrics keys on its ids, CoinGecko on `tron`, DefiLlama on display names,
yfinance on tickers. This task makes `symbol` the conformed key and seeds the dimension from
`config/assets.yml`.

Bronze and silver are **not** touched: they stay vendor-faithful, because rewriting a vendor's key in
the landing zone would destroy the audit trail the layer exists for. Resolution happens in staging.

**Files:**
- Create: `cryptoval/seeds.py`
- Create: `tests/test_seeds.py`
- Create: `dbt/seeds/assets.csv`, `dbt/seeds/asset_metric_sources.csv` (generated, committed)
- Create: `dbt/models/staging/stg_market_chart.sql`, `stg_protocol_revenue.sql`,
  `stg_equity_prices.sql`
- Modify: `dbt/models/staging/stg_asset_metrics.sql`
- Modify: `dbt/models/marts/dim_asset.sql`, `dbt/models/marts/_marts.yml`
- Modify: `Makefile`
- Test: `tests/test_duck.py`

**Interfaces:**
- Consumes: `Asset.source_for`, `Asset.cm_ids`, `Asset.defillama_slugs`, `Asset.coverage_note` from
  Task 1; `REVENUE_COLUMNS` from Task 5.
- Produces:
  - `cryptoval.seeds.write_asset_seeds(assets, seed_dir) -> list[str]` — writes both CSVs, returns
    their paths
  - `dbt/seeds/assets.csv` columns: `symbol, sector, consensus, is_stablecoin, is_equity, cm_id,
    gecko_id, yahoo_id, defillama_slugs, coverage_note`
  - `dbt/seeds/asset_metric_sources.csv` columns: `symbol, metric_group, source_kind, source_id`
  - dbt models `stg_asset_metrics` (grain `symbol, metric_date`), `stg_market_chart`,
    `stg_protocol_revenue`, `stg_equity_prices`, `dim_asset` (grain `symbol`)

- [ ] **Step 1: Write the failing seed tests**

Create `tests/test_seeds.py`:

```python
import csv
import pathlib

from cryptoval.config import Asset, load_config
from cryptoval.seeds import write_asset_seeds


def _rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_asset_seed_carries_one_row_per_symbol(tmp_path):
    _, assets = load_config()
    paths = write_asset_seeds(assets, str(tmp_path))
    rows = _rows(paths[0])
    assert len(rows) == len(assets)
    assert len({r["symbol"] for r in rows}) == len(assets)


def test_the_metric_source_seed_is_long_not_wide(tmp_path):
    assets = [
        Asset(symbol="AVAX", sector="smart_contract_l1", consensus="pos", cm_id="avaxc",
              gecko_id="avalanche-2", metric_sources={"volume": "cm:avax"}),
    ]
    rows = _rows(write_asset_seeds(assets, str(tmp_path))[1])
    # One row per (symbol, group), so a vendor id serving two groups appears twice and the
    # join in staging stays a simple equi-join.
    assert len(rows) == 3
    by_group = {r["metric_group"]: r for r in rows}
    assert by_group["network"]["source_id"] == "avaxc"
    assert by_group["valuation"]["source_id"] == "avaxc"
    assert by_group["volume"]["source_id"] == "avax"
    assert {r["source_kind"] for r in rows} == {"cm"}


def test_a_coingecko_valuation_source_is_recorded_as_such(tmp_path):
    assets = [
        Asset(symbol="TRX", sector="smart_contract_l1", consensus="pos", cm_id="trx",
              gecko_id="tron", metric_sources={"valuation": "coingecko"}),
    ]
    rows = {r["metric_group"]: r for r in _rows(write_asset_seeds(assets, str(tmp_path))[1])}
    assert rows["valuation"]["source_kind"] == "coingecko"
    assert rows["valuation"]["source_id"] == "tron"
    assert rows["network"]["source_kind"] == "cm"
    assert rows["network"]["source_id"] == "trx"


def test_slugs_are_serialised_as_a_pipe_delimited_field(tmp_path):
    assets = [
        Asset(symbol="UNI", sector="defi", consensus="other", cm_id="uni", gecko_id="uniswap",
              defillama_slugs=("Uniswap V4", "Uniswap V3")),
    ]
    row = _rows(write_asset_seeds(assets, str(tmp_path))[0])[0]
    # Pipe, not comma: these are CSV files and the slugs contain spaces but never pipes.
    assert row["defillama_slugs"] == "Uniswap V4|Uniswap V3"


def test_a_network_source_of_none_is_omitted_rather_than_written_as_a_join_key(tmp_path):
    sol = Asset(symbol="SOL", sector="smart_contract_l1", consensus="pos", cm_id=None,
                gecko_id="solana")
    rows = _rows(write_asset_seeds([sol], str(tmp_path))[1])
    # SOL has no network source at all. A row with source_id "none" would join to nothing
    # and read as a configuration error later; absence is the honest encoding.
    assert {r["metric_group"] for r in rows} == {"valuation", "volume"}


def test_the_seeds_are_deterministic(tmp_path):
    _, assets = load_config()
    first = [pathlib.Path(p).read_text(encoding="utf-8")
             for p in write_asset_seeds(assets, str(tmp_path / "a"))]
    second = [pathlib.Path(p).read_text(encoding="utf-8")
              for p in write_asset_seeds(assets, str(tmp_path / "b"))]
    # Committed generated files must not churn, or every run dirties the tree.
    assert first == second
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_seeds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptoval.seeds'`

- [ ] **Step 3: Implement `cryptoval/seeds.py`**

```python
"""Generate dbt seeds from config/assets.yml.

Spec 8.1: dim_asset is SEEDED, not derived. Deriving it from the panel made its grain
the Coin Metrics vendor id, so `trx_eth` and `avaxc` were the dimension's keys and
nothing joined the four sources. The config is the authority on the asset universe;
these CSVs are a committed projection of it that dbt can read.
"""

from __future__ import annotations

import csv
import pathlib

from cryptoval.config import METRIC_GROUPS, Asset

ASSET_COLUMNS = [
    "symbol",
    "sector",
    "consensus",
    "is_stablecoin",
    "is_equity",
    "cm_id",
    "gecko_id",
    "yahoo_id",
    "defillama_slugs",
    "coverage_note",
]
SOURCE_COLUMNS = ["symbol", "metric_group", "source_kind", "source_id"]


def _write(path: pathlib.Path, columns: list[str], rows: list[dict]) -> str:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def write_asset_seeds(assets: list[Asset], seed_dir: str) -> list[str]:
    out = pathlib.Path(seed_dir)
    out.mkdir(parents=True, exist_ok=True)

    asset_rows = [
        {
            "symbol": a.symbol,
            "sector": a.sector or "",
            "consensus": a.consensus,
            "is_stablecoin": str(a.is_stablecoin).lower(),
            "is_equity": str(a.is_equity).lower(),
            "cm_id": a.cm_id or "",
            "gecko_id": a.gecko_id or "",
            "yahoo_id": a.yahoo_id or "",
            # Pipe-delimited: these are CSVs and the slugs contain spaces but never pipes.
            "defillama_slugs": "|".join(a.defillama_slugs),
            "coverage_note": a.coverage_note,
        }
        for a in assets
    ]

    source_rows: list[dict] = []
    for a in assets:
        if a.is_equity:
            continue
        for group in METRIC_GROUPS:
            source = a.source_for(group)
            if source == "none":
                # A row pointing at nothing would join to nothing and read as a config
                # error downstream. Absence is the honest encoding of "unavailable".
                continue
            if source.startswith("cm:"):
                source_rows.append(
                    {
                        "symbol": a.symbol,
                        "metric_group": group,
                        "source_kind": "cm",
                        "source_id": source[len("cm:") :],
                    }
                )
            else:
                source_rows.append(
                    {
                        "symbol": a.symbol,
                        "metric_group": group,
                        "source_kind": "coingecko",
                        "source_id": a.gecko_id or "",
                    }
                )

    return [
        _write(out / "assets.csv", ASSET_COLUMNS, asset_rows),
        _write(out / "asset_metric_sources.csv", SOURCE_COLUMNS, source_rows),
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_seeds.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Generate and commit the seeds, and add a Makefile target**

Add to the `Makefile` (and to `.PHONY`):

```makefile
seed:
	$(PY) -I -c "from cryptoval.config import load_config; from cryptoval.seeds import write_asset_seeds; _, a = load_config(); print('\n'.join(write_asset_seeds(a, 'dbt/seeds')))"
```

Run: `make seed && head -3 dbt/seeds/assets.csv dbt/seeds/asset_metric_sources.csv`
Expected: both files written; the metric-source file has three rows for AVAX and two for SOL.

- [ ] **Step 6: Declare the seeds in `dbt/dbt_project.yml`**

Add, preserving the existing keys:

```yaml
seeds:
  cryptoval:
    +schema: main
    assets:
      +column_types:
        symbol: varchar
        is_stablecoin: boolean
        is_equity: boolean
    asset_metric_sources:
      +column_types:
        symbol: varchar
        metric_group: varchar
        source_kind: varchar
        source_id: varchar
```

- [ ] **Step 7: Rewrite `dbt/models/staging/stg_asset_metrics.sql`**

The Coin Metrics panel now holds rows for several vendor ids per asset, each carrying a different
subset of metrics. The resolution is one join per metric group:

```sql
-- Current-vintage view of the bi-temporal silver panel, resolved from Coin Metrics'
-- vendor ids to the conformed `symbol` key (spec 5.1).
--
-- Three assets draw different metric groups from different vendor series: AVAX takes
-- volume from `avax` because `avaxc` serves none, and TRX takes valuation from CoinGecko
-- because `cm:trx` has no free market cap. So this is one join per group, not one per
-- asset, and the groups are recombined on (symbol, metric_date).
--
-- Point-in-time analysis in Plan 4 reads the panel directly with an as-of filter.
with panel as (
    select
        asset as vendor_id,
        -- Convert to UTC before truncating: a bare cast uses the session TimeZone and
        -- shifts the date by a day on non-UTC machines.
        cast(metric_date at time zone 'UTC' as date) as metric_date,
        "CapMrktCurUSD"                as market_cap_usd,
        "CapMVRVCur"                   as mvrv,
        realized_cap,
        "PriceUSD"                     as price_usd,
        "SplyCur"                      as supply,
        "HashRate"                     as hash_rate,
        "IssTotUSD"                    as issuance_usd,
        "FeeTotNtv"                    as fees_native,
        "AdrActCnt"                    as active_addresses,
        "TxCnt"                        as tx_count,
        "volume_reported_spot_usd_1d"  as volume_reported_spot_usd,
        effective_start_ts,
        effective_end_ts,
        row_hash
    from read_parquet('{{ var("data_root") }}/silver/asset_metrics/*.parquet')
    where is_current
),
sources as (
    select symbol, metric_group, source_id
    from {{ ref('asset_metric_sources') }}
    where source_kind = 'cm'
),
network as (
    select s.symbol, p.metric_date, p.supply, p.hash_rate, p.issuance_usd, p.fees_native,
           p.active_addresses, p.tx_count, p.effective_start_ts, p.row_hash
    from panel p
    join sources s on p.vendor_id = s.source_id and s.metric_group = 'network'
),
valuation as (
    select s.symbol, p.metric_date, p.market_cap_usd, p.mvrv, p.realized_cap, p.price_usd
    from panel p
    join sources s on p.vendor_id = s.source_id and s.metric_group = 'valuation'
),
volume as (
    select s.symbol, p.metric_date, p.volume_reported_spot_usd
    from panel p
    join sources s on p.vendor_id = s.source_id and s.metric_group = 'volume'
),
spine as (
    select symbol, metric_date from network
    union
    select symbol, metric_date from valuation
    union
    select symbol, metric_date from volume
)
select
    sp.symbol,
    sp.metric_date,
    v.market_cap_usd,
    v.mvrv,
    v.realized_cap,
    v.price_usd,
    n.supply,
    n.hash_rate,
    n.issuance_usd,
    n.fees_native,
    n.active_addresses,
    n.tx_count,
    vol.volume_reported_spot_usd,
    n.effective_start_ts,
    n.row_hash
from spine sp
left join network   n   on sp.symbol = n.symbol   and sp.metric_date = n.metric_date
left join valuation v   on sp.symbol = v.symbol   and sp.metric_date = v.metric_date
left join volume    vol on sp.symbol = vol.symbol and sp.metric_date = vol.metric_date
```

- [ ] **Step 8: Create the three new staging models**

`dbt/models/staging/stg_market_chart.sql`:

```sql
-- CoinGecko's panel, resolved to `symbol`. This is the only market cap available for TRX
-- and SOL, and its volume is a DIFFERENT series from Coin Metrics' reported spot volume
-- (2.97e9 vs 1.48e10 for BTC on the same day) -- never interchange them (spec 10.1).
select
    a.symbol,
    cast(m.metric_date at time zone 'UTC' as date) as metric_date,
    m.price_usd        as cg_price_usd,
    m.market_cap_usd   as cg_market_cap_usd,
    m.volume_usd       as cg_volume_total_usd,
    m.velocity         as cg_velocity,
    m.holding_time_days as cg_holding_time_days
from read_parquet('{{ var("data_root") }}/silver/market_chart/*.parquet') m
join {{ ref('assets') }} a on m.asset = a.gecko_id
where m.is_current
```

`dbt/models/staging/stg_protocol_revenue.sql`:

```sql
-- DefiLlama rolled up to `symbol`. A protocol's revenue is split across versions and
-- renamed on rebrand, so the mapping is a LIST of slugs (spec 5.1): UNI is Uniswap V4
-- plus V3 plus V2, and MKR is Sky Lending plus Sky Money.
--
-- sum(...) over an all-null group would return 0.0 in SQL, which asserts "retained
-- nothing" rather than "unknown", so each column is nulled back out when no input row
-- carried a value.
with mapped as (
    select
        a.symbol,
        cast(r.metric_date at time zone 'UTC' as date) as metric_date,
        r.protocol,
        r.total_fees_24h,
        r.total_fees_30d,
        r.net_revenue_24h,
        r.methodology_hash
    from read_parquet('{{ var("data_root") }}/silver/protocol_revenue/*.parquet') r
    join {{ ref('assets') }} a
      on list_contains(string_split(a.defillama_slugs, '|'), r.protocol)
    where r.is_current
)
select
    symbol,
    metric_date,
    case when count(total_fees_24h)  > 0 then sum(total_fees_24h)  end as total_fees_24h,
    case when count(total_fees_30d)  > 0 then sum(total_fees_30d)  end as total_fees_30d,
    case when count(net_revenue_24h) > 0 then sum(net_revenue_24h) end as net_revenue_24h,
    count(*)                                                           as protocol_count,
    -- One hash per symbol-day: a change in any component protocol's methodology
    -- invalidates the rolled-up series (spec 9.2).
    md5(string_agg(coalesce(methodology_hash, ''), '|' order by protocol))
                                                                       as methodology_hash
from mapped
group by symbol, metric_date
```

`dbt/models/staging/stg_equity_prices.sql`:

```sql
-- Equities and macro series, kept separate from the crypto panel: a different trading
-- calendar, a different freshness SLA (144h, because markets close), and no network
-- metrics at all (spec 8.1).
select
    a.symbol,
    cast(e.metric_date at time zone 'UTC' as date) as metric_date,
    e.price_usd
from read_parquet('{{ var("data_root") }}/silver/equity_prices/*.parquet') e
join {{ ref('assets') }} a on e.asset = a.yahoo_id
where e.is_current
```

- [ ] **Step 9: Rewrite `dbt/models/marts/dim_asset.sql`**

```sql
-- Seeded from config/assets.yml, not derived from the panel (spec 8.1). Observed coverage
-- is joined on as an attribute, so "what does this asset actually have" stays answerable
-- without making it the dimension's grain.
with observed as (
    select
        symbol,
        min(metric_date) as first_observed_on,
        max(metric_date) as last_observed_on,
        count(*)         as observation_days,
        -- Density per load-bearing column: a method's coverage floor is checked against
        -- these (spec 10.1, Review Focus 3 -- BNB's market cap is 19% dense).
        count(market_cap_usd)::double / nullif(count(*), 0) as market_cap_density,
        count(active_addresses)::double / nullif(count(*), 0) as active_address_density,
        count(volume_reported_spot_usd)::double / nullif(count(*), 0) as volume_density,
        count(hash_rate)::double / nullif(count(*), 0) as hash_rate_density
    from {{ ref('stg_asset_metrics') }}
    group by symbol
)
select
    a.symbol,
    a.sector,
    a.consensus,
    a.is_stablecoin,
    a.is_equity,
    a.cm_id,
    a.gecko_id,
    a.yahoo_id,
    a.defillama_slugs,
    a.coverage_note,
    o.first_observed_on,
    o.last_observed_on,
    coalesce(o.observation_days, 0) as observation_days,
    o.market_cap_density,
    o.active_address_density,
    o.volume_density,
    o.hash_rate_density
from {{ ref('assets') }} a
left join observed o on a.symbol = o.symbol
```

- [ ] **Step 10: Update `dbt/models/marts/_marts.yml`**

Change every `dim_asset` and `fct_asset_metrics_daily` test that names the column `asset` to name
`symbol`. Add to `dim_asset`: `unique` and `not_null` on `symbol`, and a `relationships` test from
`stg_asset_metrics.symbol` to `dim_asset.symbol`. Add `not_null` on `symbol` for each new staging
model. Do not add a `not_null` test on any density column: an asset with no observations
legitimately has a null density.

- [ ] **Step 11: Build and inspect**

Run: `make seed && make dbt`
Expected: `dbt build` completes; seeds load; every model builds. If the `list_contains`/
`string_split` pair is unavailable in DuckDB 1.5.6, use
`'|' || a.defillama_slugs || '|' like '%|' || r.protocol || '|%'` instead and note the substitution
in the commit message.

Then verify the resolution actually fixed the three assets:

```bash
.venv/bin/python -I -c "
import duckdb
con = duckdb.connect('data/cryptoval.duckdb', read_only=True)
print(con.execute('''
  select symbol, observation_days, round(market_cap_density,3) as mcap_d,
         round(volume_density,3) as vol_d
  from dim_asset where symbol in (\'TRX\',\'AVAX\',\'POL\',\'BNB\',\'BTC\',\'SOL\')
  order by symbol''').fetchdf().to_string())
"
```
Expected: `AVAX` now has a non-zero `vol_d` (it was 0.0), `TRX` has observation days from the real
TRON series, and every symbol appears even where density is null.

- [ ] **Step 12: Write the DuckDB-level test**

Add to `tests/test_duck.py`:

```python
def test_staging_resolves_vendor_ids_to_symbols(tmp_path):
    """AVAX's two vendor series must recombine into one row per (symbol, date)."""
    import duckdb

    con = duckdb.connect()
    con.execute(
        """
        create table panel as select * from (values
            ('avaxc', date '2026-10-01', 1.0e9, 100.0, null),
            ('avax',  date '2026-10-01', null,  null,  5.0e7)
        ) as t(vendor_id, metric_date, market_cap_usd, active_addresses,
               volume_reported_spot_usd)
        """
    )
    con.execute(
        """
        create table sources as select * from (values
            ('AVAX','network','avaxc'), ('AVAX','valuation','avaxc'), ('AVAX','volume','avax')
        ) as t(symbol, metric_group, source_id)
        """
    )
    out = con.execute(
        """
        with network as (
            select s.symbol, p.metric_date, p.active_addresses from panel p
            join sources s on p.vendor_id = s.source_id and s.metric_group='network'),
        valuation as (
            select s.symbol, p.metric_date, p.market_cap_usd from panel p
            join sources s on p.vendor_id = s.source_id and s.metric_group='valuation'),
        volume as (
            select s.symbol, p.metric_date, p.volume_reported_spot_usd from panel p
            join sources s on p.vendor_id = s.source_id and s.metric_group='volume')
        select n.symbol, n.active_addresses, v.market_cap_usd, vol.volume_reported_spot_usd
        from network n
        join valuation v on n.symbol=v.symbol and n.metric_date=v.metric_date
        join volume vol on n.symbol=vol.symbol and n.metric_date=vol.metric_date
        """
    ).fetchall()
    # One row, carrying the network metrics from avaxc AND the volume from avax.
    assert out == [("AVAX", 100.0, 1.0e9, 5.0e7)]
```

- [ ] **Step 13: Run the full suite and lint, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`

```bash
git add cryptoval/seeds.py tests/test_seeds.py tests/test_duck.py Makefile \
        dbt/seeds dbt/dbt_project.yml dbt/models
git commit -m "$(cat <<'MSG'
feat: symbol becomes the conformed key, and dim_asset is seeded from config

dim_asset was derived from whatever the panel contained, which made its
grain the Coin Metrics vendor id -- trx_eth, avaxc, usdc -- so the four
sources shared no key: CoinGecko keyed on tron, DefiLlama on display
names, yfinance on tickers, and nothing joined.

Staging now resolves vendor id to symbol with one join per metric group,
which is what lets AVAX take volume from cm:avax while its network
metrics come from cm:avaxc. Bronze and silver stay vendor-faithful; the
landing zone keeps the vendor's own key because that is the audit trail.

dim_asset carries per-column observed density, which Task 10's method
registry checks coverage floors against.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
## Task 8: The conformed facts — both volume definitions, a recorded market-cap source

Spec §8.1 and §10.1. `fct_asset_metrics_daily` becomes the join of all three crypto sources on
`(symbol, metric_date)`. Two rules govern it, and both exist because a silent choice here would make
a method's answer unreproducible:

- **The two volume series are never coalesced.** Coin Metrics' reported spot volume and CoinGecko's
  total volume differ 5× on BTC on the same day. Both are carried; methods pick and declare.
- **Market cap *is* resolved**, because every method needs one — but the row records which source
  supplied it, so TRX's CoinGecko-sourced market cap is never mistaken for a Coin Metrics figure.

**Files:**
- Modify: `dbt/models/marts/fct_asset_metrics_daily.sql`
- Create: `dbt/models/marts/fct_protocol_revenue_daily.sql`, `fct_equity_prices_daily.sql`
- Modify: `dbt/models/marts/_marts.yml`
- Test: `tests/test_duck.py`

**Interfaces:**
- Consumes: `stg_asset_metrics`, `stg_market_chart`, `stg_protocol_revenue`, `stg_equity_prices`,
  `asset_metric_sources` from Task 7.
- Produces: `fct_asset_metrics_daily` at grain `(symbol, metric_date)` with columns
  `market_cap_usd, market_cap_source, mvrv, realized_cap, price_usd, supply, hash_rate,
  issuance_usd, fees_native, active_addresses, tx_count, volume_reported_spot_usd,
  volume_total_usd, velocity_reported, velocity_total, miner_revenue_usd, thermocap_usd,
  thermocap_has_gap, mvtv, investor_cap_usd, panel_vintage_ts`.

- [ ] **Step 1: Rewrite `dbt/models/marts/fct_asset_metrics_daily.sql`**

```sql
-- The conformed daily fact: one row per (symbol, date), joining Coin Metrics' network
-- metrics, its valuation metrics where free, and CoinGecko's panel.
--
-- Two deliberate asymmetries (spec 10.1):
--  * VOLUME is never coalesced. Coin Metrics' reported spot volume and CoinGecko's total
--    volume are different constructions with different exchange coverage -- 2.97e9 against
--    1.48e10 for BTC on the same day. Velocity, NVT and P/U all turn on this term, so both
--    are carried and every method names which it used.
--  * MARKET CAP is resolved, because every method needs one, but market_cap_source records
--    where it came from: TRX has no free Coin Metrics market cap at all (spec 5.1).
--
-- A NULL input day contributes 0 to thermocap (so the series stays usable as a lower
-- bound) and raises thermocap_has_gap from that row onward.
with base as (
    select
        m.*,
        c.cg_market_cap_usd,
        c.cg_volume_total_usd,
        v.source_kind as valuation_source_kind
    from {{ ref('stg_asset_metrics') }} m
    left join {{ ref('stg_market_chart') }} c
           on m.symbol = c.symbol and m.metric_date = c.metric_date
    left join (
        select symbol, source_kind
        from {{ ref('asset_metric_sources') }}
        where metric_group = 'valuation'
    ) v on m.symbol = v.symbol
),
resolved as (
    select
        *,
        case
            when valuation_source_kind = 'coingecko' then cg_market_cap_usd
            else market_cap_usd
        end as resolved_market_cap_usd,
        case
            when valuation_source_kind = 'coingecko' then 'coingecko'
            when market_cap_usd is not null then 'coinmetrics'
        end as market_cap_source
    from base
),
derived as (
    select
        *,
        issuance_usd + (fees_native * price_usd)                    as miner_revenue_usd,
        sum(coalesce(issuance_usd + (fees_native * price_usd), 0)) over (
            partition by symbol order by metric_date
            rows between unbounded preceding and current row
        )                                                           as thermocap_usd,
        coalesce(bool_or(issuance_usd is null or fees_native is null or price_usd is null) over (
            partition by symbol order by metric_date
            rows between unbounded preceding and current row
        ), false)                                                   as thermocap_has_gap
    from resolved
)
select
    symbol,
    metric_date,
    resolved_market_cap_usd                        as market_cap_usd,
    market_cap_source,
    mvrv,
    realized_cap,
    coalesce(price_usd, cg_price_usd)              as price_usd,
    supply,
    hash_rate,
    issuance_usd,
    fees_native,
    active_addresses,
    tx_count,
    volume_reported_spot_usd,
    cg_volume_total_usd                            as volume_total_usd,
    -- Two velocities, one per volume definition. Zero or negative denominators yield NULL,
    -- never infinity.
    case when resolved_market_cap_usd > 0
         then volume_reported_spot_usd / resolved_market_cap_usd end as velocity_reported,
    case when resolved_market_cap_usd > 0
         then cg_volume_total_usd / resolved_market_cap_usd end      as velocity_total,
    miner_revenue_usd,
    thermocap_usd,
    thermocap_has_gap,
    case when thermocap_usd > 0
         then resolved_market_cap_usd / thermocap_usd end            as mvtv,
    realized_cap - thermocap_usd                                     as investor_cap_usd,
    -- The panel vintage this row was built from. Review Focus 4: a quarantined cell
    -- changes velocity and therefore every Family A and D estimate, so an estimate must
    -- be traceable to the panel state that produced it.
    effective_start_ts                                               as panel_vintage_ts
from derived
```

Note `price_usd` is coalesced while volume is not. That is not an inconsistency: price is the same
quantity measured by two vendors to within a fraction of a percent, whereas reported-spot and
total volume are *different constructions*. Record that reasoning in the `_marts.yml` description.

- [ ] **Step 2: Create `dbt/models/marts/fct_protocol_revenue_daily.sql`**

```sql
-- Protocol revenue at the valuation grain. Only mapped assets appear: the feed carries
-- 2,991 protocols and at most a handful map to anything we value (spec 5.1).
select
    symbol,
    metric_date,
    total_fees_24h,
    total_fees_30d,
    net_revenue_24h,
    protocol_count,
    methodology_hash,
    -- The denominator discipline spec 10.4 demands, made explicit: a zero net revenue
    -- against non-zero fees is the Uniswap case -- the protocol fee switch is off and all
    -- fees go to liquidity providers -- which is a finding, not a gap.
    case
        when net_revenue_24h is null then 'unknown'
        when net_revenue_24h = 0 and total_fees_24h > 0 then 'fees_fully_distributed'
        when net_revenue_24h < 0 then 'net_outflow'
        else 'retains_revenue'
    end as revenue_regime
from {{ ref('stg_protocol_revenue') }}
```

- [ ] **Step 3: Create `dbt/models/marts/fct_equity_prices_daily.sql`**

```sql
-- Equities and macro series. Separate from the crypto fact by design (spec 8.1): a
-- different trading calendar means a shared date spine would manufacture nulls on every
-- weekend and holiday.
select
    e.symbol,
    e.metric_date,
    e.price_usd,
    a.sector
from {{ ref('stg_equity_prices') }} e
left join {{ ref('assets') }} a on e.symbol = a.symbol
```

- [ ] **Step 4: Add the mart tests to `dbt/models/marts/_marts.yml`**

For `fct_asset_metrics_daily`: `not_null` on `symbol` and `metric_date`;
`dbt_utils.unique_combination_of_columns` on `[symbol, metric_date]`;
`accepted_values` on `market_cap_source` with `values: ['coinmetrics', 'coingecko']` and
`config: {where: "market_cap_source is not null"}`; `dbt_utils.accepted_range` on `mvrv`
`{min_value: 0, inclusive: false, where: "mvrv is not null"}`.

For `fct_protocol_revenue_daily`: `accepted_values` on `revenue_regime` with
`['unknown', 'fees_fully_distributed', 'net_outflow', 'retains_revenue']`;
`dbt_utils.unique_combination_of_columns` on `[symbol, metric_date]`.

For `fct_equity_prices_daily`: `not_null` on `symbol`, `metric_date`, `price_usd`;
`dbt_utils.unique_combination_of_columns` on `[symbol, metric_date]`.

Keep the existing `market_cap_usd` `not_null` test as a **warn**, not an error — it already warns on
7,805 nulls, which is normal vendor sparsity.

- [ ] **Step 5: Write the failing fact test**

Add to `tests/test_duck.py`:

```python
def test_the_two_volume_series_are_both_carried_and_never_coalesced():
    """Spec 10.1: they differ 5x on BTC, so a method must choose explicitly."""
    import duckdb

    con = duckdb.connect()
    con.execute(
        """
        create table fct as select * from (values
            ('BTC', date '2026-10-10', 1.666e12, 'coinmetrics', 2.97e9, 1.48e10),
            ('TRX', date '2026-10-10', 3.14e10,  'coingecko',   5.84e7, 2.54e8)
        ) as t(symbol, metric_date, market_cap_usd, market_cap_source,
               volume_reported_spot_usd, volume_total_usd)
        """
    )
    rows = con.execute(
        """
        select symbol, market_cap_source,
               volume_total_usd / nullif(volume_reported_spot_usd, 0) as ratio
        from fct order by symbol
        """
    ).fetchall()
    btc = rows[0]
    # If a model ever coalesced them the ratio would be exactly 1.0 and the 5x gap would
    # vanish without anyone noticing.
    assert btc[2] > 4.0
    # TRX's market cap can only come from CoinGecko; the row says so rather than implying
    # a Coin Metrics figure.
    assert rows[1][1] == "coingecko"
```

- [ ] **Step 6: Build and verify on live data**

Run: `make dbt`
Expected: `PASS` on every test, `ERROR=0`. One `WARN` on `market_cap_usd` nulls is expected.

Then confirm the three corrections landed:

```bash
.venv/bin/python -I -c "
import duckdb
con = duckdb.connect('data/cryptoval.duckdb', read_only=True)
print(con.execute('''
  select symbol, market_cap_source, count(*) days,
         count(volume_reported_spot_usd) vol_cm, count(volume_total_usd) vol_cg,
         count(market_cap_usd) mcap
  from fct_asset_metrics_daily
  where symbol in (\'TRX\',\'AVAX\',\'BTC\',\'SOL\',\'UNI\')
  group by 1,2 order by 1''').fetchdf().to_string())
print(con.execute('''
  select revenue_regime, count(*) from fct_protocol_revenue_daily
  group by 1 order by 2 desc''').fetchdf().to_string())
"
```
Expected: `TRX` shows `market_cap_source = coingecko` with a non-zero `mcap`; `AVAX` shows a non-zero
`vol_cm`; `UNI`'s revenue regime includes `fees_fully_distributed`.

- [ ] **Step 7: Run the full suite and lint, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`

```bash
git add dbt/models/marts tests/test_duck.py
git commit -m "$(cat <<'MSG'
feat: conformed facts carrying both volume series and the market-cap source

fct_asset_metrics_daily now joins all three crypto sources on
(symbol, metric_date), and adds the equity and protocol-revenue facts.

Volume is never coalesced: Coin Metrics' reported spot volume and
CoinGecko's total volume are different constructions that differ 5x on
BTC on the same day, and velocity, NVT and P/U all turn on that term.
Market cap IS resolved, because every method needs one, but the row
records which vendor supplied it -- TRX has no free Coin Metrics market
cap at all.

Rows carry panel_vintage_ts so an estimate stays traceable to the panel
state it was computed from, which matters now that quarantine can change
a velocity between runs.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 9: Scope the blocking gate to the valuation universe, then re-backfill

Spec §9.2. The revenue feed carries 2,991 protocols. Garbage in the long tail currently blocks a
build that will never value it — `make validate` exits 1 on `protocol_revenue/ranges` because of
negative fees at Lombard LBTC and Yield Basis, neither of which maps to any asset. A contract failure
must block for a *mapped* protocol and warn for the rest. With that done, the whole of stage 2a is
re-run against live data.

**Files:**
- Modify: `cryptoval/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `rollup_protocols`'s mapping logic — reuse `Asset.defillama_slugs` directly rather than
  re-deriving it.
- Produces: `run_validate(settings, assets) -> Validation` — **signature change**, `assets` is now
  required so the gate knows which entities are in scope.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_an_unmapped_protocols_failure_warns_while_a_mapped_one_blocks(tmp_path):
    """Spec 9.2: 2,991 protocols are ingested; a handful matter. The long tail must not
    hold the build hostage, and a real defect in a valuation input must still stop it."""
    settings = Settings(data_root=str(tmp_path))
    uni = Asset(
        symbol="UNI", sector="defi", consensus="other", cm_id="uni", gecko_id="uniswap",
        defillama_slugs=("Uniswap V3",),
    )

    def _write(rows: list[dict]) -> None:
        frame = pd.DataFrame(rows)
        frame["effective_start_ts"] = pd.Timestamp("2026-10-11", tz="UTC")
        frame["effective_end_ts"] = pd.NaT
        frame["is_current"] = True
        frame["row_hash"] = "h"
        write_silver(settings.resolver, "protocol_revenue", frame)

    # A negative GROSS fee is a vendor error the contract blocks -- but on a protocol we
    # never value, so it is only a warning.
    _write([
        {"protocol": "Lombard LBTC", "metric_date": pd.Timestamp("2026-10-11", tz="UTC"),
         "total_fees_24h": -23915.0, "total_fees_30d": 1581269.0, "net_revenue_24h": -1913.0},
    ])
    report = cli.run_validate(settings, [uni])
    assert not [f for f in report.failures if "protocol_revenue/ranges" in f]
    assert [w for w in report.warnings if "protocol_revenue/ranges" in w]

    # The same defect on a MAPPED protocol blocks.
    _write([
        {"protocol": "Uniswap V3", "metric_date": pd.Timestamp("2026-10-11", tz="UTC"),
         "total_fees_24h": -450744.0, "total_fees_30d": 1.2e7, "net_revenue_24h": 92267.0},
    ])
    report = cli.run_validate(settings, [uni])
    assert [f for f in report.failures if "protocol_revenue/ranges" in f]


def test_scoping_leaves_the_other_sources_blocking():
    """Only DefiLlama has a long tail. An asset-metrics defect always blocks."""
    assert cli.SCOPED_SOURCES == {"defillama"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py -k "unmapped or scoping" -v`
Expected: FAIL — `run_validate() takes 1 positional argument but 2 were given`

- [ ] **Step 3: Implement the scoping in `cryptoval/cli.py`**

Add near `VOLUME_CHECKS`:

```python
# Only DefiLlama has a long tail: it serves 2,991 protocols of which a handful map to an
# asset we value. A contract failure on an unmapped protocol is real but not ours to
# block on, so it warns (spec 9.2). Every other source is entirely in scope.
SCOPED_SOURCES = {"defillama"}
SCOPE_KEY = {"defillama": "protocol"}
```

Replace `run_validate`:

```python
def run_validate(settings: Settings, assets: list[Asset]) -> Validation:
    failures: list[str] = []
    warnings: list[str] = []
    in_scope = {slug for a in assets for slug in a.defillama_slugs}

    for name, table in SILVER_TABLE.items():
        history = _read_silver(settings, table)
        if history.empty:
            failures.append(f"{table}: no data ingested")
            continue
        # Superseded vintages are kept by design; contracts apply to what is current.
        df = current_view(history)
        contract = load_contract(SOURCES[name].contract_path)

        scoped, unscoped = df, df.iloc[0:0]
        if name in SCOPED_SOURCES:
            key = SCOPE_KEY[name]
            if key in df.columns:
                mask = df[key].isin(in_scope)
                scoped, unscoped = df[mask], df[~mask]

        for result in run_checks(scoped, contract, now=dt.datetime.now(dt.UTC)):
            if not result.passed:
                failures.append(f"{table}/{result.name}: {result.detail}")
        for result in run_identity_checks(scoped):
            if not result.passed:
                failures.append(f"{table}/{result.name}: {result.detail}")

        if not unscoped.empty:
            for result in run_checks(unscoped, contract, now=dt.datetime.now(dt.UTC)):
                if not result.passed:
                    warnings.append(f"{table}/{result.name} [unmapped]: {result.detail}")

        warnings.extend(_volume_anomaly_warnings(name, table, df, contract))
    return Validation(failures, warnings)
```

In `main`, pass the assets: `report = run_validate(settings, assets)`.

**Watch out:** when `in_scope` is empty — a config with no slugs at all — `scoped` becomes empty and
every check trivially passes. That is correct behaviour (nothing is in scope) but it would hide a
real defect from an operator who forgot the mapping. Add this guard right after `in_scope` is built:

```python
    if not in_scope:
        warnings.append(
            "no defillama_slugs configured on any asset, so every protocol is unmapped "
            "and its contract failures only warn"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Re-run the whole pipeline against live data**

This is the stage-2a exit criterion. Bronze now carries `run_id`, so this adds partitions rather than
replacing them, and the existing bronze stays readable.

```bash
make seed
.venv/bin/python -m cryptoval.cli backfill --start 2015-01-01 --sources coinmetrics
.venv/bin/python -m cryptoval.cli ingest --trailing-days 1 --sources defillama
.venv/bin/python -m cryptoval.cli backfill --start 2015-01-01 --sources equities
.venv/bin/python -m cryptoval.cli ingest --trailing-days 300 --sources coingecko
make dbt
make validate
```

CoinGecko's keyless tier serves at most 365 days, so `--trailing-days 300` is deliberate; do not try
to backfill it to 2015.

- [ ] **Step 6: Record the live outcome in the task report**

Capture, verbatim: the row counts per silver table, `make dbt`'s PASS/WARN/ERROR line, `make
validate`'s exit code and its failure list, and the quarantine table's contents
(`select column, count(*), min(raw_value), max(raw_value) from quarantine group by 1`).

**Expected:** `make validate` exits 0. The two blocking failures Plan 1 ended with should both be
gone — `asset_metrics/ranges` because the 14 impossible volumes are now quarantined rather than
stored, and `protocol_revenue/ranges` because the negative net revenues are legitimate and the
negative gross fees belong to unmapped protocols.

**If it does not exit 0, stop and report rather than widening a bound.** A failure here is
information: it means a gate is catching something this plan did not predict, and the fix belongs in
a ruling, not in a quietly relaxed threshold.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/cli.py tests/test_cli.py dbt/seeds
git commit -m "$(cat <<'MSG'
fix: scope the blocking contract gate to the valuation universe

DefiLlama serves 2,991 protocols and a handful map to an asset we value,
so negative fees at Lombard LBTC and Yield Basis were blocking a build
that will never value either. A contract failure now blocks for a mapped
protocol and warns for the rest, and an empty mapping warns loudly rather
than passing everything silently.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
# Stage 2b — The Method Registry

## Task 10: The method contract, with applicability computed and circularity declared

Spec §10.1. Every method is a pure function over the panel behind one signature. Two things are
*computed* here rather than written per method, because writing them 11 times is where they get
forgotten:

- **Applicability** from declared data dependencies and the measured coverage density Task 7 put in
  `dim_asset`. **Review Focus 3:** BNB's market cap is 19% dense — a regression on 646 scattered days
  converges and returns a confident number, so density is a precondition, not a footnote.
- **Circularity.** **Review Focus 1:** QTM with velocity taken from the panel is
  `volume / market_cap`, so the whole expression algebraically collapses to
  `price / circulating_fraction`. It returns the market price back, dressed as a fair value, and every
  residual is zero by construction. A method that consumes any market-cap-derived column is flagged,
  so the card and the report cannot present it as independent evidence.

**Files:**
- Create: `cryptoval/methods/__init__.py`, `cryptoval/methods/base.py`
- Create: `tests/test_methods_base.py`

**Interfaces:**
- Consumes: nothing from stage 2a's Python; it reads frames shaped like
  `fct_asset_metrics_daily`.
- Produces:
  - `Estimate` — frozen dataclass: `symbol, metric_date, method_id, param_set_hash, fair_value,
    band_low, band_high, inputs_used, diagnostics, applicable, reason, panel_vintage_ts`
  - `MethodSpec` — frozen dataclass: `method_id, family, requires, min_density, min_rows, params,
    citation, returns_fair_value`
  - `MARKET_CAP_DERIVED: frozenset[str]`
  - `resolve_applicability(spec, frame) -> tuple[bool, str]`
  - `circularity(spec) -> tuple[str, ...]` — the market-cap-derived columns the method consumes
  - `Method` protocol: `spec: MethodSpec` and
    `estimate(frame: pd.DataFrame, params: dict) -> list[Estimate]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_methods_base.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.base import (
    MARKET_CAP_DERIVED,
    Estimate,
    MethodSpec,
    circularity,
    resolve_applicability,
)

SPEC = MethodSpec(
    method_id="realized_price",
    family="E_cost",
    requires=("realized_cap", "supply"),
    min_density=0.5,
    min_rows=30,
    params={},
    citation="Coinmetrics (2018)",
)


def _frame(n: int = 100, realized_nulls: int = 0) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "symbol": ["BTC"] * n,
            "metric_date": pd.date_range("2026-01-01", periods=n, freq="D").date,
            "realized_cap": np.linspace(1e12, 1.1e12, n),
            "supply": np.linspace(1.9e7, 1.95e7, n),
            "panel_vintage_ts": pd.Timestamp("2026-10-12", tz="UTC"),
        }
    )
    if realized_nulls:
        frame.loc[: realized_nulls - 1, "realized_cap"] = np.nan
    return frame


def test_a_complete_frame_is_applicable():
    ok, reason = resolve_applicability(SPEC, _frame())
    assert ok and reason == ""


def test_a_missing_column_is_not_applicable_and_names_it():
    ok, reason = resolve_applicability(SPEC, _frame().drop(columns=["supply"]))
    assert not ok
    assert "supply" in reason and "absent" in reason


def test_a_column_below_the_density_floor_is_not_applicable():
    """Review Focus 3: BNB's market cap is 19% dense. A fit on 646 scattered days
    converges and returns a confident number, so density gates the method."""
    ok, reason = resolve_applicability(SPEC, _frame(n=100, realized_nulls=81))
    assert not ok
    assert "realized_cap" in reason
    assert "19.0% dense" in reason
    assert "floor 50.0%" in reason


def test_too_few_rows_is_not_applicable():
    ok, reason = resolve_applicability(SPEC, _frame(n=10))
    assert not ok
    assert "10 rows" in reason and "30" in reason


def test_an_empty_frame_is_not_applicable_rather_than_an_error():
    ok, reason = resolve_applicability(SPEC, _frame().iloc[0:0])
    assert not ok
    assert "no rows" in reason


def test_a_method_consuming_a_market_cap_derived_column_is_flagged_circular():
    """Review Focus 1: QTM with panel velocity is volume/market_cap, so the expression
    collapses to price/circulating_fraction and returns the market price back."""
    qtm = MethodSpec(
        method_id="crypto_qtm",
        family="A_monetary",
        requires=("volume_total_usd", "supply", "velocity_total"),
        min_density=0.5,
        min_rows=30,
        params={},
        citation="dissertation 3.1",
    )
    assert circularity(qtm) == ("velocity_total",)


def test_a_method_on_independent_fundamentals_is_not_circular():
    metcalfe = MethodSpec(
        method_id="metcalfe_estimated",
        family="B_network",
        requires=("market_cap_usd", "active_addresses", "supply"),
        min_density=0.5,
        min_rows=180,
        params={},
        citation="Metcalfe (1995)",
    )
    # market_cap_usd itself is the REGRESSAND, not a derived input, so it is not circular.
    assert circularity(metcalfe) == ()


def test_every_market_cap_derived_column_is_named():
    # Each of these is literally computed from market cap in fct_asset_metrics_daily, so a
    # method consuming one and returning a price is reusing its own answer.
    assert MARKET_CAP_DERIVED == frozenset(
        {
            "velocity_reported",
            "velocity_total",
            "mvrv",
            "mvtv",
            "realized_cap",
            "investor_cap_usd",
        }
    )


def test_an_estimate_refuses_a_fair_value_when_inapplicable():
    with pytest.raises(ValueError, match="inapplicable estimate cannot carry a fair_value"):
        Estimate(
            symbol="BNB",
            metric_date=dt.date(2026, 10, 12),
            method_id="realized_price",
            param_set_hash="abc",
            fair_value=42.0,
            band_low=None,
            band_high=None,
            inputs_used={},
            diagnostics={},
            applicable=False,
            reason="market_cap_usd is 19.1% dense",
            panel_vintage_ts=pd.Timestamp("2026-10-12", tz="UTC"),
        )


def test_an_estimate_refuses_an_inverted_band():
    with pytest.raises(ValueError, match="band_low .* above band_high"):
        Estimate(
            symbol="BTC",
            metric_date=dt.date(2026, 10, 12),
            method_id="metcalfe_estimated",
            param_set_hash="abc",
            fair_value=50000.0,
            band_low=60000.0,
            band_high=40000.0,
            inputs_used={},
            diagnostics={},
            applicable=True,
            reason="",
            panel_vintage_ts=pd.Timestamp("2026-10-12", tz="UTC"),
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptoval.methods'`

- [ ] **Step 3: Create `cryptoval/methods/__init__.py`**

```python
"""Valuation methods: pure functions over the conformed panel (spec section 10)."""
```

- [ ] **Step 4: Implement `cryptoval/methods/base.py`**

```python
"""The method contract (spec 10.1).

Every method is a pure function over the panel with one signature, so estimates from
different families are directly comparable -- and the comparison IS the research output.

Two properties are computed here rather than restated by each method, because a property
restated eleven times is a property forgotten once:

* APPLICABILITY, from declared dependencies and measured density. The free tier is not
  uniform: BNB's market cap is 19% dense and TRX has no free MVRV at all. A fit on sparse
  data converges and returns a confident number, which is worse than a refusal.
* CIRCULARITY. Several panel columns are themselves computed from market cap. A method
  consuming one and returning a price in exchange is algebraically handing back the market
  price: QTM with panel velocity reduces to price/circulating_fraction, and every residual
  is zero by construction. The flag travels into the method card and the report so such a
  number is never presented as independent evidence (spec 14, "method circularity").
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

# Columns in fct_asset_metrics_daily that are derived FROM market cap.
MARKET_CAP_DERIVED = frozenset(
    {
        "velocity_reported",  # volume / market_cap
        "velocity_total",  # volume / market_cap
        "mvrv",  # market_cap / realized_cap
        "mvtv",  # market_cap / thermocap
        "realized_cap",  # market_cap / mvrv
        "investor_cap_usd",  # realized_cap - thermocap
    }
)


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    family: str
    requires: tuple[str, ...]
    min_density: float
    min_rows: int
    params: dict[str, Any]
    citation: str
    returns_fair_value: bool = True


@dataclass(frozen=True)
class Estimate:
    symbol: str
    metric_date: dt.date
    method_id: str
    param_set_hash: str
    fair_value: float | None
    band_low: float | None
    band_high: float | None
    inputs_used: dict[str, Any]
    diagnostics: dict[str, Any]
    applicable: bool
    reason: str = ""
    panel_vintage_ts: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if not self.applicable and self.fair_value is not None:
            raise ValueError(
                f"{self.method_id}/{self.symbol}: an inapplicable estimate cannot carry a "
                f"fair_value (got {self.fair_value!r}); that is how a refusal becomes a number"
            )
        if self.band_low is not None and self.band_high is not None:
            if self.band_low > self.band_high:
                raise ValueError(
                    f"{self.method_id}/{self.symbol}: band_low {self.band_low} is above "
                    f"band_high {self.band_high}"
                )


def circularity(spec: MethodSpec) -> tuple[str, ...]:
    """The market-cap-derived columns this method consumes, in declaration order."""
    return tuple(c for c in spec.requires if c in MARKET_CAP_DERIVED)


def resolve_applicability(spec: MethodSpec, frame: pd.DataFrame) -> tuple[bool, str]:
    """Decide whether ``spec`` can honestly run on ``frame``.

    Returns (applicable, reason). The reason is empty when applicable, and otherwise
    carries the measurement that made it inapplicable -- a reason without a number is
    not actionable by whoever reads the method card.
    """
    if frame.empty:
        return False, "no rows in the panel slice"

    missing = [c for c in spec.requires if c not in frame.columns]
    if missing:
        return False, f"required column(s) absent from the panel: {missing}"

    if len(frame) < spec.min_rows:
        return False, f"only {len(frame)} rows, below the method's minimum of {spec.min_rows}"

    for column in spec.requires:
        density = float(frame[column].notna().mean())
        if density < spec.min_density:
            return False, (
                f"{column} is {density:.1%} dense, below this method's floor "
                f"{spec.min_density:.1%}"
            )
    return True, ""


@runtime_checkable
class Method(Protocol):
    spec: MethodSpec

    def estimate(self, frame: pd.DataFrame, params: dict[str, Any]) -> list[Estimate]: ...
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_methods_base.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 6: Run the full suite and lint, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`

```bash
git add cryptoval/methods tests/test_methods_base.py
git commit -m "$(cat <<'MSG'
feat: the method contract, with applicability and circularity computed

Spec 10.1. One signature for every method, so estimates across families
are directly comparable.

Applicability is resolved from declared dependencies and measured density
rather than written per method: the free tier is not uniform, and a fit on
BNB's 19%-dense market cap converges and returns a confident number, which
is worse than a refusal. An inapplicable Estimate cannot carry a
fair_value at all -- the dataclass refuses it.

Circularity is declared the same way. Six panel columns are computed from
market cap, and a method consuming one while returning a price hands the
market price back: QTM with panel velocity reduces to
price/circulating_fraction with every residual zero by construction. The
flag travels into the method card so such a number is never presented as
independent evidence.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 11: The registry — parameter sets, persistence, and the estimate fact

Spec §8.1 and §10.1. The registry discovers methods from `config/methods/*.yml`, runs each over each
asset's panel slice, and writes `gold/valuation_estimate`. **Review Focus 4:** a quarantined cell
changes velocity and therefore every Family A and D estimate for that asset-day, so every row carries
the `panel_vintage_ts` it was computed from.

**Files:**
- Create: `cryptoval/methods/registry.py`
- Create: `tests/test_methods_registry.py`
- Create: `config/methods/realized_price.yml`
- Create: `dbt/models/marts/fct_valuation_estimate.sql`, `dbt/models/marts/dim_method.sql`
- Create: `dbt/seeds/methods.csv` (generated)

**Interfaces:**
- Consumes: `Estimate`, `MethodSpec`, `resolve_applicability`, `circularity` from Task 10;
  `stable_hash` from `cryptoval.hashing` (existing).
- Produces:
  - `load_method_specs(config_dir="config/methods") -> dict[str, MethodSpec]`
  - `METHODS: dict[str, Method]` — the registry, populated by `register()` at import of each family
    module
  - `register(cls: type) -> type` — class decorator; registers an *instance*
  - `register_instance(method: Method) -> Method` — for the case where one class serves two specs
  - `run_methods(panel, specs, method_ids=None) -> pd.DataFrame` with columns
    `symbol, metric_date, method_id, param_set_hash, fair_value, band_low, band_high,
    applicable, reason, circular_inputs, inputs_used, diagnostics, panel_vintage_ts`
  - `param_set_hash(params: dict) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_methods_registry.py`:

```python
import json

import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.registry import (
    ESTIMATE_COLUMNS,
    load_method_specs,
    param_set_hash,
    run_methods,
)


class DoublePrice:
    """A stub method: fair value is twice the realized price."""

    spec = MethodSpec(
        method_id="double_realized",
        family="E_cost",
        requires=("realized_cap", "supply"),
        min_density=0.5,
        min_rows=3,
        params={"multiple": 2.0},
        citation="test",
    )

    def estimate(self, frame, params):
        out = []
        for _, row in frame.iterrows():
            out.append(
                Estimate(
                    symbol=row["symbol"],
                    metric_date=row["metric_date"],
                    method_id=self.spec.method_id,
                    param_set_hash=param_set_hash(params),
                    fair_value=params["multiple"] * row["realized_cap"] / row["supply"],
                    band_low=None,
                    band_high=None,
                    inputs_used={"realized_cap": row["realized_cap"], "supply": row["supply"]},
                    diagnostics={},
                    applicable=True,
                    panel_vintage_ts=row["panel_vintage_ts"],
                )
            )
        return out


def _panel(symbols=("BTC", "BNB")) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        n = 10
        frame = pd.DataFrame(
            {
                "symbol": [sym] * n,
                "metric_date": pd.date_range("2026-01-01", periods=n, freq="D").date,
                "realized_cap": np.linspace(1e12, 1.1e12, n),
                "supply": np.full(n, 1.9e7),
                "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
            }
        )
        if sym == "BNB":
            # 20% dense, like the real BNB market cap: below the method's floor.
            frame.loc[2:, "realized_cap"] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_estimates_are_produced_per_symbol_and_date():
    out = run_methods(_panel(("BTC",)), {"double_realized": DoublePrice()})
    assert len(out) == 10
    assert list(out.columns) == ESTIMATE_COLUMNS
    assert out["fair_value"].iloc[0] == pytest.approx(2.0 * 1e12 / 1.9e7)


def test_an_inapplicable_asset_yields_one_refusal_row_not_a_silent_gap():
    out = run_methods(_panel(), {"double_realized": DoublePrice()})
    bnb = out[out["symbol"] == "BNB"]
    # One row, not ten, and not zero: a refusal is a finding and must be queryable.
    assert len(bnb) == 1
    assert not bool(bnb["applicable"].iloc[0])
    assert bnb["fair_value"].isna().all()
    assert "20.0% dense" in bnb["reason"].iloc[0]


def test_the_panel_vintage_travels_onto_every_estimate():
    """Review Focus 4: quarantine can change a velocity between runs, so an estimate must
    name the panel state it was computed from."""
    out = run_methods(_panel(("BTC",)), {"double_realized": DoublePrice()})
    assert (out["panel_vintage_ts"] == pd.Timestamp("2026-10-12T08:00:00Z")).all()


def test_circular_inputs_are_recorded_on_every_row():
    class CircularMethod(DoublePrice):
        spec = MethodSpec(
            method_id="circular_stub",
            family="A_monetary",
            requires=("realized_cap", "supply", "velocity_total"),
            min_density=0.0,
            min_rows=1,
            params={"multiple": 2.0},
            citation="test",
        )

    panel = _panel(("BTC",))
    panel["velocity_total"] = 0.05
    out = run_methods(panel, {"circular_stub": CircularMethod()})
    # realized_cap AND velocity_total are both market-cap-derived.
    assert out["circular_inputs"].iloc[0] == "realized_cap|velocity_total"


def test_the_same_params_hash_the_same_and_different_params_do_not():
    assert param_set_hash({"a": 1, "b": 2}) == param_set_hash({"b": 2, "a": 1})
    assert param_set_hash({"a": 1}) != param_set_hash({"a": 2})


def test_inputs_and_diagnostics_are_stored_as_json_strings():
    out = run_methods(_panel(("BTC",)), {"double_realized": DoublePrice()})
    # Parquet has no dict column type worth relying on, and JSON keeps the row readable
    # from SQL via DuckDB's json functions.
    parsed = json.loads(out["inputs_used"].iloc[0])
    assert parsed["supply"] == 1.9e7


def test_a_method_that_raises_produces_a_refusal_rather_than_killing_the_run():
    class Exploding(DoublePrice):
        spec = MethodSpec(
            method_id="boom",
            family="E_cost",
            requires=("realized_cap", "supply"),
            min_density=0.0,
            min_rows=1,
            params={},
            citation="test",
        )

        def estimate(self, frame, params):
            raise ZeroDivisionError("supply was zero")

    out = run_methods(_panel(("BTC",)), {"boom": Exploding()})
    assert len(out) == 1
    assert not bool(out["applicable"].iloc[0])
    assert "ZeroDivisionError" in out["reason"].iloc[0]
    assert "supply was zero" in out["reason"].iloc[0]


def test_method_specs_load_from_yaml():
    specs = load_method_specs()
    assert "realized_price" in specs
    spec = specs["realized_price"]
    assert spec.family == "E_cost"
    assert "realized_cap" in spec.requires
    assert 0.0 <= spec.min_density <= 1.0
    assert spec.citation
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptoval.methods.registry'`

- [ ] **Step 3: Implement `cryptoval/methods/registry.py`**

```python
"""Method discovery, execution and persistence (spec 10.1, 8.1).

A method whose inputs are unavailable produces ONE refusal row rather than nothing:
"this method cannot be applied to this asset, and here is the measurement that says so"
is a research finding, and a silent gap is not. Same for a method that raises -- the
exception becomes a recorded refusal so one broken method cannot void a whole run.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pandas as pd
import yaml

from cryptoval.hashing import stable_hash
from cryptoval.methods.base import Estimate, Method, MethodSpec, circularity, resolve_applicability

ESTIMATE_COLUMNS = [
    "symbol",
    "metric_date",
    "method_id",
    "param_set_hash",
    "fair_value",
    "band_low",
    "band_high",
    "applicable",
    "reason",
    "circular_inputs",
    "inputs_used",
    "diagnostics",
    "panel_vintage_ts",
]

METHODS: dict[str, Method] = {}


def register_instance(method: Method) -> Method:
    """Add an already-built method to the registry, refusing a duplicate id.

    Used where one class serves two specs -- QTM ships twice, once with an exogenous
    velocity and once with the panel's (spec 10.2).
    """
    if method.spec.method_id in METHODS:
        raise ValueError(f"method id already registered: {method.spec.method_id}")
    METHODS[method.spec.method_id] = method
    return method


def register(cls: type) -> type:
    """Class decorator: instantiate and register, returning the class unchanged.

    It registers an INSTANCE, not the class: run_methods calls method.estimate(...), which
    on a class would be an unbound call failing on the missing self. The class is returned
    untouched so tests can still construct it directly.
    """
    register_instance(cls())
    return cls


def param_set_hash(params: dict[str, Any]) -> str:
    """Stable hash of a parameter set, so a DCF at 12% and at 16% coexist as rows."""
    return stable_hash(params)


def load_method_specs(config_dir: str = "config/methods") -> dict[str, MethodSpec]:
    specs: dict[str, MethodSpec] = {}
    for path in sorted(pathlib.Path(config_dir).glob("*.yml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = MethodSpec(
            method_id=raw["method_id"],
            family=raw["family"],
            requires=tuple(raw.get("requires", []) or []),
            min_density=float(raw.get("min_density", 0.5)),
            min_rows=int(raw.get("min_rows", 30)),
            params=dict(raw.get("params", {}) or {}),
            citation=raw.get("citation", ""),
            returns_fair_value=bool(raw.get("returns_fair_value", True)),
        )
        if spec.method_id in specs:
            raise ValueError(f"duplicate method_id {spec.method_id} in {path}")
        specs[spec.method_id] = spec
    return specs


def _row(estimate: Estimate, circular: tuple[str, ...]) -> dict:
    return {
        "symbol": estimate.symbol,
        "metric_date": estimate.metric_date,
        "method_id": estimate.method_id,
        "param_set_hash": estimate.param_set_hash,
        "fair_value": estimate.fair_value,
        "band_low": estimate.band_low,
        "band_high": estimate.band_high,
        "applicable": estimate.applicable,
        "reason": estimate.reason,
        # Pipe-joined rather than a list column: Parquet list columns are awkward to
        # query from SQL and this is read far more often than it is parsed.
        "circular_inputs": "|".join(circular),
        "inputs_used": json.dumps(estimate.inputs_used, default=str, sort_keys=True),
        "diagnostics": json.dumps(estimate.diagnostics, default=str, sort_keys=True),
        "panel_vintage_ts": estimate.panel_vintage_ts,
    }


def _refusal(
    symbol: str, spec: MethodSpec, reason: str, frame: pd.DataFrame
) -> dict:
    latest = frame["metric_date"].max() if not frame.empty else None
    vintage = (
        frame["panel_vintage_ts"].max()
        if "panel_vintage_ts" in frame.columns and not frame.empty
        else None
    )
    return _row(
        Estimate(
            symbol=symbol,
            metric_date=latest,
            method_id=spec.method_id,
            param_set_hash=param_set_hash(spec.params),
            fair_value=None,
            band_low=None,
            band_high=None,
            inputs_used={},
            diagnostics={},
            applicable=False,
            reason=reason,
            panel_vintage_ts=vintage,
        ),
        circularity(spec),
    )


def run_methods(
    panel: pd.DataFrame,
    methods: dict[str, Method],
    method_ids: list[str] | None = None,
) -> pd.DataFrame:
    selected = method_ids or sorted(methods)
    rows: list[dict] = []

    for method_id in selected:
        method = methods[method_id]
        spec = method.spec
        circular = circularity(spec)
        for symbol, group in panel.groupby("symbol", sort=True):
            frame = group.sort_values("metric_date").reset_index(drop=True)
            ok, reason = resolve_applicability(spec, frame)
            if not ok:
                rows.append(_refusal(str(symbol), spec, reason, frame))
                continue
            try:
                estimates = method.estimate(frame, dict(spec.params))
            except Exception as exc:
                # One broken method must not void a whole run; the failure becomes a
                # recorded refusal naming the exception type and its message.
                rows.append(
                    _refusal(
                        str(symbol),
                        spec,
                        f"method raised {type(exc).__name__}: {exc}",
                        frame,
                    )
                )
                continue
            rows.extend(_row(e, circular) for e in estimates)

    return pd.DataFrame(rows, columns=ESTIMATE_COLUMNS)
```

- [ ] **Step 4: Create `config/methods/realized_price.yml`**

```yaml
method_id: realized_price
family: E_cost
# Realized cap divided by supply: the average on-chain acquisition price of the circulating
# supply, and the dissertation's cost-approach anchor. Realized cap is DERIVED from market
# cap via the MVRV identity (spec 6.2), so this method is flagged circular by construction
# and is a cost basis rather than independent evidence of value.
requires:
  - realized_cap
  - supply
min_density: 0.5
min_rows: 30
params: {}
citation: "Coinmetrics (2018), realized capitalisation; dissertation section 3.5"
returns_fair_value: true
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_methods_registry.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 6: Create the estimate mart and the method dimension**

`dbt/models/marts/fct_valuation_estimate.sql`:

```sql
-- Method registry output (spec 8.1). param_set_hash lets a DCF at 12% and at 16% coexist
-- as rows rather than overwrite, and panel_vintage_ts ties an estimate to the panel state
-- it was computed from.
--
-- Inapplicable rows are kept deliberately: "this method cannot be applied to this asset,
-- and here is the measurement that says so" is a research finding.
select
    symbol,
    metric_date,
    method_id,
    param_set_hash,
    fair_value,
    band_low,
    band_high,
    applicable,
    reason,
    circular_inputs,
    nullif(circular_inputs, '') is not null as is_circular,
    inputs_used,
    diagnostics,
    panel_vintage_ts
from read_parquet('{{ var("data_root") }}/gold/valuation_estimate/*.parquet')
```

`dbt/models/marts/dim_method.sql`:

```sql
-- Seeded from config/methods/, which makes "which methods disagree, and why" a SQL
-- question and is the source of the method cards (spec 8.1).
select
    method_id,
    family,
    requires,
    min_density,
    min_rows,
    citation,
    returns_fair_value
from {{ ref('methods') }}
```

Add a `methods` seed generator to `cryptoval/seeds.py` (`write_method_seed(specs, seed_dir)`,
columns `method_id, family, requires, min_density, min_rows, citation, returns_fair_value`, with
`requires` pipe-joined) and extend the `make seed` target to write it. Add a test to
`tests/test_seeds.py` mirroring `test_the_seeds_are_deterministic` for the new file.

- [ ] **Step 7: Run the full suite and lint, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`

```bash
git add cryptoval/methods/registry.py cryptoval/seeds.py config/methods \
        tests/test_methods_registry.py tests/test_seeds.py dbt/models/marts dbt/seeds Makefile
git commit -m "$(cat <<'MSG'
feat: the method registry, its parameter sets and the estimate fact

Methods are discovered from config/methods/*.yml and run per asset over
the panel slice. A method whose inputs are unavailable produces ONE
refusal row carrying the measurement that refused it, because a silent
gap is not a finding and a refusal is. A method that raises becomes the
same kind of recorded refusal, so one broken method cannot void a run.

Every row carries param_set_hash, so a DCF at 12% and at 16% coexist
rather than overwrite, and panel_vintage_ts, so an estimate stays tied to
the panel state it was computed from now that quarantine can change a
velocity between runs.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
## Task 12: Family E(a) — the cost approach

Spec §10.6. Five methods, four of them trivial arithmetic over columns Task 8 already derived, plus
Hayes, which is the only one with real unit risk. Hash rate exists on the free tier for exactly BTC,
LTC, BCH and DOGE plus ETH to the Merge (63.7% dense), and `consensus: pow` in `assets.yml` predicts
that set exactly — so Hayes needs no asset allow-list, only its density floor.

**Files:**
- Create: `cryptoval/methods/cost.py`
- Create: `tests/test_methods_cost.py`
- Create: `config/methods/{thermocap_floor,mvrv_reversion,mvtv_reversion,investor_cap_price,hayes_cost_of_production}.yml`

**Interfaces:**
- Consumes: `Estimate`, `MethodSpec` from Task 10; `register`, `param_set_hash` from Task 11.
- Produces five registered methods, plus the helpers `per_unit(series, supply)` and
  `reversion_to_multiple(level_usd, multiple_series, supply, window)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_methods_cost.py`:

```python
import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.cost import (
    HayesCostOfProduction,
    RealizedPrice,
    ThermocapFloor,
    hayes_cost_per_coin,
)
from cryptoval.methods.registry import param_set_hash


def _pow_frame(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTC"] * n,
            "metric_date": pd.date_range("2025-01-01", periods=n, freq="D").date,
            "realized_cap": np.linspace(1.0e12, 1.08e12, n),
            "thermocap_usd": np.linspace(4.0e11, 4.2e11, n),
            "investor_cap_usd": np.linspace(6.0e11, 6.6e11, n),
            "market_cap_usd": np.linspace(1.5e12, 1.7e12, n),
            "mvrv": np.linspace(1.5, 1.6, n),
            "mvtv": np.linspace(3.7, 4.0, n),
            "supply": np.linspace(1.95e7, 1.96e7, n),
            "price_usd": np.linspace(77000, 87000, n),
            # ~900 TH/s... in Coin Metrics' own unit, which Step 5 pins empirically.
            "hash_rate": np.full(n, 9.0e8),
            "issuance_usd": np.full(n, 4.0e7),
            "thermocap_has_gap": False,
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )


def test_realized_price_is_realized_cap_per_unit():
    frame = _pow_frame()
    out = RealizedPrice().estimate(frame, {})
    assert len(out) == len(frame)
    expected = frame["realized_cap"].iloc[-1] / frame["supply"].iloc[-1]
    assert out[-1].fair_value == pytest.approx(expected)
    assert out[-1].inputs_used["realized_cap"] == pytest.approx(frame["realized_cap"].iloc[-1])


def test_the_thermocap_floor_flags_a_gap_as_a_lower_bound():
    frame = _pow_frame()
    frame.loc[10:, "thermocap_has_gap"] = True
    out = ThermocapFloor().estimate(frame, {})
    last = out[-1]
    # A gapped thermocap understates the cumulative cost, so the figure is a LOWER bound
    # and anything derived from it (MVTV, investor cap) is biased UPWARD.
    assert last.diagnostics["is_lower_bound"] is True
    assert "understate" in last.diagnostics["bias_direction"]
    assert last.band_high is None
    assert last.band_low == pytest.approx(last.fair_value)


def test_a_zero_supply_day_yields_no_estimate_rather_than_infinity():
    frame = _pow_frame(n=40)
    frame.loc[5, "supply"] = 0.0
    out = RealizedPrice().estimate(frame, {})
    at_zero = [e for e in out if e.metric_date == frame["metric_date"].iloc[5]][0]
    assert at_zero.fair_value is None
    assert not at_zero.applicable
    assert "supply" in at_zero.reason


def test_hayes_cost_per_coin_is_energy_cost_divided_by_coins_mined():
    # 1e9 hashes/s at 1e-10 J per hash = 0.1 W... deliberately tiny, so the arithmetic is
    # checkable by hand rather than by reimplementing the function.
    cost = hayes_cost_per_coin(
        hash_rate=1.0e9,
        hashes_per_unit=1.0,
        joules_per_hash=1.0e-10,
        electricity_usd_per_kwh=0.05,
        coins_per_day=2.0,
    )
    joules_per_day = 1.0e9 * 1.0 * 1.0e-10 * 86400
    kwh = joules_per_day / 3.6e6
    assert cost == pytest.approx(kwh * 0.05 / 2.0)


def test_hayes_returns_none_when_no_coins_were_mined():
    assert hayes_cost_per_coin(1.0e9, 1.0, 1.0e-10, 0.05, 0.0) is None
    assert hayes_cost_per_coin(1.0e9, 1.0, 1.0e-10, 0.05, float("nan")) is None


def test_hayes_is_inapplicable_without_hash_rate():
    frame = _pow_frame()
    frame["hash_rate"] = np.nan
    method = HayesCostOfProduction()
    from cryptoval.methods.base import resolve_applicability

    ok, reason = resolve_applicability(method.spec, frame)
    assert not ok
    assert "hash_rate" in reason


def test_hayes_records_both_parameters_in_inputs_used():
    out = HayesCostOfProduction().estimate(
        _pow_frame(),
        {
            "joules_per_hash": 2.0e-11,
            "electricity_usd_per_kwh": 0.05,
            "hashes_per_unit": 1.0e3,
        },
    )
    used = out[-1].inputs_used
    # A cost of production is only as meaningful as its two assumptions, so they travel
    # with every row rather than living only in the method card.
    assert used["electricity_usd_per_kwh"] == 0.05
    assert used["joules_per_hash"] == 2.0e-11


def test_param_sets_distinguish_two_electricity_assumptions():
    cheap = param_set_hash({"electricity_usd_per_kwh": 0.03, "joules_per_hash": 2.0e-11})
    dear = param_set_hash({"electricity_usd_per_kwh": 0.08, "joules_per_hash": 2.0e-11})
    assert cheap != dear
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_cost.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptoval.methods.cost'`

- [ ] **Step 3: Write the five method configs FIRST**

`cost.py` calls `load_method_specs()` at import time and indexes the result by `method_id`, so
importing it before its YAML exists raises a bare `KeyError`. Write Step 5's configs now, then
return here. The same ordering applies in Tasks 13, 14 and 15.

- [ ] **Step 4: Implement `cryptoval/methods/cost.py`**

```python
"""Family E(a) -- the cost approach (spec 10.6).

Four of these are arithmetic over columns the mart already derives. All four inherit the
MVRV identity's fragility (spec 6.2): realized cap is market_cap / mvrv, so every method
here is flagged circular and is a COST BASIS rather than independent evidence of value.
The method cards say so; this module makes sure the flag is never absent.

Hayes is the exception: it is genuinely independent of market cap, and genuinely exposed
to a unit error in Coin Metrics' HashRate series, which is why Step 6 of its task
calibrates it against observed price before it ships.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def _finite(value: Any) -> float | None:
    """None for anything that cannot be a price: NaN, inf, or a non-number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _per_unit_estimates(
    frame: pd.DataFrame,
    spec: MethodSpec,
    level_column: str,
    params: dict,
    diagnostics_for=None,
) -> list[Estimate]:
    """One estimate per row: ``level_column`` divided by supply.

    A row whose supply is missing, zero or negative yields an INAPPLICABLE estimate rather
    than an infinity -- the asset is fine, that day is not.
    """
    out: list[Estimate] = []
    phash = param_set_hash(params)
    for row in frame.itertuples(index=False):
        level = _finite(getattr(row, level_column))
        supply = _finite(row.supply)
        vintage = getattr(row, "panel_vintage_ts", None)
        if supply is None or supply <= 0 or level is None:
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=spec.method_id,
                    param_set_hash=phash,
                    fair_value=None,
                    band_low=None,
                    band_high=None,
                    inputs_used={},
                    diagnostics={},
                    applicable=False,
                    reason=(
                        f"supply is {supply!r} on this day"
                        if supply is None or supply <= 0
                        else f"{level_column} is missing on this day"
                    ),
                    panel_vintage_ts=vintage,
                )
            )
            continue
        diagnostics = diagnostics_for(row) if diagnostics_for else {}
        fair_value = level / supply
        lower_bound = bool(diagnostics.get("is_lower_bound"))
        out.append(
            Estimate(
                symbol=row.symbol,
                metric_date=row.metric_date,
                method_id=spec.method_id,
                param_set_hash=phash,
                fair_value=fair_value,
                # A lower bound is a one-sided band, not a point with no uncertainty.
                band_low=fair_value if lower_bound else None,
                band_high=None,
                inputs_used={level_column: level, "supply": supply},
                diagnostics=diagnostics,
                applicable=True,
                panel_vintage_ts=vintage,
            )
        )
    return out


@register
class RealizedPrice:
    spec = _SPECS["realized_price"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _per_unit_estimates(frame, self.spec, "realized_cap", params)


@register
class ThermocapFloor:
    spec = _SPECS["thermocap_floor"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        def diagnostics(row) -> dict:
            gapped = bool(getattr(row, "thermocap_has_gap", False))
            return {
                "is_lower_bound": gapped,
                # Direction matters and is easy to state backwards: a gap means some
                # miner revenue was never added, so thermocap UNDERSTATES cumulative
                # cost, and MVTV and investor cap built on it are biased UPWARD.
                "bias_direction": (
                    "a gap makes thermocap understate cumulative cost, so this floor is "
                    "too low and MVTV/investor-cap built on it read too high"
                    if gapped
                    else "no gap"
                ),
            }

        return _per_unit_estimates(frame, self.spec, "thermocap_usd", params, diagnostics)


@register
class InvestorCapPrice:
    spec = _SPECS["investor_cap_price"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _per_unit_estimates(frame, self.spec, "investor_cap_usd", params)


def _reversion_estimates(
    frame: pd.DataFrame,
    spec: MethodSpec,
    level_column: str,
    multiple_column: str,
    params: dict,
) -> list[Estimate]:
    """Value the asset at its own historical multiple of a cost level.

    The multiple is a TRAILING median -- expanding, so no future information enters a
    past row (the no-lookahead rule that Plan 4's backtests depend on).
    """
    out: list[Estimate] = []
    phash = param_set_hash(params)
    minimum = int(params.get("min_history_rows", 180))
    multiples = pd.to_numeric(frame[multiple_column], errors="coerce")
    trailing_median = multiples.shift(1).expanding(min_periods=minimum).median()

    for position, row in enumerate(frame.itertuples(index=False)):
        level = _finite(getattr(row, level_column))
        supply = _finite(row.supply)
        target = _finite(trailing_median.iloc[position])
        vintage = getattr(row, "panel_vintage_ts", None)
        if None in (level, supply, target) or supply <= 0 or target <= 0:
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=spec.method_id,
                    param_set_hash=phash,
                    fair_value=None,
                    band_low=None,
                    band_high=None,
                    inputs_used={},
                    diagnostics={},
                    applicable=False,
                    reason=(
                        f"fewer than {minimum} prior observations of {multiple_column}"
                        if target is None
                        else "missing level or supply on this day"
                    ),
                    panel_vintage_ts=vintage,
                )
            )
            continue
        observed = _finite(getattr(row, multiple_column))
        out.append(
            Estimate(
                symbol=row.symbol,
                metric_date=row.metric_date,
                method_id=spec.method_id,
                param_set_hash=phash,
                fair_value=level * target / supply,
                band_low=None,
                band_high=None,
                inputs_used={
                    level_column: level,
                    "supply": supply,
                    f"trailing_median_{multiple_column}": target,
                },
                diagnostics={
                    f"observed_{multiple_column}": observed,
                    "deviation_from_trailing_median": (
                        observed / target if observed and target else None
                    ),
                },
                applicable=True,
                panel_vintage_ts=vintage,
            )
        )
    return out


@register
class MvrvReversion:
    spec = _SPECS["mvrv_reversion"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _reversion_estimates(frame, self.spec, "realized_cap", "mvrv", params)


@register
class MvtvReversion:
    spec = _SPECS["mvtv_reversion"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        return _reversion_estimates(frame, self.spec, "thermocap_usd", "mvtv", params)


def hayes_cost_per_coin(
    hash_rate: float,
    hashes_per_unit: float,
    joules_per_hash: float,
    electricity_usd_per_kwh: float,
    coins_per_day: float,
) -> float | None:
    """Marginal electricity cost of producing one coin in a day (Hayes 2017).

    energy = hash_rate x hashes_per_unit x joules_per_hash x 86400 seconds
    cost   = energy / 3.6e6 J-per-kWh x price-per-kWh
    result = cost / coins mined that day

    ``hashes_per_unit`` converts Coin Metrics' HashRate unit into hashes per second. It is
    a PARAMETER rather than a constant because the unit differs by asset and getting it
    wrong scales the answer by orders of magnitude without changing its shape -- which is
    exactly the error a plausible-looking chart hides. Step 6 of this task calibrates it.
    """
    coins = _finite(coins_per_day)
    rate = _finite(hash_rate)
    if coins is None or coins <= 0 or rate is None or rate <= 0:
        return None
    joules = rate * hashes_per_unit * joules_per_hash * 86_400
    kwh = joules / 3.6e6
    return kwh * electricity_usd_per_kwh / coins


@register
class HayesCostOfProduction:
    spec = _SPECS["hayes_cost_of_production"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        out: list[Estimate] = []
        phash = param_set_hash(params)
        joules = float(params["joules_per_hash"])
        price_kwh = float(params["electricity_usd_per_kwh"])
        hashes_per_unit = float(params["hashes_per_unit"])

        for row in frame.itertuples(index=False):
            vintage = getattr(row, "panel_vintage_ts", None)
            issuance = _finite(row.issuance_usd)
            price = _finite(row.price_usd)
            coins = issuance / price if issuance is not None and price not in (None, 0) else None
            cost = (
                hayes_cost_per_coin(row.hash_rate, hashes_per_unit, joules, price_kwh, coins)
                if coins is not None
                else None
            )
            if cost is None:
                out.append(
                    Estimate(
                        symbol=row.symbol,
                        metric_date=row.metric_date,
                        method_id=self.spec.method_id,
                        param_set_hash=phash,
                        fair_value=None,
                        band_low=None,
                        band_high=None,
                        inputs_used={},
                        diagnostics={},
                        applicable=False,
                        reason="no hash rate or no coins issued on this day",
                        panel_vintage_ts=vintage,
                    )
                )
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=cost,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "hash_rate": _finite(row.hash_rate),
                        "coins_per_day": coins,
                        "joules_per_hash": joules,
                        "electricity_usd_per_kwh": price_kwh,
                        "hashes_per_unit": hashes_per_unit,
                    },
                    diagnostics={
                        # Hayes is a FLOOR argument, not a fair value: miners exit below
                        # cost, which constrains price from beneath.
                        "interpretation": "marginal production cost, a floor rather than a target",
                        "price_to_cost": (price / cost) if price and cost else None,
                    },
                    applicable=True,
                    panel_vintage_ts=vintage,
                )
            )
        return out
```

- [ ] **Step 5: The five method configs** (written in Step 3)

Each follows `realized_price.yml`'s shape. The values that matter:

```yaml
# thermocap_floor.yml
method_id: thermocap_floor
family: E_cost
requires: [thermocap_usd, supply]
min_density: 0.5
min_rows: 30
params: {}
citation: "Nick Carter / Coinmetrics thermocap; dissertation section 3.5"

# investor_cap_price.yml
method_id: investor_cap_price
family: E_cost
requires: [investor_cap_usd, supply]
min_density: 0.5
min_rows: 30
params: {}
citation: "dissertation section 3.5, investor capitalisation"

# mvrv_reversion.yml
method_id: mvrv_reversion
family: E_cost
requires: [realized_cap, mvrv, supply]
min_density: 0.5
min_rows: 365
params: {min_history_rows: 180}
citation: "Murad Mahmudov & David Puell (2018), MVRV"

# mvtv_reversion.yml
method_id: mvtv_reversion
family: E_cost
requires: [thermocap_usd, mvtv, supply]
min_density: 0.5
min_rows: 365
params: {min_history_rows: 180}
citation: "dissertation section 3.5, market-value-to-thermocap"

# hayes_cost_of_production.yml
method_id: hayes_cost_of_production
family: E_cost
# No asset allow-list is needed: HashRate exists on the free tier for exactly the four
# proof-of-work assets plus ETH to the Merge, so the density floor selects them (spec 10.1).
requires: [hash_rate, issuance_usd, price_usd]
min_density: 0.9
min_rows: 180
params:
  # Calibrated in Step 5 against BTC. Both are assumptions, not measurements, and they
  # travel onto every estimate row because the answer is only as good as they are.
  joules_per_hash: 2.0e-11
  electricity_usd_per_kwh: 0.05
  hashes_per_unit: 1.0e12
citation: "Hayes (2017), cost of production model; dissertation section 3.4"
```

- [ ] **Step 6: Calibrate the Hayes unit against live data — do not skip this**

Coin Metrics' `HashRate` unit is not documented in the response, and a unit error scales the
answer by orders of magnitude while leaving the curve's shape plausible. Measure it:

```bash
.venv/bin/python -I -c "
import duckdb
from cryptoval.methods.cost import hayes_cost_per_coin
con = duckdb.connect('data/cryptoval.duckdb', read_only=True)
rows = con.execute('''
  select metric_date, hash_rate, issuance_usd, price_usd
  from fct_asset_metrics_daily
  where symbol='BTC' and hash_rate is not null and issuance_usd is not null
    and price_usd is not null and metric_date >= date '2026-01-01'
  order by metric_date desc limit 1''').fetchone()
date, hr, iss, px = rows
for unit, label in [(1.0,'H/s'), (1e3,'kH/s'), (1e6,'MH/s'), (1e9,'GH/s'), (1e12,'TH/s'), (1e15,'PH/s'), (1e18,'EH/s')]:
    cost = hayes_cost_per_coin(hr, unit, 2.0e-11, 0.05, iss/px)
    print(f'{label:5s} cost={cost:>18,.2f}  price={px:>12,.2f}  ratio={px/cost if cost else float(\"nan\"):>10,.3f}')
print('raw hash_rate =', hr, 'on', date)
"
```

Pick the unit whose implied cost lands within an order of magnitude of price — Hayes's own
finding is that price and production cost track each other closely, so a ratio between roughly
0.1 and 10 is the only defensible choice. Record the measured table in the task report and set
`hashes_per_unit` in the config to the chosen value, with a comment naming the measured ratio.

Then pin it with a test in `tests/test_methods_cost.py`:

```python
@pytest.mark.integration
def test_the_hayes_unit_puts_btc_cost_within_an_order_of_magnitude_of_price():
    """A unit error scales Hayes by 1000x while leaving the curve's shape plausible."""
    import duckdb

    from cryptoval.methods.registry import load_method_specs

    params = load_method_specs()["hayes_cost_of_production"].params
    con = duckdb.connect("data/cryptoval.duckdb", read_only=True)
    hr, iss, px = con.execute(
        """
        select hash_rate, issuance_usd, price_usd from fct_asset_metrics_daily
        where symbol = 'BTC' and hash_rate is not null and issuance_usd is not null
          and price_usd is not null
        order by metric_date desc limit 1
        """
    ).fetchone()
    cost = hayes_cost_per_coin(
        hr,
        params["hashes_per_unit"],
        params["joules_per_hash"],
        params["electricity_usd_per_kwh"],
        iss / px,
    )
    assert 0.1 <= px / cost <= 10.0, f"implied cost {cost:,.2f} against price {px:,.2f}"
```

- [ ] **Step 7: Run the tests, the full suite and lint, then commit**

Run: `.venv/bin/python -m pytest tests/test_methods_cost.py -v` then
`.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`

```bash
git add cryptoval/methods/cost.py config/methods tests/test_methods_cost.py
git commit -m "$(cat <<'MSG'
feat: Family E(a), the cost approach

Realized price, the thermocap floor, investor-cap price, and MVRV/MVTV
reversion to a TRAILING median -- expanding, so no future information
enters a past row, which is what Plan 4's backtests depend on.

Four of the five inherit the MVRV identity's circularity and are cost
bases rather than independent evidence; the registry flags them and the
cards say so. A missing or zero supply yields an inapplicable estimate for
that day rather than an infinity.

Hayes is genuinely independent of market cap and genuinely exposed to a
unit error in Coin Metrics' HashRate series, so hashes_per_unit is a
calibrated parameter and an integration test pins BTC's implied cost to
within an order of magnitude of price.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
## Task 13: Family A — monetary, and the holding-time correction that invalidates its inversion

Spec §10.2. The holding-time correction is the reason this family sits in Phase 2b rather than 3a:
its finding is that mean holding time and `1/V` differ at the mean, which invalidates the inversion
the dissertation's own QTM and P/U sections rely on. Reproducing those without it would reproduce a
known error and call it a baseline.

**A deviation from the spec's letter, deliberately.** §10.2 says to "derive mean holding time by
simulation". For a log-normal velocity there is an exact closed form instead: if
`V ~ logN(mu, sigma)` then `E[1/V] = exp(-mu + sigma^2/2)` while the naive inversion gives
`1/E[V] = exp(-mu - sigma^2/2)`, so the two differ by exactly **`exp(sigma^2)`**. We implement the
closed form, which is exact and reproducible, and use Monte Carlo in the *test* to verify it. A
simulation in production would add sampling noise to a quantity we can state precisely.

**Files:**
- Create: `cryptoval/methods/monetary.py`, `tests/test_methods_monetary.py`
- Create: `config/methods/{holding_time_correction,crypto_qtm,crypto_qtm_panel_velocity,j_curve}.yml`
- Modify: `requirements.txt` (add `statsmodels==0.15.0`)

**Interfaces:**
- Consumes: Task 10's contract, Task 11's `register`, `register_instance`, `param_set_hash`.
- Produces: `fit_lognormal_velocity(series) -> dict`, `holding_time_ratio(sigma) -> float`, and four
  registered methods.

- [ ] **Step 1: Pin statsmodels — resolve the version, do not trust this plan's guess**

Every other pin in this project was verified to resolve before it was written down, and this one was
not. Find the version that actually installs against Python 3.14.5 and pandas 3.0.6:

```bash
.venv/bin/pip index versions statsmodels
```

Install the newest that resolves, then pin **that exact version** in `requirements.txt` and confirm:

```bash
.venv/bin/python -c "import statsmodels, numpy, pandas; print(statsmodels.__version__)"
.venv/bin/python -c "from statsmodels.stats.diagnostic import normal_ad; from statsmodels.stats.stattools import durbin_watson; import statsmodels.api as sm; print('imports ok')"
```

Both imports must succeed — Tasks 13 and 14 use `normal_ad`, `durbin_watson` and `sm.OLS`. If
statsmodels does not resolve at all on this Python, stop and report: the alternative is hand-rolling
OLS with confidence intervals and the Durbin-Watson statistic over numpy, which is a different task
with a different test surface, and that is a decision rather than an implementation detail.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_methods_monetary.py`:

```python
import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.monetary import (
    HoldingTimeCorrection,
    JCurve,
    fit_lognormal_velocity,
    holding_time_ratio,
)
from cryptoval.methods.registry import METHODS


def _frame(n: int = 500, sigma: float = 0.6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    velocity = np.exp(rng.normal(np.log(0.04), sigma, n))
    market_cap = np.full(n, 1.0e12)
    return pd.DataFrame(
        {
            "symbol": ["ETH"] * n,
            "metric_date": pd.date_range("2025-01-01", periods=n, freq="D").date,
            "market_cap_usd": market_cap,
            "volume_total_usd": velocity * market_cap,
            "velocity_total": velocity,
            "supply": np.full(n, 1.2e8),
            "price_usd": market_cap / 1.2e8,
            "tx_count": np.full(n, 1.2e6),
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )


def test_the_two_holding_times_differ_by_exp_sigma_squared():
    """Spec 10.2's central result, stated exactly rather than simulated."""
    fit = fit_lognormal_velocity(_frame(n=20000, sigma=0.6)["velocity_total"])
    assert fit["sigma"] == pytest.approx(0.6, rel=0.05)
    assert fit["mean_holding_time"] / fit["naive_inverse_velocity"] == pytest.approx(
        np.exp(fit["sigma"] ** 2), rel=1e-9
    )
    assert holding_time_ratio(0.6) == pytest.approx(np.exp(0.36))


def test_the_closed_form_matches_monte_carlo():
    """Simulation used as verification, which is what it is good for."""
    rng = np.random.default_rng(11)
    mu, sigma, n = np.log(0.04), 0.5, 2_000_000
    draws = np.exp(rng.normal(mu, sigma, n))
    simulated_mean_holding = float(np.mean(1.0 / draws))
    closed_form = float(np.exp(-mu + sigma**2 / 2))
    assert simulated_mean_holding == pytest.approx(closed_form, rel=0.01)


def test_a_zero_sigma_collapses_the_correction_to_one():
    # With no dispersion the inversion is exact, which is the degenerate case the
    # dissertation implicitly assumed.
    assert holding_time_ratio(0.0) == pytest.approx(1.0)


def test_the_correction_returns_no_fair_value_but_records_the_diagnosis():
    out = HoldingTimeCorrection().estimate(_frame(), {"window_rows": 365})
    last = out[-1]
    assert last.fair_value is None
    assert last.applicable  # applicable, but it is a correction, not a valuation
    assert last.diagnostics["holding_time_ratio"] > 1.0
    assert "anderson_darling_p" in last.diagnostics


def test_a_degenerate_velocity_series_is_refused_rather_than_fitted():
    frame = _frame(n=400)
    frame["velocity_total"] = 0.0
    out = HoldingTimeCorrection().estimate(frame, {"window_rows": 365})
    assert not out[-1].applicable
    assert "positive" in out[-1].reason


def test_qtm_with_panel_velocity_returns_the_price_back():
    """Review Focus 1, demonstrated numerically: the expression collapses."""
    frame = _frame(n=400)
    method = METHODS["crypto_qtm_panel_velocity"]
    out = method.estimate(frame, {"circulating_fraction": 1.0, "velocity_source": "panel"})
    applicable = [e for e in out if e.applicable]
    prices = frame["price_usd"].to_numpy()
    values = np.array([e.fair_value for e in applicable])
    # fair_value == price / f exactly, so every residual is zero by construction.
    assert values == pytest.approx(prices[: len(values)], rel=1e-9)
    assert applicable[0].diagnostics["collapses_to_price"] is True


def test_qtm_with_a_parameter_velocity_is_a_genuine_sensitivity():
    frame = _frame(n=400)
    method = METHODS["crypto_qtm"]
    slow = method.estimate(
        frame, {"circulating_fraction": 1.0, "velocity_source": "parameter", "velocity_daily": 0.02}
    )
    fast = method.estimate(
        frame, {"circulating_fraction": 1.0, "velocity_source": "parameter", "velocity_daily": 0.08}
    )
    # Halving assumed velocity doubles the implied price; this is the one usable reading
    # of QTM on free data (the on-chain transfer value it really wants is not free).
    assert slow[-1].fair_value == pytest.approx(4.0 * fast[-1].fair_value, rel=1e-9)
    assert slow[-1].diagnostics["collapses_to_price"] is False


def test_the_j_curve_splits_price_into_utility_and_expectation():
    out = JCurve().estimate(
        _frame(n=400), {"velocity_source": "parameter", "velocity_daily": 0.04}
    )
    last = out[-1]
    # CUV is the fair value; DEUV is the residual the market pays above it.
    assert last.fair_value is not None
    assert last.diagnostics["deuv"] == pytest.approx(
        last.inputs_used["price_usd"] - last.fair_value
    )
    assert -1.0 <= last.diagnostics["deuv_share"] <= 1.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_monetary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptoval.methods.monetary'`

- [ ] **Step 4: Write the four method configs FIRST**

As in Task 12, `monetary.py` indexes `load_method_specs()` at import time, so Step 6's YAML must
exist before the module can be imported at all. Write it now, then return here.

- [ ] **Step 5: Implement `cryptoval/methods/monetary.py`**

```python
"""Family A -- monetary / equation of exchange (spec 10.2).

The holding-time correction is the load-bearing piece. If velocity is log-normal, mean
holding time is E[1/V] = exp(-mu + sigma^2/2) while the naive inversion 1/E[V] gives
exp(-mu - sigma^2/2): the two differ by exactly exp(sigma^2), and the dissertation's QTM
and Price-to-Utility sections use the inversion. On a realistic sigma of 0.6 that is a 43%
error in the denominator, in a known direction.

The spec says to derive the mean by simulation; the closed form above is exact, so it is
used here and Monte Carlo verifies it in the tests instead. Adding sampling noise to a
quantity we can state precisely would be a strange choice.

On QTM's circularity: the equation wants Q as on-chain transaction VALUE, which the free
tier does not serve. Using exchange volume makes V = volume/market_cap, and the whole
expression collapses to price/f -- it returns the market price back. Rather than hide
that, both readings ship: crypto_qtm takes velocity as an exogenous assumption and is a
genuine sensitivity, and crypto_qtm_panel_velocity uses the panel and is flagged circular
with collapses_to_price in its diagnostics. Where they differ IS the finding (spec 14).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import normal_ad

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.registry import (
    load_method_specs,
    param_set_hash,
    register,
    register_instance,
)

_SPECS = load_method_specs()


def holding_time_ratio(sigma: float) -> float:
    """E[1/V] / (1/E[V]) for a log-normal V. Exactly exp(sigma^2)."""
    return float(math.exp(sigma**2))


def fit_lognormal_velocity(series: pd.Series) -> dict:
    """Fit log(V) ~ Normal and return both holding-time constructions.

    Only strictly positive velocities enter the fit: a zero or negative velocity is not a
    draw from a log-normal, it is a day with no measurable turnover.
    """
    values = pd.to_numeric(series, errors="coerce")
    positive = values[values > 0].dropna()
    if len(positive) < 30:
        raise ValueError(
            f"only {len(positive)} strictly positive velocity observations; "
            "need at least 30 to fit a distribution"
        )
    logs = np.log(positive.to_numpy())
    mu = float(logs.mean())
    sigma = float(logs.std(ddof=1))
    try:
        _, ad_p = normal_ad(logs)
    except Exception:
        ad_p = float("nan")
    return {
        "mu": mu,
        "sigma": sigma,
        "n": int(len(positive)),
        "mean_velocity": float(math.exp(mu + sigma**2 / 2)),
        "mean_holding_time": float(math.exp(-mu + sigma**2 / 2)),
        "naive_inverse_velocity": float(math.exp(-mu - sigma**2 / 2)),
        "holding_time_ratio": holding_time_ratio(sigma),
        # p < 0.05 rejects log-normality, which does not invalidate the ratio but does
        # mean the closed form is an approximation for this asset. Reported, not hidden.
        "anderson_darling_p": float(ad_p),
    }


def _refuse(spec: MethodSpec, row, params: dict, reason: str) -> Estimate:
    return Estimate(
        symbol=row.symbol,
        metric_date=row.metric_date,
        method_id=spec.method_id,
        param_set_hash=param_set_hash(params),
        fair_value=None,
        band_low=None,
        band_high=None,
        inputs_used={},
        diagnostics={},
        applicable=False,
        reason=reason,
        panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
    )


@register
class HoldingTimeCorrection:
    spec = _SPECS["holding_time_correction"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        out: list[Estimate] = []
        phash = param_set_hash(params)
        for position, row in enumerate(frame.itertuples(index=False)):
            # Trailing window only: a fit that saw future velocity would leak into
            # Plan 4's backtests.
            history = frame["velocity_total"].iloc[max(0, position - window + 1) : position + 1]
            try:
                fit = fit_lognormal_velocity(history)
            except ValueError as exc:
                out.append(_refuse(self.spec, row, params, f"no positive velocity: {exc}"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    # A correction, not a valuation: returns_fair_value is false in config.
                    fair_value=None,
                    band_low=None,
                    band_high=None,
                    inputs_used={"window_rows": window, "observations": fit["n"]},
                    diagnostics=fit,
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


def _velocity_for(row, params: dict) -> tuple[float | None, bool]:
    """Resolve the velocity term. Returns (value, collapses_to_price)."""
    if params.get("velocity_source") == "panel":
        value = getattr(row, "velocity_total", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None, True
        return (value if math.isfinite(value) and value > 0 else None), True
    value = float(params["velocity_daily"])
    return (value if value > 0 else None), False


class CryptoQTM:
    """price = Q / (M* x f x V), dissertation section 3.1."""

    def __init__(self, spec: MethodSpec) -> None:
        self.spec = spec

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        fraction = float(params.get("circulating_fraction", 1.0))
        phash = param_set_hash(params)
        out: list[Estimate] = []
        for row in frame.itertuples(index=False):
            velocity, collapses = _velocity_for(row, params)
            quantity = pd.to_numeric(pd.Series([row.volume_total_usd]), errors="coerce").iloc[0]
            supply = float(row.supply) if row.supply else None
            if velocity is None or supply is None or supply <= 0 or not np.isfinite(quantity):
                out.append(
                    _refuse(self.spec, row, params, "missing volume, supply or velocity")
                )
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=float(quantity) / (supply * fraction * velocity),
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "volume_total_usd": float(quantity),
                        "supply": supply,
                        "circulating_fraction": fraction,
                        "velocity": velocity,
                        "velocity_source": params.get("velocity_source", "parameter"),
                        "price_usd": float(row.price_usd) if row.price_usd else None,
                    },
                    diagnostics={
                        # True means this row IS the market price divided by f, so its
                        # residual against price is zero by construction.
                        "collapses_to_price": collapses,
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


register_instance(CryptoQTM(_SPECS["crypto_qtm"]))
register_instance(CryptoQTM(_SPECS["crypto_qtm_panel_velocity"]))


@register
class JCurve:
    """Burniske's CUV/DEUV split.

    CUV = network transaction value / (circulating supply x velocity); DEUV = price - CUV.
    The dissertation renders CUV as "token's current supply / (token supply x velocity)",
    which is dimensionally incoherent as printed -- it yields a ratio, not a price. We
    implement the economically intended form and the method card records the deviation
    (spec 10.2).
    """

    spec = _SPECS["j_curve"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        phash = param_set_hash(params)
        out: list[Estimate] = []
        for row in frame.itertuples(index=False):
            velocity, collapses = _velocity_for(row, params)
            supply = float(row.supply) if row.supply else None
            price = float(row.price_usd) if row.price_usd else None
            quantity = float(row.volume_total_usd) if row.volume_total_usd else None
            if None in (velocity, supply, price, quantity) or supply <= 0:
                out.append(_refuse(self.spec, row, params, "missing volume, supply, price or velocity"))
                continue
            cuv = quantity / (supply * velocity)
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=cuv,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "volume_total_usd": quantity,
                        "supply": supply,
                        "velocity": velocity,
                        "price_usd": price,
                    },
                    diagnostics={
                        "deuv": price - cuv,
                        "deuv_share": (price - cuv) / price if price else None,
                        "collapses_to_price": collapses,
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
```

- [ ] **Step 6: The four method configs** (written in Step 4)

```yaml
# holding_time_correction.yml
method_id: holding_time_correction
family: A_monetary
requires: [velocity_total]
min_density: 0.6
min_rows: 365
params: {window_rows: 365}
returns_fair_value: false
citation: "Kampakis et al. (2024); corrects dissertation sections 3.1 and 3.6"

# crypto_qtm.yml -- velocity as an exogenous assumption, so NOT circular
method_id: crypto_qtm
family: A_monetary
requires: [volume_total_usd, supply, price_usd]
min_density: 0.6
min_rows: 180
params:
  velocity_source: parameter
  velocity_daily: 0.04        # ~14.6 turns a year; vary it, that is the point
  circulating_fraction: 1.0   # SplyCur is already the circulating supply
citation: "dissertation section 3.1, crypto quantity theory of money"

# crypto_qtm_panel_velocity.yml -- the dissertation's own reading, circular by construction
method_id: crypto_qtm_panel_velocity
family: A_monetary
requires: [volume_total_usd, supply, price_usd, velocity_total]
min_density: 0.6
min_rows: 180
params:
  velocity_source: panel
  circulating_fraction: 1.0
citation: "dissertation section 3.1 as literally specified; retained to measure the circularity"

# j_curve.yml
method_id: j_curve
family: A_monetary
requires: [volume_total_usd, supply, price_usd]
min_density: 0.6
min_rows: 180
params:
  velocity_source: parameter
  velocity_daily: 0.04
citation: "Burniske (2017), crypto J-curve; dissertation section 3.2"
```

- [ ] **Step 7: Run the tests, full suite and lint, then commit**

Run: `.venv/bin/python -m pytest tests/test_methods_monetary.py -v` then
`.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`

```bash
git add cryptoval/methods/monetary.py config/methods tests/test_methods_monetary.py requirements.txt
git commit -m "$(cat <<'MSG'
feat: Family A, and the holding-time correction its inversion needs

Mean holding time is E[1/V] = exp(-mu + sigma^2/2); the naive inversion
1/E[V] gives exp(-mu - sigma^2/2). The two differ by exactly exp(sigma^2),
which on a realistic sigma of 0.6 is a 43% error in the denominator the
dissertation's QTM and P/U sections rely on.

The spec asked for simulation; the closed form is exact, so it ships and
Monte Carlo verifies it in the tests instead.

QTM ships twice on purpose. The equation wants on-chain transaction value,
which is not free; exchange volume makes V = volume/market_cap and the
whole expression collapses to price/f. crypto_qtm takes velocity as an
exogenous assumption and is a real sensitivity; crypto_qtm_panel_velocity
is the dissertation's literal reading, flagged circular with
collapses_to_price in its diagnostics. Where they differ is the finding.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
## Task 14: Family B — Metcalfe, estimated rather than assumed

Spec §10.3. `log(market_cap) = alpha + beta x log(active_addresses)`, with **beta estimated and
reported with a confidence interval**, never assumed to be 2.0 — published estimates give ≈1.69.

**Review Focus 2:** a high R² between two trending series is nearly guaranteed and is weak evidence.
So every fit also reports the Durbin-Watson statistic and the residual, which is what makes spurious
regression visible. Active addresses are 100% dense on 20 of 23 assets, so this family is
well-supported — BNB, TRX and POL are the exceptions and the density floor refuses them.

**Files:**
- Create: `cryptoval/methods/network.py`, `tests/test_methods_network.py`
- Create: `config/methods/metcalfe_estimated.yml`

**Interfaces:**
- Consumes: Task 10's contract; Task 11's `register`, `param_set_hash`.
- Produces: `fit_metcalfe(market_cap, addresses) -> dict` and one registered method.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_methods_network.py`:

```python
import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.network import MetcalfeEstimated, fit_metcalfe


def _frame(n: int = 800, beta: float = 1.69, noise: float = 0.15, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    addresses = np.exp(np.linspace(np.log(2e5), np.log(1.1e6), n))
    log_cap = 4.0 + beta * np.log(addresses) + rng.normal(0, noise, n)
    return pd.DataFrame(
        {
            "symbol": ["ETH"] * n,
            "metric_date": pd.date_range("2024-01-01", periods=n, freq="D").date,
            "market_cap_usd": np.exp(log_cap),
            "active_addresses": addresses,
            "supply": np.full(n, 1.2e8),
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )


def test_beta_is_recovered_with_a_confidence_interval():
    frame = _frame(beta=1.69)
    fit = fit_metcalfe(frame["market_cap_usd"], frame["active_addresses"])
    assert fit["beta"] == pytest.approx(1.69, rel=0.05)
    assert fit["beta_ci_low"] < fit["beta"] < fit["beta_ci_high"]
    # Never assumed: the theoretical 2.0 must not be inside a correctly recovered 1.69 CI
    # on this much data, which is the whole point of estimating it.
    assert not (fit["beta_ci_low"] <= 2.0 <= fit["beta_ci_high"])


def test_the_fit_reports_durbin_watson_and_r_squared():
    """Review Focus 2: R-squared alone cannot distinguish a real relationship from two
    trending series, so the autocorrelation statistic travels with it."""
    fit = fit_metcalfe(_frame()["market_cap_usd"], _frame()["active_addresses"])
    assert 0.0 <= fit["r_squared"] <= 1.0
    assert 0.0 <= fit["durbin_watson"] <= 4.0
    assert fit["n"] == 800


def test_two_unrelated_trending_series_get_a_high_r_squared_and_a_damning_dw():
    """The failure mode itself, demonstrated: both series merely grow."""
    n = 800
    t = np.linspace(0.0, 1.0, n)
    addresses = np.exp(np.log(1e5) + t * (np.log(1e6) - np.log(1e5)))
    # Both series merely grow, and the second's growth bends independently of the first.
    # The bend matters: two EXACT exponentials would fit each other perfectly in log-log,
    # leaving only floating-point noise as residual and making DW meaningless.
    unrelated_cap = np.exp(np.log(1e9) + t * (np.log(1e12) - np.log(1e9))) * (
        1.0 + 0.30 * np.sin(3.0 * np.pi * t)
    )
    fit = fit_metcalfe(pd.Series(unrelated_cap), pd.Series(addresses))
    assert fit["r_squared"] > 0.99
    # Strong positive autocorrelation: DW far below 2 says the residuals are not
    # independent, so the R-squared is not evidence.
    assert fit["durbin_watson"] < 0.5
    assert fit["residuals_autocorrelated"] is True


def test_the_estimate_uses_only_trailing_data():
    """No row may be fitted on information from its own future (spec 11.2)."""
    frame = _frame(n=600)
    out = MetcalfeEstimated().estimate(frame, {"min_fit_rows": 180, "refit_every_rows": 30})
    applicable = [e for e in out if e.applicable]
    # The first 180 rows cannot be fitted, so they are refusals, not estimates.
    assert len(applicable) == 600 - 180
    assert all(e.diagnostics["fit_rows"] <= i + 180 for i, e in enumerate(applicable))


def test_the_band_comes_from_the_beta_confidence_interval():
    out = MetcalfeEstimated().estimate(_frame(n=400), {"min_fit_rows": 180, "refit_every_rows": 30})
    last = [e for e in out if e.applicable][-1]
    assert last.band_low < last.fair_value < last.band_high
    assert last.diagnostics["beta"] > 1.0


def test_a_flat_address_series_is_refused_rather_than_fitted():
    frame = _frame(n=400)
    frame["active_addresses"] = 500000.0
    out = MetcalfeEstimated().estimate(frame, {"min_fit_rows": 180, "refit_every_rows": 30})
    assert not any(e.applicable for e in out)
    assert "no variation" in out[-1].reason
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_network.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptoval.methods.network'`

- [ ] **Step 3: Implement `cryptoval/methods/network.py`**

```python
"""Family B -- network / adoption (spec 10.3).

beta is ESTIMATED, never assumed. Metcalfe's law implies 2.0; published estimates give
about 1.69, and the gap matters because the exponent is what the valuation turns on.

The honest caveat travels in the diagnostics rather than the prose: an active address is
not a user, and a high R-squared between two trending series is nearly guaranteed. So
every fit reports beta's confidence interval, R-squared, and the Durbin-Watson statistic,
with residuals_autocorrelated set when DW says the residuals are not independent -- which
is the condition under which the R-squared is not evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

from cryptoval.methods.base import Estimate
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()

# Below this, residuals are strongly positively autocorrelated and the fit's R-squared
# tells you the series both trend, not that one explains the other.
DW_AUTOCORRELATION_FLOOR = 1.5


def fit_metcalfe(market_cap: pd.Series, addresses: pd.Series) -> dict:
    """OLS of log(market_cap) on log(active_addresses)."""
    cap = pd.to_numeric(market_cap, errors="coerce")
    adr = pd.to_numeric(addresses, errors="coerce")
    usable = pd.DataFrame({"cap": cap, "adr": adr}).dropna()
    usable = usable[(usable["cap"] > 0) & (usable["adr"] > 0)]
    if len(usable) < 30:
        raise ValueError(f"only {len(usable)} usable observations; need at least 30")
    if usable["adr"].nunique() < 2:
        raise ValueError("active_addresses has no variation, so beta is not identified")

    y = np.log(usable["cap"].to_numpy())
    x = sm.add_constant(np.log(usable["adr"].to_numpy()))
    model = sm.OLS(y, x).fit()
    low, high = model.conf_int(alpha=0.05)[1]
    dw = float(durbin_watson(model.resid))
    return {
        "alpha": float(model.params[0]),
        "beta": float(model.params[1]),
        "beta_ci_low": float(low),
        "beta_ci_high": float(high),
        "r_squared": float(model.rsquared),
        "durbin_watson": dw,
        # The flag that stops a 0.99 R-squared from being mistaken for evidence.
        "residuals_autocorrelated": bool(dw < DW_AUTOCORRELATION_FLOOR),
        "n": int(len(usable)),
    }


@register
class MetcalfeEstimated:
    spec = _SPECS["metcalfe_estimated"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        minimum = int(params.get("min_fit_rows", 180))
        refit_every = int(params.get("refit_every_rows", 30))
        phash = param_set_hash(params)
        out: list[Estimate] = []
        fit: dict | None = None
        last_error = "insufficient trailing history"

        for position, row in enumerate(frame.itertuples(index=False)):
            vintage = getattr(row, "panel_vintage_ts", None)
            # Expanding TRAILING window: rows 0..position-1 only, so no row is fitted on
            # its own future. Refitting every N rows keeps a 4,000-day backfill tractable
            # without ever reaching forward.
            if position >= minimum and (fit is None or position % refit_every == 0):
                history = frame.iloc[:position]
                try:
                    fit = fit_metcalfe(history["market_cap_usd"], history["active_addresses"])
                except ValueError as exc:
                    fit, last_error = None, str(exc)

            addresses = pd.to_numeric(
                pd.Series([row.active_addresses]), errors="coerce"
            ).iloc[0]
            supply = float(row.supply) if row.supply else None
            if fit is None or supply is None or supply <= 0 or not np.isfinite(addresses) or addresses <= 0:
                out.append(
                    Estimate(
                        symbol=row.symbol,
                        metric_date=row.metric_date,
                        method_id=self.spec.method_id,
                        param_set_hash=phash,
                        fair_value=None,
                        band_low=None,
                        band_high=None,
                        inputs_used={},
                        diagnostics={},
                        applicable=False,
                        reason=last_error if fit is None else "missing addresses or supply",
                        panel_vintage_ts=vintage,
                    )
                )
                continue

            log_adr = float(np.log(addresses))
            implied = float(np.exp(fit["alpha"] + fit["beta"] * log_adr))
            low = float(np.exp(fit["alpha"] + fit["beta_ci_low"] * log_adr))
            high = float(np.exp(fit["alpha"] + fit["beta_ci_high"] * log_adr))
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=implied / supply,
                    band_low=min(low, high) / supply,
                    band_high=max(low, high) / supply,
                    inputs_used={"active_addresses": float(addresses), "supply": supply},
                    diagnostics={**fit, "fit_rows": fit["n"]},
                    applicable=True,
                    panel_vintage_ts=vintage,
                )
            )
        return out
```

- [ ] **Step 4: Write the method config**

```yaml
# metcalfe_estimated.yml
method_id: metcalfe_estimated
family: B_network
# market_cap_usd is the REGRESSAND, not a derived input, so this method is not circular --
# active addresses are a genuinely independent fundamental (spec 10.1).
requires: [market_cap_usd, active_addresses, supply]
min_density: 0.6
min_rows: 365
params:
  min_fit_rows: 180
  refit_every_rows: 30
citation: "Metcalfe (1995); Alabi (2017); Peterson (2018). beta estimated, never assumed to be 2.0"
```

- [ ] **Step 5: Run the tests, full suite and lint, then commit**

Run: `.venv/bin/python -m pytest tests/test_methods_network.py -v` then
`.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests`

```bash
git add cryptoval/methods/network.py config/methods tests/test_methods_network.py
git commit -m "$(cat <<'MSG'
feat: Family B, Metcalfe with beta estimated rather than assumed

Metcalfe's law implies beta = 2.0 and published estimates give about 1.69,
so the exponent the valuation turns on is fitted and reported with a
confidence interval.

Every fit also carries Durbin-Watson and a residuals_autocorrelated flag,
because a high R-squared between two trending series is nearly guaranteed
and is not evidence. A test demonstrates the failure mode directly: two
unrelated growing series score R-squared above 0.99 with DW below 0.5.

Fits use an expanding trailing window and refit every 30 rows, so no row
is fitted on its own future.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 15: Family D — relative multiples

Spec §10.5. NVT, the NVT golden ratio, and the dissertation's Price-to-Utility ratio. P/U needs a
staking ratio, which the free tier does not serve, so it is an explicit parameter and the method
refuses rather than guessing — this is the applicability mechanism earning its place.

**Files:**
- Create: `cryptoval/methods/multiples.py`, `tests/test_methods_multiples.py`
- Create: `config/methods/{nvt_reversion,nvt_golden_ratio,price_to_utility}.yml`

**Interfaces:**
- Consumes: Task 10's contract; Task 11's `register`, `param_set_hash`; `dilution_rate` is computed
  here from `supply`.
- Produces: `dilution_rate(supply, window_rows) -> pd.Series` and three registered methods.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_methods_multiples.py`:

```python
import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.multiples import (
    NvtGoldenRatio,
    NvtReversion,
    PriceToUtility,
    dilution_rate,
)


def _frame(n: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTC"] * n,
            "metric_date": pd.date_range("2024-01-01", periods=n, freq="D").date,
            "market_cap_usd": np.linspace(1.0e12, 1.7e12, n),
            "volume_reported_spot_usd": np.linspace(2.0e9, 3.0e9, n),
            "volume_total_usd": np.linspace(1.0e10, 1.5e10, n),
            "velocity_total": np.linspace(0.009, 0.010, n),
            "supply": np.linspace(1.9e7, 1.96e7, n),
            "price_usd": np.linspace(52000, 87000, n),
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )


def test_dilution_rate_is_annualised_supply_growth():
    supply = pd.Series(np.linspace(100.0, 102.0, 731))
    rate = dilution_rate(supply, window_rows=365)
    # Roughly 1% over the trailing year at the midpoint.
    assert rate.iloc[365] == pytest.approx(0.01, rel=0.15)
    # The first year has no trailing year to compare against.
    assert rate.iloc[:365].isna().all()


def test_nvt_uses_the_declared_volume_series():
    frame = _frame()
    reported = NvtReversion().estimate(
        frame, {"volume_column": "volume_reported_spot_usd", "min_history_rows": 180}
    )
    total = NvtReversion().estimate(
        frame, {"volume_column": "volume_total_usd", "min_history_rows": 180}
    )
    last_reported = [e for e in reported if e.applicable][-1]
    last_total = [e for e in total if e.applicable][-1]
    # Spec 10.1: the two volume series differ ~5x, so NVT built on them must differ too.
    # If a model had coalesced them these would be equal and the discipline would be lost.
    assert last_reported.fair_value != pytest.approx(last_total.fair_value)
    assert last_reported.inputs_used["volume_column"] == "volume_reported_spot_usd"


def test_the_golden_ratio_compares_nvt_to_its_two_year_average():
    out = NvtGoldenRatio().estimate(_frame(), {"volume_column": "volume_total_usd", "ma_rows": 730})
    last = [e for e in out if e.applicable][-1]
    assert last.diagnostics["nvt"] > 0
    assert last.diagnostics["nvt_ma"] > 0
    assert last.diagnostics["golden_ratio"] == pytest.approx(
        last.diagnostics["nvt"] / last.diagnostics["nvt_ma"]
    )


def test_the_golden_ratio_needs_two_years_before_it_says_anything():
    out = NvtGoldenRatio().estimate(
        _frame(n=500), {"volume_column": "volume_total_usd", "ma_rows": 730}
    )
    assert not any(e.applicable for e in out)
    assert "730" in out[-1].reason


def test_price_to_utility_refuses_without_a_staking_ratio():
    """The free tier serves no staking ratio, so the method declines rather than guessing."""
    out = PriceToUtility().estimate(_frame(), {"staking_ratio": None, "min_history_rows": 180})
    assert not any(e.applicable for e in out)
    assert "staking_ratio" in out[-1].reason


def test_price_to_utility_computes_when_the_ratio_is_supplied():
    out = PriceToUtility().estimate(
        _frame(), {"staking_ratio": 0.25, "min_history_rows": 180}
    )
    last = [e for e in out if e.applicable][-1]
    assert last.diagnostics["token_utility"] > 0
    assert last.inputs_used["staking_ratio"] == 0.25
    # The holding-time correction's finding applies here too: P/U inverts velocity.
    assert last.diagnostics["uses_velocity_inversion"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_methods_multiples.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptoval.methods.multiples'`

- [ ] **Step 3: Write the three method configs FIRST** (Step 5's YAML; same import-time reason as Task 12)

- [ ] **Step 4: Implement `cryptoval/methods/multiples.py`**

```python
"""Family D -- relative multiples (spec 10.5).

Every multiple here reverts to a TRAILING statistic, never a full-sample one: a median
over the whole series tells a 2017 row what 2026 looked like, and Plan 4's walk-forward
tests would then be measuring leakage rather than skill.

NVT records which volume column produced it on every row. The two series differ about
fivefold (spec 10.1), so "NVT" without that label is two different numbers wearing one
name.
"""

from __future__ import annotations

import pandas as pd

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.cost import _finite
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def dilution_rate(supply: pd.Series, window_rows: int = 365) -> pd.Series:
    """Trailing supply growth, NaN until a full window exists.

    NaN rather than 0.0 for the first window: a zero dilution rate is a claim about
    monetary policy, and "we cannot see a year back yet" is not that claim.
    """
    values = pd.to_numeric(supply, errors="coerce")
    previous = values.shift(window_rows)
    return (values / previous.where(previous > 0)) - 1.0


def _refuse(spec: MethodSpec, row, phash: str, reason: str) -> Estimate:
    return Estimate(
        symbol=row.symbol,
        metric_date=row.metric_date,
        method_id=spec.method_id,
        param_set_hash=phash,
        fair_value=None,
        band_low=None,
        band_high=None,
        inputs_used={},
        diagnostics={},
        applicable=False,
        reason=reason,
        panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
    )


@register
class NvtReversion:
    spec = _SPECS["nvt_reversion"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        column = params.get("volume_column", "volume_total_usd")
        minimum = int(params.get("min_history_rows", 180))
        phash = param_set_hash(params)

        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
        volume = pd.to_numeric(frame[column], errors="coerce")
        nvt = market_cap / volume.where(volume > 0)
        # shift(1) then expanding: this row's own NVT never enters its own target.
        target = nvt.shift(1).expanding(min_periods=minimum).median()

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            today_volume = _finite(volume.iloc[position])
            supply = _finite(row.supply)
            multiple = _finite(target.iloc[position])
            if multiple is None:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        phash,
                        f"fewer than {minimum} prior NVT observations on {column}",
                    )
                )
                continue
            if today_volume is None or today_volume <= 0 or supply is None or supply <= 0:
                out.append(_refuse(self.spec, row, phash, "missing volume or supply on this day"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=today_volume * multiple / supply,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        # Load-bearing: NVT on the two volume series is two different
                        # numbers, so a row that did not name its source would be
                        # unreproducible.
                        "volume_column": column,
                        "volume": today_volume,
                        "supply": supply,
                        "trailing_median_nvt": multiple,
                    },
                    diagnostics={
                        "nvt": _finite(nvt.iloc[position]),
                        "deviation_from_trailing_median": (
                            _finite(nvt.iloc[position]) / multiple if multiple else None
                        ),
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class NvtGoldenRatio:
    """NVT against its own two-year moving average (Kalichkin 2018)."""

    spec = _SPECS["nvt_golden_ratio"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        column = params.get("volume_column", "volume_total_usd")
        ma_rows = int(params.get("ma_rows", 730))
        phash = param_set_hash(params)

        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
        volume = pd.to_numeric(frame[column], errors="coerce")
        nvt = market_cap / volume.where(volume > 0)
        nvt_ma = nvt.shift(1).rolling(ma_rows, min_periods=ma_rows).mean()

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            average = _finite(nvt_ma.iloc[position])
            today_volume = _finite(volume.iloc[position])
            supply = _finite(row.supply)
            if average is None or average <= 0:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        phash,
                        f"fewer than {ma_rows} prior NVT observations for the moving average",
                    )
                )
                continue
            if today_volume is None or today_volume <= 0 or supply is None or supply <= 0:
                out.append(_refuse(self.spec, row, phash, "missing volume or supply on this day"))
                continue
            today_nvt = _finite(nvt.iloc[position])
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    # Reversion to the two-year average NVT.
                    fair_value=today_volume * average / supply,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "volume_column": column,
                        "volume": today_volume,
                        "supply": supply,
                        "ma_rows": ma_rows,
                    },
                    diagnostics={
                        "nvt": today_nvt,
                        "nvt_ma": average,
                        "golden_ratio": (today_nvt / average) if today_nvt else None,
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class PriceToUtility:
    """The dissertation's P/U ratio (section 3.6).

    token_utility = (velocity x staking_ratio) / dilution_rate

    Two honest caveats ride along. The staking ratio is not served by any free source, so
    it is an unverified parameter and the method declines without it rather than guessing.
    And the construction inverts velocity, which holding_time_correction shows to be
    biased by exp(sigma^2) -- so uses_velocity_inversion is set on every row.
    """

    spec = _SPECS["price_to_utility"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        staking_ratio = params.get("staking_ratio")
        minimum = int(params.get("min_history_rows", 180))
        window = int(params.get("dilution_window_rows", 365))
        phash = param_set_hash(params)

        if staking_ratio is None:
            return [
                _refuse(
                    self.spec,
                    row,
                    phash,
                    "staking_ratio is not supplied and no free source serves one, so "
                    "token_utility is not computable",
                )
                for row in frame.itertuples(index=False)
            ]

        velocity = pd.to_numeric(frame["velocity_total"], errors="coerce")
        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        dilution = dilution_rate(frame["supply"], window)
        utility = (velocity * float(staking_ratio)) / dilution.where(dilution > 0)
        target = (price / utility).shift(1).expanding(min_periods=minimum).median()

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            today_utility = _finite(utility.iloc[position])
            multiple = _finite(target.iloc[position])
            if today_utility is None or today_utility <= 0:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        phash,
                        f"token_utility is not computable: needs {window} rows of supply "
                        "history and a positive dilution rate",
                    )
                )
                continue
            if multiple is None:
                out.append(
                    _refuse(self.spec, row, phash, f"fewer than {minimum} prior P/U observations")
                )
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=multiple * today_utility,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "staking_ratio": float(staking_ratio),
                        "dilution_window_rows": window,
                        "velocity_total": _finite(velocity.iloc[position]),
                        "trailing_median_price_to_utility": multiple,
                    },
                    diagnostics={
                        "token_utility": today_utility,
                        "dilution_rate": _finite(dilution.iloc[position]),
                        # holding_time_correction measures this bias at exp(sigma^2).
                        "uses_velocity_inversion": True,
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
```

`_finite` is imported from `cost.py` rather than copied: one definition of "this cannot be a price".

- [ ] **Step 5: The three method configs** (written in Step 3)

```yaml
# nvt_reversion.yml
method_id: nvt_reversion
family: D_multiples
requires: [market_cap_usd, volume_total_usd, supply]
min_density: 0.6
min_rows: 365
params: {volume_column: volume_total_usd, min_history_rows: 180}
citation: "Willy Woo / Chris Burniske (2017), NVT ratio; dissertation section 3.6"

# nvt_golden_ratio.yml
method_id: nvt_golden_ratio
family: D_multiples
requires: [market_cap_usd, volume_total_usd, supply]
min_density: 0.6
min_rows: 1095          # two years of MA plus a year to use it
params: {volume_column: volume_total_usd, ma_rows: 730}
citation: "Dmitry Kalichkin (2018), NVT golden ratio"

# price_to_utility.yml
method_id: price_to_utility
family: D_multiples
# velocity_total is market-cap-derived, so this method is flagged circular -- and it also
# inverts velocity, which is the inversion the holding-time correction invalidates.
requires: [price_usd, velocity_total, supply]
min_density: 0.6
min_rows: 730
params:
  staking_ratio: null    # the free tier serves none; supply one to enable the method
  min_history_rows: 180
citation: "dissertation section 3.6, price-to-utility ratio"
```

- [ ] **Step 6: Run the tests, full suite and lint, then commit**

```bash
git add cryptoval/methods/multiples.py config/methods tests/test_methods_multiples.py
git commit -m "$(cat <<'MSG'
feat: Family D, relative multiples

NVT and the NVT golden ratio, both reverting to a trailing statistic
rather than a full-sample one, and the dissertation's Price-to-Utility.

NVT records which volume series it used on every row: the two differ ~5x,
so NVT computed on them is two different numbers and a row that did not
say which would be unreproducible.

Price-to-Utility needs a staking ratio the free tier does not serve, so it
is a parameter and the method declines with a reason rather than guessing.
It also inverts velocity, so it carries the holding-time finding in its
diagnostics.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
## Task 16: Method cards, the `estimate` command, and the dissertation comparison

Spec §12 deliverable 4 and Phase 2b's exit criterion. One card per method, the registry wired into
the CLI, and the honest verdict on what reproduces.

**Files:**
- Create: `cryptoval/cards.py`, `tests/test_cards.py`
- Create: `docs/methods/<method_id>.md` (generated skeleton, hand-written failure modes)
- Modify: `cryptoval/cli.py`, `tests/test_cli.py`
- Create: `reports/dissertation-comparison.md`
- Modify: `Makefile`, `README` section if one exists

**Interfaces:**
- Consumes: `load_method_specs`, `METHODS`, `run_methods`, `ESTIMATE_COLUMNS` from Task 11;
  `circularity` from Task 10.
- Produces:
  - `cryptoval.cards.render_card(spec, extras) -> str`
  - `write_cards(specs, out_dir) -> list[str]`
  - `cli.run_estimate(settings, assets, method_ids=None) -> dict`
  - a new `estimate` subcommand

- [ ] **Step 1: Write the failing card tests**

Create `tests/test_cards.py`:

```python
from cryptoval.cards import render_card, write_cards
from cryptoval.methods.registry import load_method_specs


def test_a_card_states_the_formula_citation_and_failure_modes():
    spec = load_method_specs()["metcalfe_estimated"]
    card = render_card(spec, {"failure_modes": ["an active address is not a user"]})
    assert "metcalfe_estimated" in card
    assert spec.citation in card
    assert "an active address is not a user" in card
    assert "market_cap_usd" in card  # data dependencies are listed


def test_a_circular_method_says_so_prominently():
    """Spec 14: such metrics are labelled diagnostics, not independent fair values."""
    spec = load_method_specs()["crypto_qtm_panel_velocity"]
    card = render_card(spec, {})
    assert "Circular" in card
    assert "velocity_total" in card
    assert "not independent evidence" in card


def test_a_non_circular_method_does_not_claim_circularity():
    card = render_card(load_method_specs()["metcalfe_estimated"], {})
    assert "Circular" not in card


def test_a_card_is_written_per_method(tmp_path):
    specs = load_method_specs()
    paths = write_cards(specs, str(tmp_path))
    assert len(paths) == len(specs)
    assert all(p.endswith(".md") for p in paths)


def test_every_registered_method_has_a_config_and_vice_versa():
    """A method with no config cannot be run; a config with no method is a dead card."""
    import cryptoval.methods.cost  # noqa: F401  -- import registers the families
    import cryptoval.methods.monetary  # noqa: F401
    import cryptoval.methods.multiples  # noqa: F401
    import cryptoval.methods.network  # noqa: F401
    from cryptoval.methods.registry import METHODS

    assert set(METHODS) == set(load_method_specs())
```

- [ ] **Step 2: Run it to verify it fails, then implement `cryptoval/cards.py`**

Run: `.venv/bin/python -m pytest tests/test_cards.py -v` → `ModuleNotFoundError`

```python
"""Method cards (spec 12, deliverable 4).

One card per method: what it computes, from which columns, under what assumptions, how it
fails, and what it cites. The circularity warning is generated rather than written, because
a warning that has to be remembered is a warning that will be missing from exactly the
card where it matters.
"""

from __future__ import annotations

import pathlib

from cryptoval.methods.base import MethodSpec, circularity

_TEMPLATE = """# {method_id}

**Family:** {family}
**Citation:** {citation}
**Returns a fair value:** {returns_fair_value}

## Data dependencies

{requires}

Minimum coverage density: **{min_density:.0%}** per column. Minimum observations:
**{min_rows}**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

{params}
{circular_section}
## Failure modes

{failure_modes}

## Assumptions

{assumptions}
"""

_CIRCULAR = """
## Circular inputs

This method consumes **{columns}**, each of which is itself computed from market
capitalisation. Its output is therefore **not independent evidence** of value: it is a
diagnostic or a cost basis, and the report must treat it as one (spec section 14,
"method circularity").
"""


def render_card(spec: MethodSpec, extras: dict) -> str:
    circular = circularity(spec)
    return _TEMPLATE.format(
        method_id=spec.method_id,
        family=spec.family,
        citation=spec.citation or "(none recorded)",
        returns_fair_value="yes" if spec.returns_fair_value else "no, it is a correction",
        requires="\n".join(f"- `{c}`" for c in spec.requires) or "- (none)",
        min_density=spec.min_density,
        min_rows=spec.min_rows,
        params="\n".join(f"- `{k}` = `{v!r}`" for k, v in sorted(spec.params.items()))
        or "- (none)",
        circular_section=_CIRCULAR.format(columns=", ".join(f"`{c}`" for c in circular))
        if circular
        else "",
        failure_modes="\n".join(f"- {f}" for f in extras.get("failure_modes", []))
        or "- (to be written)",
        assumptions="\n".join(f"- {a}" for a in extras.get("assumptions", []))
        or "- (to be written)",
    )


def write_cards(specs: dict[str, MethodSpec], out_dir: str, extras: dict | None = None) -> list[str]:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extras = extras or {}
    written = []
    for method_id, spec in sorted(specs.items()):
        path = out / f"{method_id}.md"
        path.write_text(render_card(spec, extras.get(method_id, {})), encoding="utf-8")
        written.append(str(path))
    return written
```

- [ ] **Step 3: Generate the cards, then write the failure modes by hand**

Add to the `Makefile`:

```makefile
cards:
	$(PY) -I -c "import cryptoval.methods.cost, cryptoval.methods.monetary, cryptoval.methods.network, cryptoval.methods.multiples; from cryptoval.methods.registry import load_method_specs; from cryptoval.cards import write_cards; print('\n'.join(write_cards(load_method_specs(), 'docs/methods')))"
```

Run `make cards`, then fill in each card's **Failure modes** and **Assumptions** by hand. The
generated skeleton carries everything derivable from config; these two sections are the judgment
and must not be left as `(to be written)`. At minimum:

- `metcalfe_estimated` — an active address is not a user; exchange consolidation moves the series
  without any change in adoption; a high R² between trending series is not evidence, so read
  `durbin_watson` and `residuals_autocorrelated` first.
- `hayes_cost_of_production` — a floor, not a target; both parameters are assumptions, not
  measurements; the hash-rate unit is calibrated (record the measured ratio from Task 12 Step 5);
  mining with stranded or subsidised power breaks the cost argument.
- `thermocap_floor` — `thermocap_has_gap` means the figure understates cumulative cost, so the floor
  reads too low and anything built on it reads too high.
- `crypto_qtm_panel_velocity` — returns the market price divided by the circulating fraction; every
  residual is zero by construction; retained only to measure the circularity against `crypto_qtm`.
- `holding_time_correction` — the closed form assumes log-normal velocity, so read
  `anderson_darling_p`; a rejection means the ratio is an approximation for that asset.
- `price_to_utility` — the staking ratio is an unverified input, and the method inverts velocity,
  which `holding_time_correction` shows to be biased by `exp(sigma²)`.
- `mvrv_reversion` / `mvtv_reversion` / `realized_price` / `investor_cap_price` — all inherit the
  MVRV identity's fragility (spec §6.2): if Coin Metrics redefines either input, every number here
  is silently wrong, which is what the identity assertions guard.
- `nvt_reversion` / `nvt_golden_ratio` — exchange volume is not on-chain transaction value, and the
  two volume series differ ~5×; read `inputs_used.volume_column`.
- `j_curve` — the dissertation's printed CUV formula is dimensionally incoherent; this implements the
  economically intended form, and that deviation is the card's headline.

- [ ] **Step 4: Write the failing CLI test, then wire the command**

Add to `tests/test_cli.py`:

```python
def test_the_estimate_command_writes_the_gold_table(tmp_path, monkeypatch):
    settings = Settings(data_root=str(tmp_path))
    panel = pd.DataFrame(
        {
            "symbol": ["BTC"] * 60,
            "metric_date": pd.date_range("2026-01-01", periods=60, freq="D").date,
            "realized_cap": np.linspace(1.0e12, 1.05e12, 60),
            "supply": np.full(60, 1.9e7),
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )
    monkeypatch.setattr(cli, "_read_panel", lambda _settings: panel)
    report = cli.run_estimate(settings, [], method_ids=["realized_price"])

    assert report["realized_price"]["rows"] == 60
    out = pd.read_parquet(
        pathlib.Path(settings.resolver.table("gold", "valuation_estimate")) / "part-0.parquet"
    )
    assert set(out["method_id"]) == {"realized_price"}
    assert out["fair_value"].notna().all()


def test_the_estimate_command_reports_refusals_without_failing(tmp_path, monkeypatch):
    """A method that declines is a finding, so the command exits 0 and says how many."""
    settings = Settings(data_root=str(tmp_path))
    thin = pd.DataFrame(
        {
            "symbol": ["BNB"] * 5,
            "metric_date": pd.date_range("2026-01-01", periods=5, freq="D").date,
            "realized_cap": [1.0e10, None, None, None, None],
            "supply": np.full(5, 1.5e8),
            "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
        }
    )
    monkeypatch.setattr(cli, "_read_panel", lambda _settings: thin)
    report = cli.run_estimate(settings, [], method_ids=["realized_price"])
    assert report["realized_price"]["refusals"] == 1
    assert report["realized_price"]["rows"] == 1
```

In `cryptoval/cli.py` add the imports — the four family modules purely for their registration side
effect, plus the registry symbols the command actually calls:

```python
import cryptoval.methods.cost  # noqa: F401 -- import registers the family
import cryptoval.methods.monetary  # noqa: F401
import cryptoval.methods.multiples  # noqa: F401
import cryptoval.methods.network  # noqa: F401
from cryptoval.io.bronze import write_gold
from cryptoval.methods.registry import METHODS, run_methods
```

then:

```python
def _read_panel(settings: Settings) -> pd.DataFrame:
    """The conformed panel, read from the mart DuckDB built."""
    import duckdb

    con = duckdb.connect(str(pathlib.Path(settings.data_root) / "cryptoval.duckdb"), read_only=True)
    try:
        return con.execute("select * from fct_asset_metrics_daily").fetchdf()
    finally:
        con.close()


def run_estimate(
    settings: Settings, assets: list[Asset], method_ids: list[str] | None = None
) -> dict[str, dict]:
    panel = _read_panel(settings)
    estimates = run_methods(panel, METHODS, method_ids)
    write_gold(settings.resolver, "valuation_estimate", estimates)
    report: dict[str, dict] = {}
    for method_id, group in estimates.groupby("method_id"):
        report[str(method_id)] = {
            "rows": int(len(group)),
            "refusals": int((~group["applicable"].astype(bool)).sum()),
            "symbols": int(group["symbol"].nunique()),
        }
    return report
```

`write_gold` does not exist yet — add it to `cryptoval/io/bronze.py` as a two-line sibling of
`write_silver` that takes the layer name `"gold"`, and give it the same atomic-rename treatment.
Register the subcommand in `main` with an optional `--methods` flag, print the report, and
**return 0 even when there are refusals** — a refusal is a finding, not a failure.

- [ ] **Step 5: Run the whole pipeline end to end**

```bash
make seed && make cards
.venv/bin/python -m cryptoval.cli ingest --trailing-days 30
make dbt
.venv/bin/python -m cryptoval.cli estimate
make dbt           # rebuilds fct_valuation_estimate and dim_method over the new gold table
make validate
```

Then capture the headline table for the report:

```bash
.venv/bin/python -I -c "
import duckdb
con = duckdb.connect('data/cryptoval.duckdb', read_only=True)
print('=== estimates per method ===')
print(con.execute('''
  select method_id, count(*) rows, sum(case when applicable then 1 else 0 end) applicable,
         count(distinct symbol) symbols, max(is_circular::int) circular
  from fct_valuation_estimate group by 1 order by 1''').fetchdf().to_string())
print()
print('=== latest fair value per method for BTC, against price ===')
print(con.execute('''
  with latest as (select max(metric_date) d from fct_valuation_estimate where symbol='BTC')
  select e.method_id, round(e.fair_value,2) fair_value, round(e.band_low,2) lo,
         round(e.band_high,2) hi, e.is_circular, round(p.price_usd,2) price
  from fct_valuation_estimate e, latest l
  join fct_asset_metrics_daily p on p.symbol=e.symbol and p.metric_date=e.metric_date
  where e.symbol='BTC' and e.metric_date=l.d and e.applicable
  order by e.method_id''').fetchdf().to_string())
print()
print('=== refusals, with reasons ===')
print(con.execute('''
  select method_id, symbol, reason from fct_valuation_estimate
  where not applicable group by all order by method_id, symbol limit 40''').fetchdf().to_string())
"
```

- [ ] **Step 6: Write `reports/dissertation-comparison.md`**

This is the deliverable, and it is a judgment, not a dump. Structure it as:

1. **What reproduced.** Per method: the dissertation's published figure, ours, and the gap. Say
   plainly where they agree.
2. **What did not, and why.** For each disagreement, name the cause from one of: a data difference
   (a different vendor, series or window), a formula ambiguity resolved differently (the J-curve CUV
   case is already known), a corrected error (the QTM and P/U inversion, now measured at
   `exp(sigma²)` per asset — quote the measured ratios), or genuinely different market conditions
   between 2022 and 2026.
3. **What cannot be reproduced at all, and the measurement that says so.** Quote the refusal reasons
   verbatim from `fct_valuation_estimate` — TRX's missing MVRV, BNB's 19%-dense market cap, P/U's
   absent staking ratio.
4. **The circularity audit.** List every method flagged `is_circular` and state that its residuals
   against price are zero or near-zero by construction. If `crypto_qtm_panel_velocity`'s fair value
   does not equal `price / f` to floating-point tolerance on live data, something is wrong with the
   derivation — investigate before publishing.
5. **The honest verdict**, in one paragraph: which of the dissertation's ten methods survive contact
   with 2026 data, and which were measuring price against itself all along.

Carry spec §12's disclaimer verbatim: these are valuation estimates under stated assumptions, not
investment advice; the assets are volatile and the methods disagree by design.

- [ ] **Step 7: Run the full suite and lint, then commit**

```bash
.venv/bin/python -m pytest -q && .venv/bin/ruff check cryptoval tests
git add cryptoval/cards.py cryptoval/cli.py cryptoval/io/bronze.py docs/methods \
        reports tests/test_cards.py tests/test_cli.py Makefile
git commit -m "$(cat <<'MSG'
feat: method cards, the estimate command, and the dissertation comparison

One card per method, with the circularity warning generated rather than
written: a warning that has to be remembered is one that will be missing
from exactly the card where it matters.

The estimate command runs the registry over the mart and writes
gold/valuation_estimate. It exits 0 when methods decline, because a
refusal carrying the measurement that caused it is a research finding.

reports/dissertation-comparison.md states what reproduced, what did not
and why, what cannot be reproduced at all with the refusal reasons quoted,
and the circularity audit naming every method whose residuals against
price are zero by construction.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

## Done criteria for Plan 2

**Stage 2a:**
- `make lint && make test` green; roughly 290 unit tests, no network.
- `make validate` exits **0** on live data. The two failures Plan 1 ended with are gone: the
  impossible volumes are quarantined rather than stored, and the negative revenues are either
  legitimate or belong to unmapped protocols.
- `silver/quarantine` holds exactly the cells the contracts call impossible, each with its raw value
  and the bound it broke.
- `dim_asset` has one row per configured symbol — including SOL, which has no Coin Metrics data at
  all — and carries per-column observed density.
- `fct_asset_metrics_daily` is keyed on `symbol`, carries both volume series uncoalesced, and records
  `market_cap_source`. TRX shows `coingecko`; AVAX has non-zero reported-spot volume.
- Two bronze runs on one date both survive, and `read_bronze` returns the newer.

**Stage 2b:**
- All 14 methods registered, each with a config, a card whose failure modes are written, and a
  `dim_method` row.
- `cryptoval.cli estimate` populates `fct_valuation_estimate` with one row per
  (symbol, date, method, param set), every row carrying `panel_vintage_ts`.
- Every inapplicable estimate names its measurement; no inapplicable row carries a fair value, which
  the `Estimate` dataclass enforces.
- `crypto_qtm_panel_velocity` is flagged circular and its fair value equals `price / f` to
  floating-point tolerance — the demonstration that the dissertation's QTM measured price against
  itself.
- `metcalfe_estimated` reports β with a confidence interval and Durbin-Watson on every row.
- `reports/dissertation-comparison.md` states the verdict per method.

**Plan 3 starts from:** `fct_valuation_estimate` plus `dim_method`, the conformed `symbol` key, the
`revenue_regime` classification on `fct_protocol_revenue_daily` (which tells the DCF family which
protocols actually retain revenue), and the `applicable=False` mechanism, which Family E(b)'s
Cointime methods use unchanged until the BigQuery extraction lands.

---

## Appendix: spec coverage

| Spec section | Covered by | Note |
|---|---|---|
| §5 asset scope, gaps | Task 1 | TRX/POL/BNB/SOL limits become config, then `applicable=False` reasons |
| §5 macro series | Task 1 | Gold and DXY, resolving §15 Q1 |
| §5.1 vendor identity | Tasks 1, 2, 7 | The metric-to-id map, the union fetch, the staging resolution |
| §5.1 protocol identity | Tasks 1, 5 | Version-summed, rebrand-aware slug lists with collision refusal |
| §6.2 realized-cap identity | Task 12 | Inherited by four cost methods; the existing identity assertions still guard it |
| §7.4 bronze immutability | Task 6 | `run_id` partitioning |
| §8.1 star schema | Tasks 7, 8, 11 | `dim_asset`, `dim_method`, three facts, `fct_valuation_estimate` |
| §9.1 contracts, quarantine | Tasks 3, 4 | The quarantine table and the revenue-sign correction |
| §9.2 scoped gate | Task 9 | Blocking for mapped protocols, warning for the long tail |
| §10.1 method contract | Tasks 10, 11 | Computed applicability and declared circularity |
| §10.2 Family A | Task 13 | QTM (both readings), J-curve, holding-time correction |
| §10.3 Family B | Task 14 | Metcalfe estimated. Peterson, LPPLS and NVM → Plan 3 |
| §10.4 Family C | — | **Plan 3.** `revenue_regime` prepares its denominator |
| §10.5 Family D | Task 15 | NVT, NVT golden ratio, P/U. Remaining multiples → Plan 3 |
| §10.6 Family E(a) | Task 12 | Realized price, Thermocap, MVTV, MVRV, Investor Cap, Hayes |
| §10.6 Family E(b) | — | **Plan 3**, behind the BigQuery extraction |
| §10.7 Family F | — | **Plan 3**, the falsified-benchmark shelf |
| §10.8 Family G | — | **Plan 4**, composite, after the statistical layer |
| §11 statistical layer | — | **Plan 4.** The no-lookahead discipline is enforced from Task 12 onward: every fit uses a trailing window |
| §12 method cards | Task 16 | One per method, circularity generated |
| §12 report, dashboard | Task 16 (partial) | The comparison chapter only; the full report and Streamlit → Plan 5 |
| §15 both open questions | Task 1, spec revision | Resolved 2026-10-12 |
