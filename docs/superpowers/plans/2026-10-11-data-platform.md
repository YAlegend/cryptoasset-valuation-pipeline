# Cryptoasset Valuation — Data Platform Implementation Plan (Plan 1 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, point-in-time-correct daily panel of cryptoasset market and network
fundamentals from four free data sources, with data contracts enforced as blocking tests.

**Architecture:** Medallion layout on Parquet (bronze immutable landing → silver conformed
bi-temporal panel → gold/marts), queried by DuckDB as a compute engine over Parquet rather than as a
system of record. Ingestion is a set of idempotent `(source, date_range) → DataFrame` adapters behind
one `Source` protocol; revisions are captured by an SCD Type 2 merge; dbt-core builds and tests the
silver and mart models. Everything resolves through one path resolver so local filesystem and object
storage are interchangeable.

**Tech Stack:** Python 3.14.5, httpx, pandas 3.0.6, pyarrow, DuckDB 1.5.6, dbt-core 1.12.5 with
dbt-duckdb 1.11.0, pytest 9, ruff, yfinance, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-11-cryptoasset-valuation-pipeline-design.md`

**Plan sequence:** This is Plan 1 of 5. Plan 2 (method registry, spec Phase 2+3a), Plan 3 (UTXO
extraction, Phase 3b), Plan 4 (statistical layer, Phase 4), Plan 5 (consumers and cloud, Phase 5+6)
all depend on the silver panel this plan produces.

## Global Constraints

Every task's requirements implicitly include this section.

- **Free tiers only, no exceptions.** No paid API calls. No API keys required by any source in this
  plan. BigQuery is reached only via the no-billing-account sandbox (spec §6.3).
- **Python 3.14.5**, dependencies pinned to exact versions in `requirements.txt` (verified to resolve
  2026-10-11): `duckdb==1.5.6`, `dbt-core==1.12.5`, `dbt-duckdb==1.11.0`, `pandas==3.0.6`,
  `pyarrow==26.0.0`, `httpx==0.28.1`, `pytest==9.1.1`, `pytest-cov==7.1.0`, `ruff==0.17.0`,
  `pyyaml==6.0.3`, `yfinance==0.2.66`.
- **All timestamps UTC.** Every datetime parsed with `utc=True`; naive datetimes are a bug.
- **Parquet is the system of record.** DuckDB files are derived and disposable. Never write
  authoritative data only into a `.duckdb` file.
- **Bronze is immutable.** Raw responses are never mutated, only added to. Re-running a date
  overwrites that date's partition wholesale; it never edits rows in place.
- **No network calls in unit tests.** Unit tests read recorded fixtures. Tests that hit the network
  are marked `@pytest.mark.integration` and are excluded from the default `pytest` run.
- **Trailing-window refetch:** scheduled runs refetch the last 30 days, not only T-1 (spec §8.2).
- **Point-in-time:** every silver panel row carries `effective_start_ts`, `effective_end_ts`,
  `is_current`, `row_hash` (spec §8.2).
- **No `train_test_split`, no shuffling** of time-ordered data anywhere in the repository. Enforced
  by a lint test in Task 1 so the prohibition is in place before any modelling code exists (spec §11.2).

## Review Focus

Failure modes the spec implies that no obvious task test would exercise. Each has its test assigned
to the task owning the code.

1. **MVRV of zero, null or negative** makes the load-bearing `realized_cap = market_cap / mvrv`
   derivation divide by zero. Expected: `NaN` with a recorded warning, never `inf` and never a
   silently-wrong row. → Task 8.
2. **Duplicate `(asset, metric_date)` rows** in one paginated response would let the SCD2 merge create
   two `is_current=True` rows for one key, silently corrupting every point-in-time query downstream.
   Expected: incoming rows deduplicated on key before merge; exactly one current row per key, always.
   → Task 11.
3. **A 200 response with null metric columns** (Coin Metrics omits metrics for some assets) must not
   be coerced to `0.0` and stored as a real observation. Expected: nulls stay null and do not
   participate in identity checks. → Task 8.
4. **A revision landing outside the 30-day trailing window** would be missed forever by scheduled
   runs. Expected: a reconciliation mode that refetches an arbitrary range and reports how many rows
   changed, so drift is detectable rather than invisible. → Task 11.
5. **A partial fetch failing mid-pagination** must not leave a bronze partition that looks complete.
   Expected: the partition is written only after the full fetch succeeds; a failed run records
   `status="failed"` in the manifest and leaves no partition. → Task 7.

---

## File Structure

```
pyproject.toml                       # project metadata, pytest/ruff config
requirements.txt                     # exact pins
Makefile                             # setup, test, lint, ingest, backfill, validate, dbt
.github/workflows/ci.yml             # lint + unit tests + dbt build on fixtures
.env.example                         # documents env vars (none secret in this plan)

cryptoval/
  __init__.py                        # __version__
  paths.py                           # PathResolver: layer/partition URIs, local or remote
  config.py                          # Settings + Asset universe loaded from YAML
  manifest.py                        # RunManifest: append-only run records
  hashing.py                         # stable row/payload hashing
  scd2.py                            # bi-temporal merge + as-of view
  budget.py                          # byte-budget gate + monthly ledger (BigQuery guard)
  contracts/
    __init__.py
    loader.py                        # load + validate contract YAML
    checks.py                        # freshness/unique/not-null/range/volume-anomaly
    identities.py                    # cross-column identity assertions
  sources/
    __init__.py
    base.py                          # Source protocol + FetchResult
    coinmetrics.py                   # CM community: 31 series, realized-cap derivation
    defillama.py                     # protocol fees/revenue + methodology hash
    prices.py                        # CoinGecko market_chart + yfinance equities
  io/
    __init__.py
    bronze.py                        # atomic bronze partition writer
    duck.py                          # DuckDB connection, Parquet views
  cli.py                             # ingest / backfill / validate / reconcile

config/
  settings.yml
  assets.yml                         # the 23-asset universe + 5 equities
  contracts/coinmetrics_asset_metrics.yml
  contracts/defillama_fees.yml
  contracts/coingecko_market_chart.yml

dbt/
  dbt_project.yml
  profiles.yml                       # duckdb target + bigquery target (spec §7.3)
  models/silver/silver_asset_metrics.sql
  models/silver/silver_protocol_revenue.sql
  models/silver/_silver.yml           # dbt tests
  models/marts/dim_asset.sql
  models/marts/dim_date.sql
  models/marts/fct_asset_metrics_daily.sql
  models/marts/_marts.yml

tests/
  conftest.py                        # fixture loaders, tmp data root
  fixtures/coinmetrics_btc_eth.json
  fixtures/coinmetrics_null_metrics.json
  fixtures/defillama_fees_overview.json
  fixtures/coingecko_btc_market_chart.json
  test_repo_hygiene.py               # the no-shuffle lint
  test_paths.py
  test_config.py
  test_manifest.py
  test_contracts.py
  test_identities.py
  test_bronze.py
  test_coinmetrics.py
  test_defillama.py
  test_prices.py
  test_scd2.py
  test_budget.py
  test_cli.py
  test_integration_sources.py        # @pytest.mark.integration
```

**Responsibility boundaries:** `sources/*` know about HTTP and vendor payload shapes and nothing
about storage. `io/*` knows about storage and nothing about vendors. `scd2.py` and `contracts/*` are
pure functions over DataFrames with no IO. `cli.py` is the only module that wires them together.
This is what lets Plan 2's method registry depend on the silver panel without importing any source
code.

---

## Task 1: Repo scaffold, pinned dependencies, CI

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `Makefile`, `.github/workflows/ci.yml`,
  `.env.example`, `.gitignore`, `cryptoval/__init__.py`
- Test: `tests/test_repo_hygiene.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `cryptoval.__version__: str`. A working `make test` and `make lint`. The repository
  invariant that no module calls `train_test_split` or shuffles time-ordered data.

- [ ] **Step 1: Initialise the repository and write the ignore file**

```bash
cd "/Users/yas/Documents/Crypto/Crypto Valuation"
git init
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
data/
*.duckdb
dbt/target/
dbt/logs/
.pytest_cache/
.coverage
htmlcov/
.env
EOF
```

`data/` is ignored because Parquet is regenerable from bronze and bronze is regenerable from the
APIs; the repository stays small and the pipeline stays reproducible.

- [ ] **Step 2: Write the dependency pins and project config**

```bash
cat > requirements.txt <<'EOF'
duckdb==1.5.6
dbt-core==1.12.5
dbt-duckdb==1.11.0
pandas==3.0.6
pyarrow==26.0.0
httpx==0.28.1
pyyaml==6.0.3
yfinance==0.2.66
pytest==9.1.1
pytest-cov==7.1.0
ruff==0.17.0
EOF

cat > pyproject.toml <<'EOF'
[project]
name = "cryptoval"
version = "0.1.0"
requires-python = ">=3.14"
description = "Cryptoasset valuation data platform"

[tool.setuptools.packages.find]
include = ["cryptoval*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not integration' --strict-markers"
markers = [
    "integration: hits live network APIs; excluded from the default run",
]

[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
EOF
```

`addopts` excludes integration tests by default, so `make test` never depends on network access.

- [ ] **Step 3: Write the package init**

```bash
mkdir -p cryptoval
cat > cryptoval/__init__.py <<'EOF'
"""Cryptoasset valuation data platform."""

__version__ = "0.1.0"
EOF
```

- [ ] **Step 4: Write the failing hygiene test**

This test encodes the Global Constraint against shuffled time-series splits *before* any modelling
code exists, so the prohibition cannot be violated later without a red test.

```python
# tests/test_repo_hygiene.py
import pathlib

import cryptoval

REPO = pathlib.Path(__file__).resolve().parent.parent
FORBIDDEN = ("train_test_split", ".sample(frac=", "shuffle=True")


def test_version_is_exposed():
    assert cryptoval.__version__ == "0.1.0"


def test_no_shuffled_splits_anywhere_in_package():
    offenders = []
    for path in (REPO / "cryptoval").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{path.relative_to(REPO)}: {token}")
    assert offenders == [], (
        "Time-ordered data must never be shuffled or randomly split "
        f"(spec section 11.2). Offenders: {offenders}"
    )
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_repo_hygiene.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval'` (the package is not yet
installed into an environment).

- [ ] **Step 6: Create the environment and install**

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_repo_hygiene.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 8: Write the Makefile**

```bash
cat > Makefile <<'EOF'
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup test lint fmt ingest backfill validate reconcile dbt all

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	$(PY) -m pytest -v

test-integration:
	$(PY) -m pytest -v -m integration

lint:
	.venv/bin/ruff check cryptoval tests

fmt:
	.venv/bin/ruff format cryptoval tests

ingest:
	$(PY) -m cryptoval.cli ingest --trailing-days 30

backfill:
	$(PY) -m cryptoval.cli backfill --start 2015-01-01

validate:
	$(PY) -m cryptoval.cli validate

reconcile:
	$(PY) -m cryptoval.cli reconcile --start 2015-01-01

dbt:
	cd dbt && ../.venv/bin/dbt build --profiles-dir .

all: lint test ingest dbt validate
EOF
```

- [ ] **Step 9: Write the CI workflow**

```bash
mkdir -p .github/workflows
cat > .github/workflows/ci.yml <<'EOF'
name: ci

on:
  push:
    branches: ["**"]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e .
      - name: Lint
        run: ruff check cryptoval tests
      - name: Unit tests
        run: pytest -v --cov=cryptoval --cov-report=term-missing
EOF
```

Unit tests only — no network, so CI is deterministic. A `dbt build` step against fixtures is added in
Task 13 once dbt models exist.

- [ ] **Step 10: Verify lint and the full suite pass, then commit**

Run: `make lint && make test`
Expected: ruff reports no issues; 2 tests pass.

```bash
cat > .env.example <<'EOF'
# No secrets are required for Plan 1: Coin Metrics Community, DefiLlama and
# CoinGecko Demo all work without an API key.
CRYPTOVAL_DATA_ROOT=./data
EOF

git add -A
git commit -m "feat: scaffold cryptoval data platform with pinned deps and CI"
```

---

## Task 2: Path resolver

**Files:**
- Create: `cryptoval/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PathResolver(root: str)` with `.layer(name: str) -> str`,
  `.bronze_partition(source: str, ingest_date: datetime.date) -> str`,
  `.table(layer: str, name: str) -> str`, `.is_remote -> bool`. Every other module obtains paths
  from this class and never joins paths itself — this is the single seam that makes the cloud
  migration in Plan 5 a configuration change (spec §7.3).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_paths.py
import datetime as dt

import pytest

from cryptoval.paths import PathResolver


def test_local_layer_path():
    r = PathResolver("./data")
    assert r.layer("bronze") == "./data/bronze"


def test_trailing_slash_in_root_is_normalised():
    assert PathResolver("./data/").layer("silver") == "./data/silver"


def test_remote_root_is_passed_through_unchanged():
    r = PathResolver("gs://my-bucket/cryptoval")
    assert r.layer("gold") == "gs://my-bucket/cryptoval/gold"
    assert r.is_remote is True


def test_local_root_is_not_remote():
    assert PathResolver("./data").is_remote is False


def test_bronze_partition_layout_is_hive_style():
    r = PathResolver("./data")
    got = r.bronze_partition("coinmetrics", dt.date(2026, 10, 11))
    assert got == "./data/bronze/source=coinmetrics/ingest_date=2026-10-11"


def test_table_path():
    r = PathResolver("./data")
    assert r.table("silver", "asset_metrics") == "./data/silver/asset_metrics"


def test_unknown_layer_is_rejected():
    with pytest.raises(ValueError, match="unknown layer"):
        PathResolver("./data").layer("platinum")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.paths'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/paths.py
"""Single source of truth for storage locations.

Local filesystem and object storage differ only in the root URI, so every
consumer asks this resolver for paths instead of joining them itself.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

LAYERS = ("bronze", "silver", "gold", "marts")


@dataclass(frozen=True)
class PathResolver:
    root: str

    def _join(self, *parts: str) -> str:
        return "/".join([self.root.rstrip("/"), *parts])

    @property
    def is_remote(self) -> bool:
        return "://" in self.root

    def layer(self, name: str) -> str:
        if name not in LAYERS:
            raise ValueError(f"unknown layer: {name!r}; expected one of {LAYERS}")
        return self._join(name)

    def table(self, layer: str, name: str) -> str:
        return "/".join([self.layer(layer), name])

    def bronze_partition(self, source: str, ingest_date: dt.date) -> str:
        return "/".join(
            [self.layer("bronze"), f"source={source}", f"ingest_date={ingest_date.isoformat()}"]
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_paths.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/paths.py tests/test_paths.py
git commit -m "feat: add storage path resolver for local and object-store roots"
```

---

## Task 3: Configuration and the asset universe

**Files:**
- Create: `cryptoval/config.py`, `config/settings.yml`, `config/assets.yml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `PathResolver` from Task 2.
- Produces: `Asset` (frozen dataclass: `symbol`, `cm_id`, `gecko_id`, `yahoo_id`, `sector`,
  `consensus`, `is_stablecoin`), `Settings` (`data_root`, `trailing_refetch_days`,
  `bigquery_monthly_byte_budget`, `request_timeout_s`, `max_retries`), and
  `load_config(config_dir: str = "config") -> tuple[Settings, list[Asset]]`. Plus
  `Settings.resolver -> PathResolver`. Plan 2's method registry reads `Asset.consensus` to decide
  whether cost-of-production applies (spec §10.6).

- [ ] **Step 1: Write the config files**

The universe is the 23 cryptoassets plus 5 equities from spec §5. `cm_id: null` records an asset
absent from Coin Metrics' free tier — the documented SOL gap — so the absence is data, not a
surprise at runtime.

```bash
mkdir -p config
cat > config/settings.yml <<'EOF'
data_root: ./data
trailing_refetch_days: 30
bigquery_monthly_byte_budget: 900000000000   # 0.9 TB, inside the 1 TB sandbox free tier
request_timeout_s: 30
max_retries: 5
EOF

cat > config/assets.yml <<'EOF'
# sector values must be one of the SECTORS tuple in cryptoval/config.py
# consensus: pow | pos | other | na   (drives cost-of-production applicability)
assets:
  - {symbol: BTC,  cm_id: btc,     gecko_id: bitcoin,       sector: pow_monetary, consensus: pow}
  - {symbol: LTC,  cm_id: ltc,     gecko_id: litecoin,      sector: pow_monetary, consensus: pow}
  - {symbol: BCH,  cm_id: bch,     gecko_id: bitcoin-cash,  sector: pow_monetary, consensus: pow}
  - {symbol: DOGE, cm_id: doge,    gecko_id: dogecoin,      sector: pow_monetary, consensus: pow}
  - {symbol: ETH,  cm_id: eth,     gecko_id: ethereum,      sector: smart_contract_l1, consensus: pos}
  - {symbol: ADA,  cm_id: ada,     gecko_id: cardano,       sector: smart_contract_l1, consensus: pos}
  - {symbol: DOT,  cm_id: dot,     gecko_id: polkadot,      sector: smart_contract_l1, consensus: pos}
  - {symbol: AVAX, cm_id: avaxc,   gecko_id: avalanche-2,   sector: smart_contract_l1, consensus: pos}
  - {symbol: ALGO, cm_id: algo,    gecko_id: algorand,      sector: smart_contract_l1, consensus: pos}
  - {symbol: TRX,  cm_id: trx_eth, gecko_id: tron,          sector: smart_contract_l1, consensus: pos}
  - {symbol: ICP,  cm_id: icp,     gecko_id: internet-computer, sector: smart_contract_l1, consensus: pos}
  - {symbol: POL,  cm_id: pol_eth, gecko_id: polygon-ecosystem-token, sector: scaling, consensus: pos}
  - {symbol: UNI,  cm_id: uni,     gecko_id: uniswap,       sector: defi, consensus: other}
  - {symbol: AAVE, cm_id: aave,    gecko_id: aave,          sector: defi, consensus: other}
  - {symbol: CRV,  cm_id: crv,     gecko_id: curve-dao-token, sector: defi, consensus: other}
  - {symbol: MKR,  cm_id: mkr,     gecko_id: maker,         sector: defi, consensus: other}
  - {symbol: COMP, cm_id: comp,    gecko_id: compound-governance-token, sector: defi, consensus: other}
  - {symbol: SNX,  cm_id: snx,     gecko_id: havven,        sector: defi, consensus: other}
  - {symbol: LDO,  cm_id: ldo,     gecko_id: lido-dao,      sector: defi, consensus: other}
  - {symbol: LINK, cm_id: link,    gecko_id: chainlink,     sector: oracle, consensus: other}
  - {symbol: BNB,  cm_id: bnb,     gecko_id: binancecoin,   sector: exchange, consensus: other}
  - {symbol: MANA, cm_id: mana,    gecko_id: decentraland,  sector: metaverse, consensus: other}
  - {symbol: USDC, cm_id: usdc,    gecko_id: usd-coin,      sector: stablecoin, consensus: na, is_stablecoin: true}
  # Documented gap (spec section 5): SOL is absent from the Coin Metrics free tier.
  - {symbol: SOL,  cm_id: null,    gecko_id: solana,        sector: smart_contract_l1, consensus: pos}

equities:
  - {symbol: TSLA,  yahoo_id: TSLA}
  - {symbol: AAPL,  yahoo_id: AAPL}
  - {symbol: AMZN,  yahoo_id: AMZN}
  - {symbol: GOOGL, yahoo_id: GOOGL}
  - {symbol: META,  yahoo_id: META}
EOF
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_config.py
import pytest

from cryptoval.config import SECTORS, load_config


def test_loads_settings():
    settings, _ = load_config("config")
    assert settings.trailing_refetch_days == 30
    assert settings.bigquery_monthly_byte_budget == 900_000_000_000
    assert settings.resolver.layer("bronze").endswith("/bronze")


def test_loads_full_universe():
    _, assets = load_config("config")
    crypto = [a for a in assets if not a.is_equity]
    equities = [a for a in assets if a.is_equity]
    assert len(crypto) == 24      # 23 valuation targets plus SOL, which has no CM id
    assert len(equities) == 5


def test_sol_has_no_coinmetrics_id_recording_the_documented_gap():
    _, assets = load_config("config")
    sol = next(a for a in assets if a.symbol == "SOL")
    assert sol.cm_id is None
    assert sol.gecko_id == "solana"


def test_every_sector_is_known():
    _, assets = load_config("config")
    for a in assets:
        if not a.is_equity:
            assert a.sector in SECTORS, f"{a.symbol} has unknown sector {a.sector}"


def test_consensus_drives_cost_of_production_applicability():
    _, assets = load_config("config")
    btc = next(a for a in assets if a.symbol == "BTC")
    eth = next(a for a in assets if a.symbol == "ETH")
    assert btc.consensus == "pow"
    assert eth.consensus == "pos"


def test_usdc_is_flagged_as_the_stablecoin_control():
    _, assets = load_config("config")
    usdc = next(a for a in assets if a.symbol == "USDC")
    assert usdc.is_stablecoin is True


def test_unknown_sector_is_rejected(tmp_path):
    (tmp_path / "settings.yml").write_text("data_root: ./data\n")
    (tmp_path / "assets.yml").write_text(
        "assets:\n  - {symbol: FOO, cm_id: foo, gecko_id: foo, sector: nonsense, consensus: pow}\n"
    )
    with pytest.raises(ValueError, match="unknown sector"):
        load_config(str(tmp_path))


def test_asset_without_any_source_id_is_rejected(tmp_path):
    (tmp_path / "settings.yml").write_text("data_root: ./data\n")
    (tmp_path / "assets.yml").write_text(
        "assets:\n  - {symbol: FOO, cm_id: null, gecko_id: null, sector: defi, consensus: other}\n"
    )
    with pytest.raises(ValueError, match="no usable source id"):
        load_config(str(tmp_path))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.config'`.

- [ ] **Step 4: Write the implementation**

```python
# cryptoval/config.py
"""Configuration and the asset universe, loaded from YAML."""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import yaml

from cryptoval.paths import PathResolver

SECTORS = (
    "pow_monetary",
    "smart_contract_l1",
    "scaling",
    "defi",
    "oracle",
    "exchange",
    "metaverse",
    "stablecoin",
)
CONSENSUS = ("pow", "pos", "other", "na")


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


@dataclass(frozen=True)
class Settings:
    data_root: str = "./data"
    trailing_refetch_days: int = 30
    bigquery_monthly_byte_budget: int = 900_000_000_000
    request_timeout_s: int = 30
    max_retries: int = 5

    @property
    def resolver(self) -> PathResolver:
        return PathResolver(self.data_root)


def _validate(asset: Asset) -> None:
    if asset.is_equity:
        if not asset.yahoo_id:
            raise ValueError(f"{asset.symbol}: equity has no usable source id")
        return
    if asset.sector not in SECTORS:
        raise ValueError(f"{asset.symbol}: unknown sector {asset.sector!r}")
    if asset.consensus not in CONSENSUS:
        raise ValueError(f"{asset.symbol}: unknown consensus {asset.consensus!r}")
    if not asset.cm_id and not asset.gecko_id:
        raise ValueError(f"{asset.symbol}: no usable source id")


def load_config(config_dir: str = "config") -> tuple[Settings, list[Asset]]:
    base = pathlib.Path(config_dir)
    raw_settings = yaml.safe_load((base / "settings.yml").read_text(encoding="utf-8")) or {}
    raw_settings["data_root"] = os.environ.get(
        "CRYPTOVAL_DATA_ROOT", raw_settings.get("data_root", "./data")
    )
    settings = Settings(**raw_settings)

    raw_assets = yaml.safe_load((base / "assets.yml").read_text(encoding="utf-8")) or {}
    assets: list[Asset] = []
    for row in raw_assets.get("assets", []) or []:
        assets.append(Asset(**row))
    for row in raw_assets.get("equities", []) or []:
        assets.append(Asset(is_equity=True, **row))
    for a in assets:
        _validate(a)
    return settings, assets
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/config.py config/settings.yml config/assets.yml tests/test_config.py
git commit -m "feat: add settings and 24-asset universe config with validation"
```

---

## Task 4: Run manifest

**Files:**
- Create: `cryptoval/manifest.py`, `cryptoval/hashing.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `PathResolver` (Task 2).
- Produces: `stable_hash(obj) -> str` (sha256 hex of canonical JSON, used by bronze, SCD2 and the
  DefiLlama methodology guard); `RunRecord` dataclass; `RunManifest(resolver)` with
  `.start(source, params) -> RunRecord`, `.finish(record, rows_in, rows_out, status, bytes_scanned,
  error)`, `.read() -> pandas.DataFrame`, `.bytes_scanned_in_month(year, month) -> int`. The byte
  ledger in Task 12 and the observability requirement in spec §9.4 both read this.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_manifest.py
import datetime as dt

from cryptoval.hashing import stable_hash
from cryptoval.manifest import RunManifest
from cryptoval.paths import PathResolver


def test_stable_hash_is_order_independent_for_dicts():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_stable_hash_changes_with_content():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_records_a_successful_run(tmp_path):
    m = RunManifest(PathResolver(str(tmp_path)))
    rec = m.start("coinmetrics", {"start": "2026-10-01"})
    m.finish(rec, rows_in=10, rows_out=10, status="success")
    df = m.read()
    assert len(df) == 1
    assert df.loc[0, "status"] == "success"
    assert df.loc[0, "source"] == "coinmetrics"
    assert df.loc[0, "rows_out"] == 10


def test_records_a_failed_run_with_the_error(tmp_path):
    m = RunManifest(PathResolver(str(tmp_path)))
    rec = m.start("defillama", {})
    m.finish(rec, rows_in=0, rows_out=0, status="failed", error="HTTP 503")
    df = m.read()
    assert df.loc[0, "status"] == "failed"
    assert "503" in df.loc[0, "error"]


def test_appends_across_runs(tmp_path):
    m = RunManifest(PathResolver(str(tmp_path)))
    for _ in range(3):
        rec = m.start("coingecko", {})
        m.finish(rec, rows_in=1, rows_out=1, status="success")
    assert len(m.read()) == 3


def test_monthly_byte_ledger_sums_only_the_requested_month(tmp_path):
    m = RunManifest(PathResolver(str(tmp_path)))
    r1 = m.start("bigquery", {}, started_at=dt.datetime(2026, 10, 5, tzinfo=dt.UTC))
    m.finish(r1, rows_in=0, rows_out=1, status="success", bytes_scanned=100)
    r2 = m.start("bigquery", {}, started_at=dt.datetime(2026, 10, 9, tzinfo=dt.UTC))
    m.finish(r2, rows_in=0, rows_out=1, status="success", bytes_scanned=250)
    r3 = m.start("bigquery", {}, started_at=dt.datetime(2026, 11, 1, tzinfo=dt.UTC))
    m.finish(r3, rows_in=0, rows_out=1, status="success", bytes_scanned=999)
    assert m.bytes_scanned_in_month(2026, 10) == 350
    assert m.bytes_scanned_in_month(2026, 11) == 999


def test_read_on_empty_manifest_returns_empty_frame(tmp_path):
    m = RunManifest(PathResolver(str(tmp_path)))
    df = m.read()
    assert df.empty
    assert "status" in df.columns
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.hashing'`.

- [ ] **Step 3: Write the hashing helper**

```python
# cryptoval/hashing.py
"""Stable content hashing.

Used for bronze response fingerprints, SCD2 change detection and the
DefiLlama methodology guard. Must be stable across processes and runs, so
Python's salted builtin hash() is unsuitable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Write the manifest implementation**

```python
# cryptoval/manifest.py
"""Append-only record of pipeline runs (spec section 9.4)."""

from __future__ import annotations

import datetime as dt
import pathlib
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from cryptoval.paths import PathResolver

COLUMNS = [
    "run_id", "git_sha", "source", "params", "started_at", "ended_at",
    "rows_in", "rows_out", "bytes_scanned", "status", "error",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class RunRecord:
    run_id: str
    git_sha: str
    source: str
    params: dict[str, Any]
    started_at: dt.datetime
    ended_at: dt.datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RunManifest:
    def __init__(self, resolver: PathResolver) -> None:
        self._path = pathlib.Path(resolver.table("gold", "run_manifest.jsonl"))

    def start(
        self, source: str, params: dict[str, Any], started_at: dt.datetime | None = None
    ) -> RunRecord:
        return RunRecord(
            run_id=str(uuid.uuid4()),
            git_sha=_git_sha(),
            source=source,
            params=params,
            started_at=started_at or dt.datetime.now(dt.UTC),
        )

    def finish(
        self,
        record: RunRecord,
        rows_in: int,
        rows_out: int,
        status: str,
        bytes_scanned: int = 0,
        error: str | None = None,
    ) -> None:
        record.ended_at = dt.datetime.now(dt.UTC)
        row = {
            "run_id": record.run_id,
            "git_sha": record.git_sha,
            "source": record.source,
            "params": record.params,
            "started_at": record.started_at.isoformat(),
            "ended_at": record.ended_at.isoformat(),
            "rows_in": rows_in,
            "rows_out": rows_out,
            "bytes_scanned": bytes_scanned,
            "status": status,
            "error": error,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(pd.Series(row).to_json() + "\n")

    def read(self) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame(columns=COLUMNS)
        df = pd.read_json(self._path, lines=True, convert_dates=False)
        return df.reindex(columns=COLUMNS)

    def bytes_scanned_in_month(self, year: int, month: int) -> int:
        df = self.read()
        if df.empty:
            return 0
        started = pd.to_datetime(df["started_at"], utc=True)
        mask = (started.dt.year == year) & (started.dt.month == month)
        return int(df.loc[mask, "bytes_scanned"].fillna(0).sum())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manifest.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/manifest.py cryptoval/hashing.py tests/test_manifest.py
git commit -m "feat: add run manifest with monthly byte ledger and stable hashing"
```

---

## Task 5: Contract loader and generic quality checks

**Files:**
- Create: `cryptoval/contracts/__init__.py`, `cryptoval/contracts/loader.py`,
  `cryptoval/contracts/checks.py`, `config/contracts/coinmetrics_asset_metrics.yml`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CheckResult(name: str, passed: bool, detail: str)`; `Contract` (with `.name`,
  `.version`, `.required_columns`, `.unique_on`, `.ranges`, `.freshness_max_delay_hours`,
  `.volume_anomaly_threshold`); `load_contract(path) -> Contract`; and the check functions
  `check_required_columns`, `check_unique`, `check_not_null`, `check_ranges`, `check_freshness`,
  `check_volume_anomaly`, plus `run_checks(df, contract, now) -> list[CheckResult]`. Task 14's
  `validate` command fails the build when any result has `passed=False`.

- [ ] **Step 1: Write the Coin Metrics contract**

```bash
mkdir -p config/contracts
cat > config/contracts/coinmetrics_asset_metrics.yml <<'EOF'
contract:
  name: coinmetrics_asset_metrics
  version: "1.0.0"
  owner: yasir
  source_url: https://community-api.coinmetrics.io/v4

schema:
  required:
    - asset
    - metric_date
    - CapMrktCurUSD
  columns:
    asset: {type: string}
    metric_date: {type: date}
    CapMrktCurUSD: {type: float, minimum: 0}
    CapMVRVCur: {type: float, minimum: 0}
    HashRate: {type: float, minimum: 0}
    IssTotUSD: {type: float, minimum: 0}
    FeeTotNtv: {type: float, minimum: 0}
    AdrActCnt: {type: float, minimum: 0}
    TxCnt: {type: float, minimum: 0}
    SplyCur: {type: float, minimum: 0}
    PriceUSD: {type: float, minimum: 0}

quality:
  freshness:
    max_delay_hours: 48          # published T+1 after UTC close, 48h allows for a late day
  uniqueness:
    combination: [asset, metric_date]
  volume:
    anomaly_threshold: 0.5       # +/-50% vs trailing average (spec section 9.2)

versioning:
  breaking_changes: [removing_required_field, changing_field_type, renaming_field]
  non_breaking_changes: [adding_optional_field, changing_description]
EOF
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_contracts.py
import datetime as dt

import pandas as pd
import pytest

from cryptoval.contracts.checks import (
    check_freshness,
    check_not_null,
    check_ranges,
    check_required_columns,
    check_unique,
    check_volume_anomaly,
    run_checks,
)
from cryptoval.contracts.loader import load_contract

CONTRACT = "config/contracts/coinmetrics_asset_metrics.yml"


def _frame(**overrides):
    base = pd.DataFrame(
        {
            "asset": ["btc", "btc"],
            "metric_date": pd.to_datetime(["2026-10-10", "2026-10-11"], utc=True),
            "CapMrktCurUSD": [1.0e12, 1.1e12],
        }
    )
    for k, v in overrides.items():
        base[k] = v
    return base


def test_loads_contract_metadata():
    c = load_contract(CONTRACT)
    assert c.name == "coinmetrics_asset_metrics"
    assert c.version == "1.0.0"
    assert c.unique_on == ["asset", "metric_date"]
    assert c.freshness_max_delay_hours == 48
    assert c.volume_anomaly_threshold == 0.5
    assert "CapMrktCurUSD" in c.required_columns


def test_required_columns_pass_and_fail():
    c = load_contract(CONTRACT)
    assert check_required_columns(_frame(), c).passed
    bad = _frame().drop(columns=["CapMrktCurUSD"])
    res = check_required_columns(bad, c)
    assert not res.passed and "CapMrktCurUSD" in res.detail


def test_uniqueness_detects_duplicate_keys():
    c = load_contract(CONTRACT)
    assert check_unique(_frame(), c).passed
    dup = pd.concat([_frame(), _frame()], ignore_index=True)
    res = check_unique(dup, c)
    assert not res.passed and "2" in res.detail


def test_not_null_on_required_columns():
    c = load_contract(CONTRACT)
    bad = _frame()
    bad.loc[0, "CapMrktCurUSD"] = None
    assert not check_not_null(bad, c).passed


def test_ranges_reject_negative_market_cap():
    c = load_contract(CONTRACT)
    bad = _frame(CapMrktCurUSD=[-1.0, 1.0e12])
    res = check_ranges(bad, c)
    assert not res.passed and "CapMrktCurUSD" in res.detail


def test_freshness_passes_when_recent_and_fails_when_stale():
    c = load_contract(CONTRACT)
    now = dt.datetime(2026, 10, 11, 12, tzinfo=dt.UTC)
    assert check_freshness(_frame(), c, "metric_date", now).passed
    stale = _frame()
    stale["metric_date"] = pd.to_datetime(["2026-09-01", "2026-09-02"], utc=True)
    assert not check_freshness(stale, c, "metric_date", now).passed


def test_volume_anomaly_flags_a_tenfold_jump():
    c = load_contract(CONTRACT)
    steady = pd.DataFrame(
        {
            "metric_date": pd.to_datetime(
                [f"2026-09-{d:02d}" for d in range(1, 21)], utc=True
            ),
            "value": [100.0] * 19 + [1000.0],
        }
    )
    res = check_volume_anomaly(steady, c, "value", "metric_date")
    assert not res.passed
    assert "1000" in res.detail or "anomal" in res.detail.lower()


def test_volume_anomaly_allows_normal_variation():
    c = load_contract(CONTRACT)
    steady = pd.DataFrame(
        {
            "metric_date": pd.to_datetime(
                [f"2026-09-{d:02d}" for d in range(1, 21)], utc=True
            ),
            "value": [100.0, 105.0, 98.0] * 6 + [102.0, 99.0],
        }
    )
    assert check_volume_anomaly(steady, c, "value", "metric_date").passed


def test_run_checks_aggregates_all_results():
    c = load_contract(CONTRACT)
    results = run_checks(_frame(), c, now=dt.datetime(2026, 10, 11, 12, tzinfo=dt.UTC))
    assert len(results) >= 4
    assert all(r.passed for r in results)


def test_unknown_contract_path_raises():
    with pytest.raises(FileNotFoundError):
        load_contract("config/contracts/does_not_exist.yml")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_contracts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.contracts'`.

- [ ] **Step 4: Write the loader**

```python
# cryptoval/contracts/__init__.py
"""Data contracts: schema plus quality SLAs per source (spec section 9)."""
```

```python
# cryptoval/contracts/loader.py
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class Contract:
    name: str
    version: str
    required_columns: list[str]
    columns: dict[str, dict[str, Any]] = field(default_factory=dict)
    unique_on: list[str] = field(default_factory=list)
    freshness_max_delay_hours: int | None = None
    volume_anomaly_threshold: float | None = None

    @property
    def ranges(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for col, spec in self.columns.items():
            bounds = {k: spec[k] for k in ("minimum", "maximum") if k in spec}
            if bounds:
                out[col] = bounds
        return out


def load_contract(path: str) -> Contract:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"contract not found: {path}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    meta = raw.get("contract", {})
    schema = raw.get("schema", {})
    quality = raw.get("quality", {})
    return Contract(
        name=meta.get("name", p.stem),
        version=meta.get("version", "0.0.0"),
        required_columns=list(schema.get("required", []) or []),
        columns=dict(schema.get("columns", {}) or {}),
        unique_on=list((quality.get("uniqueness", {}) or {}).get("combination", []) or []),
        freshness_max_delay_hours=(quality.get("freshness", {}) or {}).get("max_delay_hours"),
        volume_anomaly_threshold=(quality.get("volume", {}) or {}).get("anomaly_threshold"),
    )
```

- [ ] **Step 5: Write the checks**

```python
# cryptoval/contracts/checks.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from cryptoval.contracts.loader import Contract


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def check_required_columns(df: pd.DataFrame, contract: Contract) -> CheckResult:
    missing = [c for c in contract.required_columns if c not in df.columns]
    return CheckResult(
        "required_columns", not missing, "" if not missing else f"missing columns: {missing}"
    )


def check_unique(df: pd.DataFrame, contract: Contract) -> CheckResult:
    keys = [k for k in contract.unique_on if k in df.columns]
    if not keys:
        return CheckResult("uniqueness", True, "no key columns present")
    n_dupes = int(df.duplicated(subset=keys).sum())
    return CheckResult(
        "uniqueness", n_dupes == 0, "" if n_dupes == 0 else f"{n_dupes} duplicate rows on {keys}"
    )


def check_not_null(df: pd.DataFrame, contract: Contract) -> CheckResult:
    offenders = {
        c: int(df[c].isna().sum())
        for c in contract.required_columns
        if c in df.columns and df[c].isna().any()
    }
    return CheckResult("not_null", not offenders, "" if not offenders else f"nulls: {offenders}")


def check_ranges(df: pd.DataFrame, contract: Contract) -> CheckResult:
    offenders = []
    for col, bounds in contract.ranges.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if "minimum" in bounds and (series < bounds["minimum"]).any():
            offenders.append(f"{col} < {bounds['minimum']}")
        if "maximum" in bounds and (series > bounds["maximum"]).any():
            offenders.append(f"{col} > {bounds['maximum']}")
    return CheckResult("ranges", not offenders, "; ".join(offenders))


def check_freshness(
    df: pd.DataFrame, contract: Contract, date_col: str, now: dt.datetime
) -> CheckResult:
    if contract.freshness_max_delay_hours is None or df.empty:
        return CheckResult("freshness", True, "no SLA or no rows")
    latest = pd.to_datetime(df[date_col], utc=True).max()
    delay_h = (now - latest.to_pydatetime()).total_seconds() / 3600
    ok = delay_h <= contract.freshness_max_delay_hours
    return CheckResult(
        "freshness", ok, "" if ok else f"latest {latest.date()} is {delay_h:.1f}h old"
    )


def check_volume_anomaly(
    df: pd.DataFrame, contract: Contract, value_col: str, date_col: str
) -> CheckResult:
    threshold = contract.volume_anomaly_threshold
    if threshold is None or len(df) < 5:
        return CheckResult("volume_anomaly", True, "no SLA or too few rows")
    ordered = df.sort_values(date_col)
    values = pd.to_numeric(ordered[value_col], errors="coerce")
    trailing = values.shift(1).expanding(min_periods=3).mean()
    # trailing means of zero would give an infinite deviation; treat them as unknown.
    safe_trailing = trailing.where(trailing != 0)
    deviation = (values - safe_trailing).abs() / safe_trailing
    worst = deviation.max()
    if pd.isna(worst) or worst <= threshold:
        return CheckResult("volume_anomaly", True, "")
    idx = deviation.idxmax()
    return CheckResult(
        "volume_anomaly",
        False,
        f"anomalous {value_col}={values.loc[idx]:.0f} deviates {worst:.1%} from trailing mean",
    )


def run_checks(df: pd.DataFrame, contract: Contract, now: dt.datetime) -> list[CheckResult]:
    results = [
        check_required_columns(df, contract),
        check_unique(df, contract),
        check_not_null(df, contract),
        check_ranges(df, contract),
    ]
    if "metric_date" in df.columns:
        results.append(check_freshness(df, contract, "metric_date", now))
    return results
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_contracts.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/contracts config/contracts tests/test_contracts.py
git commit -m "feat: add data contract loader and generic quality checks"
```

---

## Task 6: Identity assertions

**Files:**
- Create: `cryptoval/contracts/identities.py`
- Test: `tests/test_identities.py`

**Interfaces:**
- Consumes: `CheckResult` (Task 5).
- Produces: `check_mvrv_identity(df, rtol)`, `check_supply_identity(df, rtol)`,
  `check_monotonic_nondecreasing(df, col, group_col)`, `check_cointime_complement(df, rtol)`, and
  `run_identity_checks(df) -> list[CheckResult]`. Spec §9.2 calls these "the single most important
  test in the project" because realized cap is a derived quantity; Plan 3 reuses
  `check_cointime_complement` for the Cointime family.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_identities.py
import numpy as np
import pandas as pd

from cryptoval.contracts.identities import (
    check_cointime_complement,
    check_monotonic_nondecreasing,
    check_mvrv_identity,
    check_supply_identity,
    run_identity_checks,
)


def test_mvrv_identity_holds_for_consistent_rows():
    df = pd.DataFrame(
        {
            "CapMrktCurUSD": [1.725e12],
            "CapMVRVCur": [1.5967],
            "realized_cap": [1.725e12 / 1.5967],
        }
    )
    assert check_mvrv_identity(df).passed


def test_mvrv_identity_fails_when_realized_cap_is_inconsistent():
    df = pd.DataFrame(
        {"CapMrktCurUSD": [1.0e12], "CapMVRVCur": [2.0], "realized_cap": [9.9e11]}
    )
    res = check_mvrv_identity(df)
    assert not res.passed
    assert "realized_cap" in res.detail


def test_mvrv_identity_ignores_null_rows_rather_than_failing():
    df = pd.DataFrame(
        {
            "CapMrktCurUSD": [1.0e12, 1.0e12],
            "CapMVRVCur": [2.0, np.nan],
            "realized_cap": [5.0e11, np.nan],
        }
    )
    assert check_mvrv_identity(df).passed


def test_supply_identity_holds_and_fails():
    good = pd.DataFrame({"SplyCur": [19_000_000.0], "PriceUSD": [100.0],
                         "CapMrktCurUSD": [1.9e9]})
    assert check_supply_identity(good).passed
    bad = pd.DataFrame({"SplyCur": [19_000_000.0], "PriceUSD": [100.0],
                        "CapMrktCurUSD": [1.0e9]})
    assert not check_supply_identity(bad).passed


def test_monotonic_check_passes_for_nondecreasing_series_per_asset():
    df = pd.DataFrame(
        {
            "asset": ["btc", "btc", "eth", "eth"],
            "metric_date": pd.to_datetime(
                ["2026-10-01", "2026-10-02", "2026-10-01", "2026-10-02"], utc=True
            ),
            "thermocap": [100.0, 110.0, 50.0, 55.0],
        }
    )
    assert check_monotonic_nondecreasing(df, "thermocap").passed


def test_monotonic_check_fails_when_a_series_decreases():
    df = pd.DataFrame(
        {
            "asset": ["btc", "btc"],
            "metric_date": pd.to_datetime(["2026-10-01", "2026-10-02"], utc=True),
            "thermocap": [110.0, 100.0],
        }
    )
    res = check_monotonic_nondecreasing(df, "thermocap")
    assert not res.passed
    assert "btc" in res.detail


def test_monotonic_check_is_not_fooled_by_unsorted_input():
    df = pd.DataFrame(
        {
            "asset": ["btc", "btc"],
            "metric_date": pd.to_datetime(["2026-10-02", "2026-10-01"], utc=True),
            "thermocap": [110.0, 100.0],
        }
    )
    assert check_monotonic_nondecreasing(df, "thermocap").passed


def test_cointime_complement_requires_liveliness_plus_vaultedness_to_equal_one():
    good = pd.DataFrame({"liveliness": [0.4], "vaultedness": [0.6]})
    assert check_cointime_complement(good).passed
    bad = pd.DataFrame({"liveliness": [0.4], "vaultedness": [0.5]})
    assert not check_cointime_complement(bad).passed


def test_run_identity_checks_skips_checks_whose_columns_are_absent():
    df = pd.DataFrame({"CapMrktCurUSD": [1.0e12]})
    results = run_identity_checks(df)
    assert all(r.passed for r in results)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_identities.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.contracts.identities'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/contracts/identities.py
"""Cross-column identity assertions (spec section 9.2).

Realized capitalisation is derived, not sourced: realized_cap =
CapMrktCurUSD / CapMVRVCur. If either input is redefined upstream, every
downstream cost-approach metric is silently wrong. These checks are the
tripwire for that, and they are blocking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.contracts.checks import CheckResult

DEFAULT_RTOL = 1e-6


def _relative_mismatch(
    left: pd.Series, right: pd.Series, rtol: float
) -> tuple[int, float]:
    both = pd.concat([left, right], axis=1).dropna()
    if both.empty:
        return 0, 0.0
    a = both.iloc[:, 0].to_numpy(dtype=float)
    b = both.iloc[:, 1].to_numpy(dtype=float)
    denom = np.where(np.abs(b) > 0, np.abs(b), np.nan)
    rel = np.abs(a - b) / denom
    rel = rel[~np.isnan(rel)]
    if rel.size == 0:
        return 0, 0.0
    return int((rel > rtol).sum()), float(rel.max())


def check_mvrv_identity(df: pd.DataFrame, rtol: float = DEFAULT_RTOL) -> CheckResult:
    needed = {"CapMrktCurUSD", "CapMVRVCur", "realized_cap"}
    if not needed.issubset(df.columns):
        return CheckResult("identity_mvrv", True, "columns absent")
    mvrv = pd.to_numeric(df["CapMVRVCur"], errors="coerce").where(lambda s: s > 0)
    implied = pd.to_numeric(df["CapMrktCurUSD"], errors="coerce") / mvrv
    n_bad, worst = _relative_mismatch(
        pd.to_numeric(df["realized_cap"], errors="coerce"), implied, rtol
    )
    return CheckResult(
        "identity_mvrv",
        n_bad == 0,
        ""
        if n_bad == 0
        else f"{n_bad} rows where realized_cap != market_cap/mvrv (worst {worst:.2e})",
    )


def check_supply_identity(df: pd.DataFrame, rtol: float = 1e-3) -> CheckResult:
    needed = {"SplyCur", "PriceUSD", "CapMrktCurUSD"}
    if not needed.issubset(df.columns):
        return CheckResult("identity_supply", True, "columns absent")
    implied = pd.to_numeric(df["SplyCur"], errors="coerce") * pd.to_numeric(
        df["PriceUSD"], errors="coerce"
    )
    n_bad, worst = _relative_mismatch(
        pd.to_numeric(df["CapMrktCurUSD"], errors="coerce"), implied, rtol
    )
    return CheckResult(
        "identity_supply",
        n_bad == 0,
        "" if n_bad == 0 else f"{n_bad} rows where supply*price != market_cap (worst {worst:.2e})",
    )


def check_monotonic_nondecreasing(
    df: pd.DataFrame, col: str, group_col: str = "asset", date_col: str = "metric_date"
) -> CheckResult:
    if col not in df.columns or group_col not in df.columns:
        return CheckResult(f"monotonic_{col}", True, "columns absent")
    offenders = []
    for key, grp in df.sort_values(date_col).groupby(group_col):
        series = pd.to_numeric(grp[col], errors="coerce").dropna()
        if (series.diff().dropna() < 0).any():
            offenders.append(str(key))
    return CheckResult(
        f"monotonic_{col}",
        not offenders,
        "" if not offenders else f"{col} decreases for: {offenders}",
    )


def check_cointime_complement(df: pd.DataFrame, rtol: float = 1e-9) -> CheckResult:
    needed = {"liveliness", "vaultedness"}
    if not needed.issubset(df.columns):
        return CheckResult("identity_cointime", True, "columns absent")
    total = pd.to_numeric(df["liveliness"], errors="coerce") + pd.to_numeric(
        df["vaultedness"], errors="coerce"
    )
    n_bad, worst = _relative_mismatch(total, pd.Series(1.0, index=df.index), rtol)
    return CheckResult(
        "identity_cointime",
        n_bad == 0,
        "" if n_bad == 0 else f"{n_bad} rows where liveliness+vaultedness != 1 (worst {worst:.2e})",
    )


def run_identity_checks(df: pd.DataFrame) -> list[CheckResult]:
    return [
        check_mvrv_identity(df),
        check_supply_identity(df),
        check_monotonic_nondecreasing(df, "thermocap"),
        check_cointime_complement(df),
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_identities.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/contracts/identities.py tests/test_identities.py
git commit -m "feat: add blocking identity assertions for derived realized cap"
```

---

## Task 7: Atomic bronze writer

**Files:**
- Create: `cryptoval/io/__init__.py`, `cryptoval/io/bronze.py`
- Test: `tests/test_bronze.py`

**Interfaces:**
- Consumes: `PathResolver` (Task 2), `stable_hash` (Task 4).
- Produces: `write_bronze(resolver, source, ingest_date, df, request_params) -> str` returning the
  partition path, and `read_bronze(resolver, source, ingest_date) -> pandas.DataFrame`. Writes
  atomically via a temporary directory then rename, so a failed fetch never leaves a partition that
  looks complete (**Review Focus 5**). Adds `_fetched_at`, `_source`, `_request_hash` columns.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bronze.py
import datetime as dt
import pathlib

import pandas as pd
import pytest

from cryptoval.io.bronze import read_bronze, write_bronze
from cryptoval.paths import PathResolver

DAY = dt.date(2026, 10, 11)


def _df():
    return pd.DataFrame({"asset": ["btc"], "value": [1.0]})


def test_writes_partition_and_round_trips(tmp_path):
    r = PathResolver(str(tmp_path))
    path = write_bronze(r, "coinmetrics", DAY, _df(), {"start": "2026-10-11"})
    assert pathlib.Path(path).exists()
    back = read_bronze(r, "coinmetrics", DAY)
    assert len(back) == 1
    assert back.loc[0, "asset"] == "btc"


def test_adds_provenance_columns(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "coinmetrics", DAY, _df(), {"start": "2026-10-11"})
    back = read_bronze(r, "coinmetrics", DAY)
    assert "_fetched_at" in back.columns
    assert back.loc[0, "_source"] == "coinmetrics"
    assert len(back.loc[0, "_request_hash"]) == 64


def test_identical_params_produce_identical_request_hash(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, _df(), {"a": 1, "b": 2})
    first = read_bronze(r, "s", DAY).loc[0, "_request_hash"]
    write_bronze(r, "s", DAY, _df(), {"b": 2, "a": 1})
    assert read_bronze(r, "s", DAY).loc[0, "_request_hash"] == first


def test_rewriting_a_date_replaces_the_partition_wholesale(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, pd.DataFrame({"asset": ["btc"], "value": [1.0]}), {})
    write_bronze(r, "s", DAY, pd.DataFrame({"asset": ["eth"], "value": [2.0]}), {})
    back = read_bronze(r, "s", DAY)
    assert len(back) == 1
    assert back.loc[0, "asset"] == "eth"


def test_other_dates_are_untouched_by_a_rewrite(tmp_path):
    r = PathResolver(str(tmp_path))
    other = dt.date(2026, 10, 10)
    write_bronze(r, "s", other, _df(), {})
    write_bronze(r, "s", DAY, _df(), {})
    write_bronze(r, "s", DAY, _df(), {})
    assert len(read_bronze(r, "s", other)) == 1


def test_empty_frame_is_rejected_so_a_failed_fetch_cannot_look_complete(tmp_path):
    r = PathResolver(str(tmp_path))
    with pytest.raises(ValueError, match="refusing to write an empty"):
        write_bronze(r, "s", DAY, pd.DataFrame(), {})
    with pytest.raises(FileNotFoundError):
        read_bronze(r, "s", DAY)


def test_no_temporary_directory_is_left_behind(tmp_path):
    r = PathResolver(str(tmp_path))
    write_bronze(r, "s", DAY, _df(), {})
    leftovers = [p.name for p in pathlib.Path(tmp_path).rglob("*.tmp*")]
    assert leftovers == []


def test_remote_root_is_rejected_with_a_clear_message(tmp_path):
    r = PathResolver("gs://bucket/root")
    with pytest.raises(NotImplementedError, match="object storage"):
        write_bronze(r, "s", DAY, _df(), {})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bronze.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.io'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/io/__init__.py
"""Storage adapters. These know about files and nothing about vendors."""
```

```python
# cryptoval/io/bronze.py
"""Immutable bronze landing zone.

Writes are atomic: data lands in a sibling temporary directory and is only
renamed into place once the whole frame is written. A fetch that dies
half-way therefore leaves no partition at all, rather than a partial one
that later runs would mistake for complete (Review Focus 5).
"""

from __future__ import annotations

import datetime as dt
import pathlib
import shutil

import pandas as pd

from cryptoval.hashing import stable_hash
from cryptoval.paths import PathResolver


def write_bronze(
    resolver: PathResolver,
    source: str,
    ingest_date: dt.date,
    df: pd.DataFrame,
    request_params: dict,
) -> str:
    if resolver.is_remote:
        raise NotImplementedError(
            "bronze writes to object storage are added in Plan 5; "
            "set a local CRYPTOVAL_DATA_ROOT for now"
        )
    if df is None or df.empty:
        raise ValueError(
            f"refusing to write an empty bronze partition for {source} {ingest_date}: "
            "an empty frame means the fetch failed"
        )

    enriched = df.copy()
    enriched["_fetched_at"] = pd.Timestamp.now(tz="UTC")
    enriched["_source"] = source
    enriched["_request_hash"] = stable_hash(request_params)

    final = pathlib.Path(resolver.bronze_partition(source, ingest_date))
    staging = final.with_name(final.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    enriched.to_parquet(staging / "data.parquet", index=False)

    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    return str(final)


def read_bronze(resolver: PathResolver, source: str, ingest_date: dt.date) -> pd.DataFrame:
    path = pathlib.Path(resolver.bronze_partition(source, ingest_date)) / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no bronze partition at {path}")
    return pd.read_parquet(path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bronze.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/io tests/test_bronze.py
git commit -m "feat: add atomic bronze partition writer with provenance columns"
```

---

## Task 8: Source protocol and the Coin Metrics adapter

**Files:**
- Create: `cryptoval/sources/__init__.py`, `cryptoval/sources/base.py`,
  `cryptoval/sources/coinmetrics.py`, `tests/fixtures/coinmetrics_btc_eth.json`,
  `tests/fixtures/coinmetrics_null_metrics.json`, `tests/conftest.py`
- Test: `tests/test_coinmetrics.py`

**Interfaces:**
- Consumes: `Asset`, `Settings` (Task 3).
- Produces: `FetchResult(frame: pandas.DataFrame, rows_in: int)`; the `Source` protocol with
  `.name: str`, `.contract_path: str`, `.fetch(start, end, assets, client) -> FetchResult`;
  `CoinMetricsSource`; `FREE_METRICS: tuple[str, ...]`; `parse_timeseries(payload) -> DataFrame`;
  `derive_realized_cap(df) -> DataFrame` adding a `realized_cap` column. This is the adapter every
  later plan depends on most.

**Review Focus coverage:** items 1 (MVRV zero/null/negative) and 3 (nulls not coerced to zero).

- [ ] **Step 1: Write the fixtures and conftest**

Fixtures are trimmed real responses, so unit tests never touch the network.

```bash
mkdir -p tests/fixtures
cat > tests/fixtures/coinmetrics_btc_eth.json <<'EOF'
{"data":[
 {"asset":"btc","time":"2026-10-05T00:00:00.000000000Z","CapMrktCurUSD":"1725000000000","CapMVRVCur":"1.5967","PriceUSD":"87000","SplyCur":"19827586.2","HashRate":"1167730645.36","IssTotUSD":"28000000","FeeTotNtv":"12.5","AdrActCnt":"660864","TxCnt":"420000"},
 {"asset":"btc","time":"2026-10-06T00:00:00.000000000Z","CapMrktCurUSD":"1719000000000","CapMVRVCur":"1.5905","PriceUSD":"86700","SplyCur":"19827700.0","HashRate":"1022588981.51","IssTotUSD":"27800000","FeeTotNtv":"11.9","AdrActCnt":"636796","TxCnt":"410000"},
 {"asset":"eth","time":"2026-10-05T00:00:00.000000000Z","CapMrktCurUSD":"331300000000","CapMVRVCur":"1.1508","PriceUSD":"2740","SplyCur":"120912408.0","IssTotUSD":"2400000","FeeTotNtv":"310.4","AdrActCnt":"647739","TxCnt":"1250000"}
]}
EOF

cat > tests/fixtures/coinmetrics_null_metrics.json <<'EOF'
{"data":[
 {"asset":"eth","time":"2026-10-05T00:00:00.000000000Z","CapMrktCurUSD":"331300000000","CapMVRVCur":null,"PriceUSD":"2740","HashRate":null},
 {"asset":"xyz","time":"2026-10-05T00:00:00.000000000Z","CapMrktCurUSD":"1000000","CapMVRVCur":"0","PriceUSD":"1.0"},
 {"asset":"neg","time":"2026-10-05T00:00:00.000000000Z","CapMrktCurUSD":"1000000","CapMVRVCur":"-0.5","PriceUSD":"1.0"}
]}
EOF

cat > tests/conftest.py <<'EOF'
import json
import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture():
    def _load(name: str):
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    return _load
EOF
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_coinmetrics.py
import numpy as np
import pandas as pd
import pytest

from cryptoval.sources.coinmetrics import (
    FREE_METRICS,
    CoinMetricsSource,
    derive_realized_cap,
    parse_timeseries,
)


def test_free_metrics_include_the_load_bearing_series():
    for m in ("CapMrktCurUSD", "CapMVRVCur", "HashRate", "IssTotUSD", "FeeTotNtv", "AdrActCnt"):
        assert m in FREE_METRICS


def test_free_metrics_exclude_paid_realized_cap():
    assert "CapRealUSD" not in FREE_METRICS


def test_parses_payload_into_a_typed_panel(load_fixture):
    df = parse_timeseries(load_fixture("coinmetrics_btc_eth.json"))
    assert len(df) == 3
    assert set(df["asset"]) == {"btc", "eth"}
    assert pd.api.types.is_numeric_dtype(df["CapMrktCurUSD"])
    assert df["CapMrktCurUSD"].iloc[0] == pytest.approx(1.725e12)


def test_metric_date_is_timezone_aware_utc(load_fixture):
    df = parse_timeseries(load_fixture("coinmetrics_btc_eth.json"))
    assert str(df["metric_date"].dt.tz) == "UTC"
    assert df["metric_date"].iloc[0] == pd.Timestamp("2026-10-05", tz="UTC")


def test_derives_realized_cap_from_the_mvrv_identity(load_fixture):
    df = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_btc_eth.json")))
    btc = df[(df["asset"] == "btc") & (df["metric_date"] == pd.Timestamp("2026-10-05", tz="UTC"))]
    assert btc["realized_cap"].iloc[0] == pytest.approx(1.725e12 / 1.5967)


def test_zero_mvrv_yields_nan_not_inf(load_fixture):
    df = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_null_metrics.json")))
    row = df[df["asset"] == "xyz"]
    assert np.isnan(row["realized_cap"].iloc[0])
    assert not np.isinf(row["realized_cap"].iloc[0])


def test_negative_mvrv_yields_nan(load_fixture):
    df = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_null_metrics.json")))
    row = df[df["asset"] == "neg"]
    assert np.isnan(row["realized_cap"].iloc[0])


def test_null_mvrv_yields_nan(load_fixture):
    df = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_null_metrics.json")))
    row = df[df["asset"] == "eth"]
    assert np.isnan(row["realized_cap"].iloc[0])


def test_null_metrics_are_not_coerced_to_zero(load_fixture):
    df = parse_timeseries(load_fixture("coinmetrics_null_metrics.json"))
    eth = df[df["asset"] == "eth"]
    assert eth["HashRate"].isna().all()
    assert (eth["HashRate"] == 0).sum() == 0


def test_missing_metric_column_is_absent_rather_than_fabricated(load_fixture):
    df = parse_timeseries(load_fixture("coinmetrics_null_metrics.json"))
    assert "AdrActCnt" not in df.columns or df["AdrActCnt"].isna().all()


def test_source_exposes_name_and_contract_path():
    s = CoinMetricsSource()
    assert s.name == "coinmetrics"
    assert s.contract_path.endswith("coinmetrics_asset_metrics.yml")


def test_derive_realized_cap_is_idempotent(load_fixture):
    once = derive_realized_cap(parse_timeseries(load_fixture("coinmetrics_btc_eth.json")))
    twice = derive_realized_cap(once)
    pd.testing.assert_series_equal(once["realized_cap"], twice["realized_cap"])
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_coinmetrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.sources'`.

- [ ] **Step 4: Write the source protocol**

```python
# cryptoval/sources/__init__.py
"""Vendor adapters. These know about HTTP and payload shapes, never about storage."""
```

```python
# cryptoval/sources/base.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
import pandas as pd

from cryptoval.config import Asset


@dataclass(frozen=True)
class FetchResult:
    frame: pd.DataFrame
    rows_in: int


@runtime_checkable
class Source(Protocol):
    name: str
    contract_path: str

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult: ...
```

- [ ] **Step 5: Write the Coin Metrics adapter**

```python
# cryptoval/sources/coinmetrics.py
"""Coin Metrics Community API adapter.

Realized capitalisation is NOT free (CapRealUSD returns 403 on the
community tier), but CapMVRVCur is, and MVRV is market cap over realized
cap by definition. So realized_cap is derived from that identity, and
cryptoval.contracts.identities guards the derivation (spec section 6.2).
"""

from __future__ import annotations

import datetime as dt
import time

import httpx
import numpy as np
import pandas as pd

from cryptoval.config import Asset
from cryptoval.sources.base import FetchResult

BASE_URL = "https://community-api.coinmetrics.io/v4"

FREE_METRICS: tuple[str, ...] = (
    "CapMrktCurUSD",
    "CapMVRVCur",
    "PriceUSD",
    "SplyCur",
    "HashRate",
    "IssTotUSD",
    "FeeTotNtv",
    "AdrActCnt",
    "AdrBalCnt",
    "TxCnt",
    "TxTfrCnt",
    "FlowInExUSD",
    "FlowOutExUSD",
)

_NON_METRIC_COLUMNS = {"asset", "time", "metric_date"}


def parse_timeseries(payload: dict) -> pd.DataFrame:
    rows = payload.get("data", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["asset", "metric_date"])
    df["metric_date"] = pd.to_datetime(df["time"], utc=True).dt.normalize()
    df = df.drop(columns=["time"])
    for col in df.columns:
        if col not in _NON_METRIC_COLUMNS:
            # errors="coerce" turns genuine nulls into NaN; it must never
            # turn them into 0.0 (Review Focus 3).
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def derive_realized_cap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not {"CapMrktCurUSD", "CapMVRVCur"}.issubset(out.columns):
        out["realized_cap"] = np.nan
        return out
    mvrv = pd.to_numeric(out["CapMVRVCur"], errors="coerce")
    # Zero, negative and null MVRV are all meaningless: yield NaN, never inf
    # and never a silently-wrong value (Review Focus 1).
    safe_mvrv = mvrv.where(mvrv > 0)
    out["realized_cap"] = pd.to_numeric(out["CapMrktCurUSD"], errors="coerce") / safe_mvrv
    return out


class CoinMetricsSource:
    name = "coinmetrics"
    contract_path = "config/contracts/coinmetrics_asset_metrics.yml"

    def __init__(self, max_retries: int = 5, page_size: int = 10_000) -> None:
        self.max_retries = max_retries
        self.page_size = page_size

    def _get(self, client: httpx.Client, url: str, params: dict | None) -> dict:
        delay = 1.0
        for attempt in range(self.max_retries):
            response = client.get(url, params=params)
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == self.max_retries - 1:
                    response.raise_for_status()
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("unreachable")

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult:
        ids = [a.cm_id for a in assets if a.cm_id]
        if not ids:
            return FetchResult(pd.DataFrame(columns=["asset", "metric_date"]), 0)

        frames: list[pd.DataFrame] = []
        rows_in = 0
        url = f"{BASE_URL}/timeseries/asset-metrics"
        params: dict | None = {
            "assets": ",".join(ids),
            "metrics": ",".join(FREE_METRICS),
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "frequency": "1d",
            "page_size": self.page_size,
        }
        next_url: str | None = url
        while next_url:
            payload = self._get(client, next_url, params)
            rows_in += len(payload.get("data", []) or [])
            frames.append(parse_timeseries(payload))
            next_url = payload.get("next_page_url")
            params = None  # next_page_url already carries the query string

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if combined.empty:
            return FetchResult(pd.DataFrame(columns=["asset", "metric_date"]), rows_in)
        return FetchResult(derive_realized_cap(combined), rows_in)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_coinmetrics.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 7: Commit**

```bash
git add cryptoval/sources tests/conftest.py tests/fixtures tests/test_coinmetrics.py
git commit -m "feat: add Coin Metrics source with guarded realized-cap derivation"
```

---

## Task 9: DefiLlama adapter with methodology guard

**Files:**
- Create: `cryptoval/sources/defillama.py`, `config/contracts/defillama_fees.yml`,
  `tests/fixtures/defillama_fees_overview.json`
- Test: `tests/test_defillama.py`

**Interfaces:**
- Consumes: `FetchResult` (Task 8), `stable_hash` (Task 4).
- Produces: `DefiLlamaSource`; `parse_fees_overview(payload) -> DataFrame` with columns
  `protocol`, `total_fees_24h`, `total_fees_30d`, `net_revenue_24h`, `methodology_hash`;
  `methodology_hash(protocol_record) -> str`. Plan 2's DCF and P/F methods consume
  `net_revenue_24h` and must never silently fall back to `total_fees_24h` — spec §10.4 denominator
  discipline.

- [ ] **Step 1: Write the contract and fixture**

```bash
cat > config/contracts/defillama_fees.yml <<'EOF'
contract:
  name: defillama_fees
  version: "1.0.0"
  owner: yasir
  source_url: https://api.llama.fi

schema:
  required:
    - protocol
    - total_fees_24h
  columns:
    protocol: {type: string}
    total_fees_24h: {type: float, minimum: 0}
    total_fees_30d: {type: float, minimum: 0}
    net_revenue_24h: {type: float, minimum: 0}
    methodology_hash: {type: string}

quality:
  uniqueness:
    combination: [protocol]
  volume:
    anomaly_threshold: 0.5
  notes: |
    methodology_hash fingerprints DefiLlama's own per-protocol methodology
    text. A change to it means the revenue series has been redefined and the
    history is no longer comparable (spec section 9.2).
EOF

cat > tests/fixtures/defillama_fees_overview.json <<'EOF'
{"protocols":[
 {"name":"Curve DEX","total24h":7415,"total30d":3447578,"revenue24h":1483,"methodology":{"Fees":"Trading fees paid by users","Revenue":"Share to veCRV holders"}},
 {"name":"Aave V2","total24h":5266,"total30d":151852,"revenue24h":null,"methodology":{"Fees":"Interest paid by borrowers"}},
 {"name":"WBTC","total24h":0,"total30d":0.31,"revenue24h":0,"methodology":{}}
]}
EOF
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_defillama.py
import numpy as np

from cryptoval.sources.defillama import (
    DefiLlamaSource,
    methodology_hash,
    parse_fees_overview,
)


def test_parses_protocols(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    assert len(df) == 3
    assert set(df["protocol"]) == {"Curve DEX", "Aave V2", "WBTC"}


def test_keeps_fees_and_net_revenue_as_distinct_columns(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    curve = df[df["protocol"] == "Curve DEX"].iloc[0]
    assert curve["total_fees_24h"] == 7415
    assert curve["net_revenue_24h"] == 1483


def test_missing_net_revenue_is_nan_not_a_fallback_to_gross_fees(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    aave = df[df["protocol"] == "Aave V2"].iloc[0]
    assert np.isnan(aave["net_revenue_24h"])
    assert aave["total_fees_24h"] == 5266


def test_methodology_hash_is_stable_across_key_order():
    a = {"methodology": {"Fees": "x", "Revenue": "y"}}
    b = {"methodology": {"Revenue": "y", "Fees": "x"}}
    assert methodology_hash(a) == methodology_hash(b)


def test_methodology_hash_changes_when_the_text_changes():
    a = {"methodology": {"Fees": "Trading fees paid by users"}}
    b = {"methodology": {"Fees": "Trading fees plus incentives"}}
    assert methodology_hash(a) != methodology_hash(b)


def test_empty_methodology_still_hashes(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    wbtc = df[df["protocol"] == "WBTC"].iloc[0]
    assert len(wbtc["methodology_hash"]) == 64


def test_source_metadata():
    s = DefiLlamaSource()
    assert s.name == "defillama"
    assert s.contract_path.endswith("defillama_fees.yml")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_defillama.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.sources.defillama'`.

- [ ] **Step 4: Write the implementation**

```python
# cryptoval/sources/defillama.py
"""DefiLlama adapter for protocol fees and revenue.

Gross fees, net protocol revenue and value reaching token holders are three
different quantities (spec section 10.4). This adapter keeps them in
separate columns and never substitutes one for another, so Plan 2's
multiples cannot quietly use the wrong denominator.
"""

from __future__ import annotations

import datetime as dt
import time

import httpx
import pandas as pd

from cryptoval.config import Asset
from cryptoval.hashing import stable_hash
from cryptoval.sources.base import FetchResult

BASE_URL = "https://api.llama.fi"
OVERVIEW_PATH = (
    "/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"
)


def methodology_hash(record: dict) -> str:
    return stable_hash(record.get("methodology") or {})


def parse_fees_overview(payload: dict) -> pd.DataFrame:
    rows = []
    for record in payload.get("protocols", []) or []:
        rows.append(
            {
                "protocol": record.get("name"),
                "total_fees_24h": record.get("total24h"),
                "total_fees_30d": record.get("total30d"),
                "net_revenue_24h": record.get("revenue24h"),
                "methodology_hash": methodology_hash(record),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "protocol",
                "total_fees_24h",
                "total_fees_30d",
                "net_revenue_24h",
                "methodology_hash",
            ]
        )
    for col in ("total_fees_24h", "total_fees_30d", "net_revenue_24h"):
        # A null revenue stays null. It is not the same as zero revenue, and
        # it is certainly not gross fees.
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class DefiLlamaSource:
    name = "defillama"
    contract_path = "config/contracts/defillama_fees.yml"

    def __init__(self, max_retries: int = 5) -> None:
        self.max_retries = max_retries

    def _get(self, client: httpx.Client, url: str) -> dict:
        delay = 1.0
        for attempt in range(self.max_retries):
            response = client.get(url)
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == self.max_retries - 1:
                    response.raise_for_status()
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("unreachable")

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult:
        payload = self._get(client, f"{BASE_URL}{OVERVIEW_PATH}")
        rows_in = len(payload.get("protocols", []) or [])
        frame = parse_fees_overview(payload)
        frame["metric_date"] = pd.Timestamp(end, tz="UTC").normalize()
        return FetchResult(frame, rows_in)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_defillama.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/sources/defillama.py config/contracts/defillama_fees.yml \
        tests/fixtures/defillama_fees_overview.json tests/test_defillama.py
git commit -m "feat: add DefiLlama source with methodology hash and denominator discipline"
```

---

## Task 10: CoinGecko and equity price adapters

**Files:**
- Create: `cryptoval/sources/prices.py`, `config/contracts/coingecko_market_chart.yml`,
  `tests/fixtures/coingecko_btc_market_chart.json`
- Test: `tests/test_prices.py`

**Interfaces:**
- Consumes: `FetchResult` (Task 8).
- Produces: `CoinGeckoSource`, `EquitySource`; `parse_market_chart(payload, gecko_id) -> DataFrame`
  with columns `asset`, `metric_date`, `price_usd`, `market_cap_usd`, `volume_usd`;
  `compute_velocity(df) -> DataFrame` adding `velocity` and `holding_time_days`. Plan 2's equation
  of exchange methods consume `velocity`; spec §10.2 defines `V = T/MC` and `H = MC/T`.

- [ ] **Step 1: Write the contract and fixture**

```bash
cat > config/contracts/coingecko_market_chart.yml <<'EOF'
contract:
  name: coingecko_market_chart
  version: "1.0.0"
  owner: yasir
  source_url: https://api.coingecko.com/api/v3

schema:
  required:
    - asset
    - metric_date
    - price_usd
  columns:
    asset: {type: string}
    metric_date: {type: date}
    price_usd: {type: float, minimum: 0}
    market_cap_usd: {type: float, minimum: 0}
    volume_usd: {type: float, minimum: 0}

quality:
  freshness:
    max_delay_hours: 48
  uniqueness:
    combination: [asset, metric_date]
EOF

cat > tests/fixtures/coingecko_btc_market_chart.json <<'EOF'
{"prices":[[1791331200000,85540.08],[1791417600000,83282.04],[1791504000000,81685.88]],
 "market_caps":[[1791331200000,1718883142480.0],[1791417600000,1673545872851.0],[1791504000000,1641000000000.0]],
 "total_volumes":[[1791331200000,45000000000.0],[1791417600000,52000000000.0],[1791504000000,61000000000.0]]}
EOF
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_prices.py
import numpy as np
import pandas as pd
import pytest

from cryptoval.sources.prices import (
    CoinGeckoSource,
    EquitySource,
    compute_velocity,
    parse_market_chart,
)


def test_parses_three_series_into_one_panel(load_fixture):
    df = parse_market_chart(load_fixture("coingecko_btc_market_chart.json"), "bitcoin")
    assert len(df) == 3
    assert list(df.columns) == [
        "asset", "metric_date", "price_usd", "market_cap_usd", "volume_usd"
    ]


def test_asset_is_the_gecko_id(load_fixture):
    df = parse_market_chart(load_fixture("coingecko_btc_market_chart.json"), "bitcoin")
    assert set(df["asset"]) == {"bitcoin"}


def test_epoch_millis_become_utc_dates(load_fixture):
    df = parse_market_chart(load_fixture("coingecko_btc_market_chart.json"), "bitcoin")
    assert str(df["metric_date"].dt.tz) == "UTC"
    assert (df["metric_date"] == df["metric_date"].dt.normalize()).all()


def test_values_align_across_the_three_series(load_fixture):
    df = parse_market_chart(load_fixture("coingecko_btc_market_chart.json"), "bitcoin")
    first = df.sort_values("metric_date").iloc[0]
    assert first["price_usd"] == pytest.approx(85540.08)
    assert first["market_cap_usd"] == pytest.approx(1.71888314248e12)
    assert first["volume_usd"] == pytest.approx(4.5e10)


def test_velocity_is_volume_over_market_cap():
    df = pd.DataFrame({"market_cap_usd": [1000.0], "volume_usd": [50.0]})
    out = compute_velocity(df)
    assert out["velocity"].iloc[0] == pytest.approx(0.05)
    assert out["holding_time_days"].iloc[0] == pytest.approx(20.0)


def test_zero_market_cap_gives_nan_velocity_not_inf():
    df = pd.DataFrame({"market_cap_usd": [0.0], "volume_usd": [50.0]})
    out = compute_velocity(df)
    assert np.isnan(out["velocity"].iloc[0])


def test_zero_volume_gives_nan_holding_time_not_inf():
    df = pd.DataFrame({"market_cap_usd": [1000.0], "volume_usd": [0.0]})
    out = compute_velocity(df)
    assert np.isnan(out["holding_time_days"].iloc[0])


def test_source_metadata():
    assert CoinGeckoSource().name == "coingecko"
    assert CoinGeckoSource().contract_path.endswith("coingecko_market_chart.yml")
    assert EquitySource().name == "equities"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prices.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.sources.prices'`.

- [ ] **Step 4: Write the implementation**

```python
# cryptoval/sources/prices.py
"""Price-history adapters: CoinGecko for crypto, yfinance for equities.

CoinGecko supplies the equation-of-exchange inputs (price, market cap,
volume) for every asset including those absent from Coin Metrics' free
tier, which is how the documented SOL gap still gets price coverage
(spec section 5).
"""

from __future__ import annotations

import datetime as dt
import time

import httpx
import pandas as pd

from cryptoval.config import Asset
from cryptoval.sources.base import FetchResult

BASE_URL = "https://api.coingecko.com/api/v3"
PANEL_COLUMNS = ["asset", "metric_date", "price_usd", "market_cap_usd", "volume_usd"]


def _series(rows: list[list[float]], name: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["metric_date", name])
    df = pd.DataFrame(rows, columns=["ts_ms", name])
    df["metric_date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.normalize()
    return df.drop(columns=["ts_ms"]).groupby("metric_date", as_index=False).last()


def parse_market_chart(payload: dict, gecko_id: str) -> pd.DataFrame:
    price = _series(payload.get("prices", []) or [], "price_usd")
    mcap = _series(payload.get("market_caps", []) or [], "market_cap_usd")
    vol = _series(payload.get("total_volumes", []) or [], "volume_usd")
    if price.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    df = price.merge(mcap, on="metric_date", how="left").merge(vol, on="metric_date", how="left")
    df["asset"] = gecko_id
    return df.reindex(columns=PANEL_COLUMNS)


def compute_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mcap = pd.to_numeric(out["market_cap_usd"], errors="coerce").where(lambda s: s > 0)
    vol = pd.to_numeric(out["volume_usd"], errors="coerce").where(lambda s: s > 0)
    out["velocity"] = vol / mcap
    out["holding_time_days"] = mcap / vol
    return out


class CoinGeckoSource:
    name = "coingecko"
    contract_path = "config/contracts/coingecko_market_chart.yml"

    def __init__(self, max_retries: int = 5, sleep_between_assets_s: float = 2.5) -> None:
        self.max_retries = max_retries
        self.sleep_between_assets_s = sleep_between_assets_s

    def _get(self, client: httpx.Client, url: str, params: dict) -> dict:
        delay = 2.0
        for attempt in range(self.max_retries):
            response = client.get(url, params=params)
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == self.max_retries - 1:
                    response.raise_for_status()
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("unreachable")

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult:
        frames: list[pd.DataFrame] = []
        rows_in = 0
        for asset in assets:
            if not asset.gecko_id:
                continue
            payload = self._get(
                client,
                f"{BASE_URL}/coins/{asset.gecko_id}/market_chart/range",
                {
                    "vs_currency": "usd",
                    "from": int(
                        dt.datetime.combine(start, dt.time.min, tzinfo=dt.UTC).timestamp()
                    ),
                    "to": int(dt.datetime.combine(end, dt.time.max, tzinfo=dt.UTC).timestamp()),
                },
            )
            rows_in += len(payload.get("prices", []) or [])
            frames.append(parse_market_chart(payload, asset.gecko_id))
            time.sleep(self.sleep_between_assets_s)  # Demo tier is rate limited

        if not frames:
            return FetchResult(pd.DataFrame(columns=PANEL_COLUMNS), 0)
        return FetchResult(compute_velocity(pd.concat(frames, ignore_index=True)), rows_in)


class EquitySource:
    name = "equities"
    contract_path = "config/contracts/coingecko_market_chart.yml"

    def fetch(
        self,
        start: dt.date,
        end: dt.date,
        assets: list[Asset],
        client: httpx.Client,
    ) -> FetchResult:
        import yfinance

        tickers = [a.yahoo_id for a in assets if a.is_equity and a.yahoo_id]
        if not tickers:
            return FetchResult(pd.DataFrame(columns=PANEL_COLUMNS), 0)

        raw = yfinance.download(
            tickers, start=start.isoformat(), end=end.isoformat(), auto_adjust=True, progress=False
        )
        closes = raw["Close"] if "Close" in raw else raw
        tidy = (
            closes.reset_index()
            .melt(id_vars=closes.index.name or "Date", var_name="asset", value_name="price_usd")
            .rename(columns={closes.index.name or "Date": "metric_date"})
        )
        tidy["metric_date"] = pd.to_datetime(tidy["metric_date"], utc=True).dt.normalize()
        tidy["market_cap_usd"] = pd.NA
        tidy["volume_usd"] = pd.NA
        frame = tidy.reindex(columns=PANEL_COLUMNS).dropna(subset=["price_usd"])
        return FetchResult(frame, len(frame))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prices.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 6: Commit**

```bash
git add cryptoval/sources/prices.py config/contracts/coingecko_market_chart.yml \
        tests/fixtures/coingecko_btc_market_chart.json tests/test_prices.py
git commit -m "feat: add CoinGecko and equity price adapters with velocity derivation"
```

---

## Task 11: SCD Type 2 bi-temporal merge

**Files:**
- Create: `cryptoval/scd2.py`
- Test: `tests/test_scd2.py`

**Interfaces:**
- Consumes: `stable_hash` (Task 4).
- Produces: `FAR_FUTURE: pandas.Timestamp`; `add_row_hash(df, value_cols) -> DataFrame`;
  `merge_scd2(existing, incoming, key, value_cols, as_of) -> DataFrame`;
  `as_of_view(df, at) -> DataFrame`; `current_view(df) -> DataFrame`;
  `reconcile_report(existing, incoming, key, value_cols) -> dict`. This is the module that makes
  every backtest in Plan 4 honest: `as_of_view` is how a model sees only what was knowable at the
  time (spec §8.2).

**Review Focus coverage:** items 2 (duplicate keys must never produce two current rows) and 4
(revisions outside the trailing window must be detectable via `reconcile_report`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scd2.py
import datetime as dt

import pandas as pd

from cryptoval.scd2 import (
    FAR_FUTURE,
    add_row_hash,
    as_of_view,
    current_view,
    merge_scd2,
    reconcile_report,
)

KEY = ["asset", "metric_date"]
VALS = ["market_cap"]
T1 = dt.datetime(2026, 10, 11, 2, tzinfo=dt.UTC)
T2 = dt.datetime(2026, 10, 12, 2, tzinfo=dt.UTC)
T3 = dt.datetime(2026, 10, 13, 2, tzinfo=dt.UTC)


def _incoming(market_cap: float, asset: str = "btc", day: str = "2026-10-10"):
    return pd.DataFrame(
        {
            "asset": [asset],
            "metric_date": pd.to_datetime([day], utc=True),
            "market_cap": [market_cap],
        }
    )


def _empty():
    return pd.DataFrame(columns=[*KEY, *VALS])


def test_row_hash_is_stable_and_content_sensitive():
    a = add_row_hash(_incoming(100.0), VALS)
    b = add_row_hash(_incoming(100.0), VALS)
    c = add_row_hash(_incoming(101.0), VALS)
    assert a.loc[0, "row_hash"] == b.loc[0, "row_hash"]
    assert a.loc[0, "row_hash"] != c.loc[0, "row_hash"]


def test_first_load_inserts_one_open_current_row():
    out = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    assert len(out) == 1
    assert out.loc[0, "is_current"]
    assert out.loc[0, "effective_start_ts"] == pd.Timestamp(T1)
    assert out.loc[0, "effective_end_ts"] == FAR_FUTURE


def test_identical_reload_writes_nothing_new():
    first = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    second = merge_scd2(first, _incoming(100.0), KEY, VALS, T2)
    assert len(second) == 1
    assert second.loc[0, "effective_start_ts"] == pd.Timestamp(T1)
    assert second.loc[0, "is_current"]


def test_revision_closes_the_old_row_and_opens_a_new_one():
    first = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    second = merge_scd2(first, _incoming(110.0), KEY, VALS, T2)
    assert len(second) == 2
    closed = second[~second["is_current"]].iloc[0]
    openrow = second[second["is_current"]].iloc[0]
    assert closed["market_cap"] == 100.0
    assert closed["effective_end_ts"] == pd.Timestamp(T2)
    assert openrow["market_cap"] == 110.0
    assert openrow["effective_start_ts"] == pd.Timestamp(T2)
    assert openrow["effective_end_ts"] == FAR_FUTURE


def test_exactly_one_current_row_per_key_after_several_revisions():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(110.0), KEY, VALS, T2)
    hist = merge_scd2(hist, _incoming(120.0), KEY, VALS, T3)
    assert len(hist) == 3
    assert int(hist["is_current"].sum()) == 1
    assert hist[hist["is_current"]].iloc[0]["market_cap"] == 120.0


def test_duplicate_incoming_keys_cannot_create_two_current_rows():
    dupes = pd.concat([_incoming(100.0), _incoming(105.0)], ignore_index=True)
    out = merge_scd2(_empty(), dupes, KEY, VALS, T1)
    assert len(out) == 1
    assert int(out["is_current"].sum()) == 1
    # last row wins, mirroring an API that repeats a key across pages
    assert out.loc[0, "market_cap"] == 105.0


def test_as_of_view_reconstructs_the_old_vintage():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(110.0), KEY, VALS, T2)
    before = as_of_view(hist, dt.datetime(2026, 10, 11, 12, tzinfo=dt.UTC))
    after = as_of_view(hist, dt.datetime(2026, 10, 12, 12, tzinfo=dt.UTC))
    assert len(before) == 1 and before.iloc[0]["market_cap"] == 100.0
    assert len(after) == 1 and after.iloc[0]["market_cap"] == 110.0


def test_as_of_view_before_any_knowledge_is_empty():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    assert as_of_view(hist, dt.datetime(2026, 1, 1, tzinfo=dt.UTC)).empty


def test_current_view_returns_only_open_rows():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(110.0), KEY, VALS, T2)
    assert len(current_view(hist)) == 1
    assert current_view(hist).iloc[0]["market_cap"] == 110.0


def test_independent_keys_do_not_interfere():
    hist = merge_scd2(_empty(), _incoming(100.0, asset="btc"), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(50.0, asset="eth"), KEY, VALS, T1)
    hist = merge_scd2(hist, _incoming(110.0, asset="btc"), KEY, VALS, T2)
    assert int(hist["is_current"].sum()) == 2
    eth = current_view(hist)
    eth = eth[eth["asset"] == "eth"].iloc[0]
    assert eth["market_cap"] == 50.0
    assert eth["effective_start_ts"] == pd.Timestamp(T1)


def test_reconcile_report_counts_changes_outside_the_trailing_window():
    hist = merge_scd2(_empty(), _incoming(100.0, day="2024-01-01"), KEY, VALS, T1)
    report = reconcile_report(hist, _incoming(999.0, day="2024-01-01"), KEY, VALS)
    assert report["changed"] == 1
    assert report["new"] == 0
    assert report["unchanged"] == 0


def test_reconcile_report_reports_a_clean_reconciliation():
    hist = merge_scd2(_empty(), _incoming(100.0), KEY, VALS, T1)
    report = reconcile_report(hist, _incoming(100.0), KEY, VALS)
    assert report == {"new": 0, "changed": 0, "unchanged": 1}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scd2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.scd2'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/scd2.py
"""Bi-temporal (SCD Type 2) merge for the silver panel.

Coin Metrics revises published metrics, so the panel carries two time
axes: metric_date (what the value describes) and the
effective_start_ts/effective_end_ts pair (when we knew it). Research
queries use as_of_view so a backtest sees only what was knowable at the
time; the dashboard uses current_view (spec section 8.2).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from cryptoval.hashing import stable_hash

FAR_FUTURE = pd.Timestamp("9999-12-31", tz="UTC")
META_COLUMNS = ["effective_start_ts", "effective_end_ts", "is_current", "row_hash"]


def add_row_hash(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    present = [c for c in value_cols if c in out.columns]
    out["row_hash"] = [
        stable_hash({c: row[c] for c in present}) for _, row in out[present].iterrows()
    ]
    return out


def _dedupe_incoming(incoming: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    # A paginated response can repeat a key. Without this, the merge would
    # open two current rows for one key (Review Focus 2).
    return incoming.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def merge_scd2(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    key: list[str],
    value_cols: list[str],
    as_of: dt.datetime,
) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of)
    fresh = add_row_hash(_dedupe_incoming(incoming, key), value_cols)

    if existing is None or existing.empty:
        opened = fresh.copy()
        opened["effective_start_ts"] = as_of_ts
        opened["effective_end_ts"] = FAR_FUTURE
        opened["is_current"] = True
        return opened.reset_index(drop=True)

    history = existing.copy()
    current = history[history["is_current"]]
    lookup = current.set_index(key)["row_hash"].to_dict()

    to_open: list[pd.DataFrame] = []
    keys_to_close: list[tuple] = []

    for _, row in fresh.iterrows():
        k = tuple(row[c] for c in key)
        known = lookup.get(k if len(k) > 1 else k[0])
        if known == row["row_hash"]:
            continue
        if known is not None:
            keys_to_close.append(k)
        to_open.append(row.to_frame().T)

    if keys_to_close:
        mask = history["is_current"] & history.set_index(key).index.isin(
            [k if len(k) > 1 else k[0] for k in keys_to_close]
        )
        history.loc[mask, "effective_end_ts"] = as_of_ts
        history.loc[mask, "is_current"] = False

    if not to_open:
        return history.reset_index(drop=True)

    opened = pd.concat(to_open, ignore_index=True)
    opened["effective_start_ts"] = as_of_ts
    opened["effective_end_ts"] = FAR_FUTURE
    opened["is_current"] = True
    merged = pd.concat([history, opened], ignore_index=True)
    merged["is_current"] = merged["is_current"].astype(bool)
    return merged


def as_of_view(df: pd.DataFrame, at: dt.datetime) -> pd.DataFrame:
    at_ts = pd.Timestamp(at)
    mask = (df["effective_start_ts"] <= at_ts) & (df["effective_end_ts"] > at_ts)
    return df[mask].reset_index(drop=True)


def current_view(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_current"]].reset_index(drop=True)


def reconcile_report(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    key: list[str],
    value_cols: list[str],
) -> dict[str, int]:
    """Compare a refetch against stored state without writing anything.

    Scheduled runs only refetch a trailing window, so a revision landing
    outside it would otherwise be invisible (Review Focus 4). The reconcile
    command runs this over an arbitrary range and reports the drift.
    """
    fresh = add_row_hash(_dedupe_incoming(incoming, key), value_cols)
    if existing is None or existing.empty:
        return {"new": len(fresh), "changed": 0, "unchanged": 0}

    current = existing[existing["is_current"]]
    lookup = current.set_index(key)["row_hash"].to_dict()

    new = changed = unchanged = 0
    for _, row in fresh.iterrows():
        k = tuple(row[c] for c in key)
        known = lookup.get(k if len(k) > 1 else k[0])
        if known is None:
            new += 1
        elif known == row["row_hash"]:
            unchanged += 1
        else:
            changed += 1
    return {"new": new, "changed": changed, "unchanged": unchanged}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scd2.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/scd2.py tests/test_scd2.py
git commit -m "feat: add bi-temporal SCD2 merge with as-of views and reconciliation"
```

---

## Task 12: Byte-budget gate and monthly ledger

**Files:**
- Create: `cryptoval/budget.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: `RunManifest` (Task 4), `Settings` (Task 3).
- Produces: `BudgetExceeded(Exception)`; `ByteBudgetGate(manifest, monthly_budget)` with
  `.remaining(now) -> int` and `.guard(estimated_bytes, now) -> None`. Plan 3 calls `.guard()` with
  the result of a BigQuery dry run before every real query, which is what makes the free-only
  decision in spec §6.3 enforceable rather than aspirational.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_budget.py
import datetime as dt

import pytest

from cryptoval.budget import BudgetExceeded, ByteBudgetGate
from cryptoval.manifest import RunManifest
from cryptoval.paths import PathResolver

NOW = dt.datetime(2026, 10, 11, tzinfo=dt.UTC)
BUDGET = 1_000


def _gate(tmp_path, scanned: list[int] | None = None):
    m = RunManifest(PathResolver(str(tmp_path)))
    for b in scanned or []:
        rec = m.start("bigquery", {}, started_at=NOW)
        m.finish(rec, rows_in=0, rows_out=1, status="success", bytes_scanned=b)
    return ByteBudgetGate(m, monthly_budget=BUDGET)


def test_full_budget_remains_when_nothing_scanned(tmp_path):
    assert _gate(tmp_path).remaining(NOW) == BUDGET


def test_remaining_subtracts_this_month_only(tmp_path):
    assert _gate(tmp_path, [300, 200]).remaining(NOW) == 500


def test_guard_allows_a_query_inside_budget(tmp_path):
    _gate(tmp_path, [300]).guard(500, NOW)  # must not raise


def test_guard_allows_a_query_exactly_at_budget(tmp_path):
    _gate(tmp_path, [300]).guard(700, NOW)


def test_guard_refuses_a_query_over_budget(tmp_path):
    with pytest.raises(BudgetExceeded, match="would scan"):
        _gate(tmp_path, [300]).guard(800, NOW)


def test_refusal_message_reports_estimate_and_remaining(tmp_path):
    with pytest.raises(BudgetExceeded) as exc:
        _gate(tmp_path, [900]).guard(500, NOW)
    msg = str(exc.value)
    assert "500" in msg and "100" in msg


def test_guard_refuses_a_single_query_larger_than_the_whole_budget(tmp_path):
    with pytest.raises(BudgetExceeded):
        _gate(tmp_path).guard(BUDGET + 1, NOW)


def test_budget_resets_in_a_new_month(tmp_path):
    gate = _gate(tmp_path, [900])
    next_month = dt.datetime(2026, 11, 1, tzinfo=dt.UTC)
    assert gate.remaining(next_month) == BUDGET
    gate.guard(900, next_month)


def test_negative_estimate_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        _gate(tmp_path).guard(-1, NOW)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_budget.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.budget'`.

- [ ] **Step 3: Write the implementation**

```python
# cryptoval/budget.py
"""Byte-budget gate for BigQuery (spec section 6.3).

The free-only decision is enforced by two guards: a sandbox project with
no billing account, and this gate. Every query dry-runs first and passes
its estimate here; the gate also tracks cumulative bytes for the calendar
month, because the sandbox allowance is monthly, not per query.
"""

from __future__ import annotations

import datetime as dt

from cryptoval.manifest import RunManifest


class BudgetExceeded(RuntimeError):
    """Raised instead of running a query that would breach the byte budget."""


class ByteBudgetGate:
    def __init__(self, manifest: RunManifest, monthly_budget: int) -> None:
        self.manifest = manifest
        self.monthly_budget = monthly_budget

    def remaining(self, now: dt.datetime) -> int:
        used = self.manifest.bytes_scanned_in_month(now.year, now.month)
        return max(self.monthly_budget - used, 0)

    def guard(self, estimated_bytes: int, now: dt.datetime) -> None:
        if estimated_bytes < 0:
            raise ValueError("estimated_bytes must be non-negative")
        left = self.remaining(now)
        if estimated_bytes > left:
            raise BudgetExceeded(
                f"query would scan {estimated_bytes} bytes but only {left} bytes "
                f"remain in the {now:%Y-%m} budget of {self.monthly_budget}. "
                "Stage this extraction into a later month (spec section 6.3)."
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_budget.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add cryptoval/budget.py tests/test_budget.py
git commit -m "feat: add byte-budget gate with monthly ledger for free-tier enforcement"
```

---

## Task 13: DuckDB views, dbt project and marts

**Files:**
- Create: `cryptoval/io/duck.py`, `dbt/dbt_project.yml`, `dbt/profiles.yml`,
  `dbt/models/staging/stg_asset_metrics.sql`, `dbt/models/marts/dim_asset.sql`,
  `dbt/models/marts/dim_date.sql`, `dbt/models/marts/fct_asset_metrics_daily.sql`,
  `dbt/models/marts/_marts.yml`
- Modify: `.github/workflows/ci.yml` (add a dbt build step)
- Test: `tests/test_duck.py`

**Interfaces:**
- Consumes: `PathResolver` (Task 2), the silver Parquet written in Task 14.
- Produces: `connect(resolver, read_only=False) -> duckdb.DuckDBPyConnection`;
  `register_silver(con, resolver) -> None` creating the view `silver_asset_metrics`;
  `query(con, sql) -> pandas.DataFrame`. dbt marts `dim_asset`, `dim_date`,
  `fct_asset_metrics_daily`. `dim_date` carries `halving_epoch` and `days_since_halving`, which
  spec §8.1 requires so Plan 2's cycle metrics are a join rather than a recalculation.

**Deliberate deferral.** Spec §8.3 requires `dim_asset` to be SCD2 on sector classification. Sector
currently comes from a static `config/assets.yml` with no recorded history, so there is no
versioning to do yet and an SCD2 structure would carry exactly one version per asset. The SCD2
treatment lands in Plan 4, where the cross-sectional factor sorts are the first consumer that can
actually be biased by it. This is recorded as a decision so it is not mistaken for an oversight.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_duck.py
import datetime as dt
import pathlib

import pandas as pd
import pytest

from cryptoval.io.duck import connect, query, register_silver
from cryptoval.paths import PathResolver
from cryptoval.scd2 import FAR_FUTURE


def _write_silver(root: pathlib.Path) -> None:
    out = pathlib.Path(root, "silver", "asset_metrics")
    out.mkdir(parents=True)
    pd.DataFrame(
        {
            "asset": ["btc", "btc"],
            "metric_date": pd.to_datetime(["2026-10-10", "2026-10-11"], utc=True),
            "CapMrktCurUSD": [1.0e12, 1.1e12],
            "realized_cap": [6.0e11, 6.1e11],
            "effective_start_ts": [pd.Timestamp("2026-10-11", tz="UTC")] * 2,
            "effective_end_ts": [FAR_FUTURE] * 2,
            "is_current": [True, True],
            "row_hash": ["a", "b"],
        }
    ).to_parquet(out / "part-0.parquet", index=False)


def test_connects_and_queries(tmp_path):
    con = connect(PathResolver(str(tmp_path)))
    assert query(con, "select 1 as x").loc[0, "x"] == 1


def test_registers_silver_view(tmp_path):
    _write_silver(tmp_path)
    r = PathResolver(str(tmp_path))
    con = connect(r)
    register_silver(con, r)
    df = query(con, "select count(*) as n from silver_asset_metrics")
    assert df.loc[0, "n"] == 2


def test_silver_view_preserves_bitemporal_columns(tmp_path):
    _write_silver(tmp_path)
    r = PathResolver(str(tmp_path))
    con = connect(r)
    register_silver(con, r)
    cols = set(query(con, "select * from silver_asset_metrics limit 1").columns)
    assert {"effective_start_ts", "effective_end_ts", "is_current", "row_hash"} <= cols


def test_as_of_filter_works_in_sql(tmp_path):
    _write_silver(tmp_path)
    r = PathResolver(str(tmp_path))
    con = connect(r)
    register_silver(con, r)
    sql = """
        select count(*) as n from silver_asset_metrics
        where effective_start_ts <= timestamptz '2026-10-12 00:00:00+00'
          and effective_end_ts  >  timestamptz '2026-10-12 00:00:00+00'
    """
    assert query(con, sql).loc[0, "n"] == 2


def test_register_silver_raises_a_clear_error_when_absent(tmp_path):
    r = PathResolver(str(tmp_path))
    con = connect(r)
    with pytest.raises(FileNotFoundError, match="no silver"):
        register_silver(con, r)


def test_duckdb_file_is_created_under_the_data_root(tmp_path):
    connect(PathResolver(str(tmp_path)))
    assert pathlib.Path(tmp_path, "cryptoval.duckdb").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_duck.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.io.duck'`.

- [ ] **Step 3: Write the DuckDB adapter**

```python
# cryptoval/io/duck.py
"""DuckDB as a query engine over Parquet.

The .duckdb file is derived and disposable; deleting it loses nothing,
because the Parquet files under silver/ are the system of record
(spec section 7.1).
"""

from __future__ import annotations

import pathlib

import duckdb
import pandas as pd

from cryptoval.paths import PathResolver


def connect(resolver: PathResolver, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if resolver.is_remote:
        raise NotImplementedError("remote DuckDB roots are added in Plan 5")
    root = pathlib.Path(resolver.root)
    root.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(root / "cryptoval.duckdb"), read_only=read_only)


def register_silver(con: duckdb.DuckDBPyConnection, resolver: PathResolver) -> None:
    table_dir = pathlib.Path(resolver.table("silver", "asset_metrics"))
    if not table_dir.exists() or not any(table_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"no silver asset_metrics parquet under {table_dir}; run `make backfill` first"
        )
    glob = str(table_dir / "*.parquet")
    con.execute(
        f"create or replace view silver_asset_metrics as select * from read_parquet('{glob}')"
    )


def query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetch_df()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_duck.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Write the dbt project and profile**

Two targets, so the cloud migration in Plan 5 is a `--target` flag (spec §7.3).

```bash
mkdir -p dbt/models/staging dbt/models/marts
cat > dbt/dbt_project.yml <<'EOF'
name: cryptoval
version: "0.1.0"
config-version: 2
profile: cryptoval
model-paths: ["models"]
target-path: "target"
clean-targets: ["target", "dbt_packages"]

vars:
  data_root: "../data"
  bitcoin_genesis: "2009-01-03"

models:
  cryptoval:
    staging:
      +materialized: view
    marts:
      +materialized: table
EOF

cat > dbt/profiles.yml <<'EOF'
cryptoval:
  target: duckdb
  outputs:
    duckdb:
      type: duckdb
      path: "../data/cryptoval.duckdb"
      threads: 4
    # Plan 5 switches to this target without changing any model SQL.
    bigquery:
      type: bigquery
      method: oauth
      project: "{{ env_var('GCP_PROJECT', 'not-set') }}"
      dataset: cryptoval
      location: US
      threads: 4
      priority: interactive
EOF
```

- [ ] **Step 6: Write the staging and mart models**

```bash
cat > dbt/models/staging/stg_asset_metrics.sql <<'EOF'
-- Current-vintage view of the bi-temporal silver panel.
-- Point-in-time analysis in Plan 4 reads the panel directly with an
-- as-of filter instead of this model.
select
    asset,
    cast(metric_date as date)      as metric_date,
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
    effective_start_ts,
    effective_end_ts,
    row_hash
from read_parquet('{{ var("data_root") }}/silver/asset_metrics/*.parquet')
where is_current
EOF

cat > dbt/models/marts/dim_date.sql <<'EOF'
-- Date dimension carrying Bitcoin halving context, so cycle metrics in
-- Plan 2 are a join rather than a recalculation (spec section 8.1).
with bounds as (
    select min(metric_date) as d0, max(metric_date) as d1 from {{ ref('stg_asset_metrics') }}
),
spine as (
    select unnest(generate_series(
        (select d0 from bounds),
        (select d1 from bounds),
        interval 1 day
    ))::date as date_day
),
halvings as (
    select * from (values
        (0, date '2009-01-03'),
        (1, date '2012-11-28'),
        (2, date '2016-07-09'),
        (3, date '2020-05-11'),
        (4, date '2024-04-19'),
        (5, date '2028-04-01')   -- projected; refreshed when the real date is known
    ) as t(epoch, started_on)
)
select
    s.date_day,
    extract(year from s.date_day)                      as year,
    extract(month from s.date_day)                     as month,
    s.date_day = (date_trunc('month', s.date_day) + interval 1 month - interval 1 day)::date
                                                       as is_month_end,
    h.epoch                                            as halving_epoch,
    date_diff('day', h.started_on, s.date_day)         as days_since_halving
from spine s
left join halvings h
    on h.started_on <= s.date_day
   and s.date_day < coalesce(
        (select min(h2.started_on) from halvings h2 where h2.started_on > h.started_on),
        date '9999-12-31'
   )
EOF

cat > dbt/models/marts/dim_asset.sql <<'EOF'
-- Asset dimension derived from what the panel actually contains.
-- Plan 2 joins sector and consensus from config/assets.yml as a dbt seed;
-- this model establishes the grain and the observed date coverage.
select
    asset,
    min(metric_date) as first_observed_on,
    max(metric_date) as last_observed_on,
    count(*)         as observation_days
from {{ ref('stg_asset_metrics') }}
group by asset
EOF

cat > dbt/models/marts/fct_asset_metrics_daily.sql <<'EOF'
-- The conformed daily fact table: one row per asset per day, with the
-- derived cost-approach quantities the spec requires (section 6.2).
with base as (
    select * from {{ ref('stg_asset_metrics') }}
),
derived as (
    select
        *,
        issuance_usd + (fees_native * price_usd)                    as miner_revenue_usd,
        sum(issuance_usd + (fees_native * price_usd)) over (
            partition by asset order by metric_date
            rows between unbounded preceding and current row
        )                                                           as thermocap_usd
    from base
)
select
    *,
    case when thermocap_usd > 0 then market_cap_usd / thermocap_usd end as mvtv,
    realized_cap - thermocap_usd                                        as investor_cap_usd
from derived
EOF

cat > dbt/models/marts/_marts.yml <<'EOF'
version: 2

models:
  - name: fct_asset_metrics_daily
    description: Conformed daily asset panel with derived cost-approach metrics.
    tests:
      - dbt_utils_unique_combination_of_columns:
          combination_of_columns: [asset, metric_date]
    columns:
      - name: asset
        tests: [not_null]
      - name: metric_date
        tests: [not_null]
      - name: market_cap_usd
        tests:
          - not_null
          - dbt_utils_accepted_range:
              min_value: 0
      - name: mvrv
        tests:
          - dbt_utils_accepted_range:
              min_value: 0
              severity: warn

  - name: dim_date
    columns:
      - name: date_day
        tests: [not_null, unique]
      - name: halving_epoch
        tests: [not_null]

  - name: dim_asset
    columns:
      - name: asset
        tests: [not_null, unique]
EOF
```

- [ ] **Step 7: Install dbt_utils and verify the project parses**

The `dbt_utils_*` tests above come from the dbt-utils package.

```bash
cat > dbt/packages.yml <<'EOF'
packages:
  - package: dbt-labs/dbt_utils
    version: 1.3.0
EOF
cd dbt && ../.venv/bin/dbt deps --profiles-dir . && ../.venv/bin/dbt parse --profiles-dir . && cd ..
```

Expected: `dbt deps` installs dbt_utils; `dbt parse` reports no errors.

Then rename the test references to their package form, which dbt requires:

```bash
sed -i '' 's/dbt_utils_unique_combination_of_columns/dbt_utils.unique_combination_of_columns/' dbt/models/marts/_marts.yml
sed -i '' 's/dbt_utils_accepted_range/dbt_utils.accepted_range/' dbt/models/marts/_marts.yml
cd dbt && ../.venv/bin/dbt parse --profiles-dir . && cd ..
```

Expected: parse succeeds with the package-qualified test names.

- [ ] **Step 8: Add the dbt step to CI**

```bash
python3 - <<'EOF'
import pathlib
p = pathlib.Path(".github/workflows/ci.yml")
s = p.read_text(encoding="utf-8")
s += """      - name: dbt parse
        run: |
          cd dbt
          dbt deps --profiles-dir .
          dbt parse --profiles-dir .
"""
p.write_text(s, encoding="utf-8")
EOF
cat .github/workflows/ci.yml
```

`dbt parse` rather than `dbt build`, because CI has no ingested data. Task 14's integration test
covers a real `dbt build`.

- [ ] **Step 9: Commit**

```bash
echo "dbt/dbt_packages/" >> .gitignore
git add cryptoval/io/duck.py dbt .github/workflows/ci.yml .gitignore tests/test_duck.py
git commit -m "feat: add DuckDB parquet views and dbt marts with halving-aware date dimension"
```

---

## Task 14: CLI, orchestration and end-to-end verification

**Files:**
- Create: `cryptoval/cli.py`, `tests/test_cli.py`, `tests/test_integration_sources.py`
- Modify: `cryptoval/io/bronze.py` (add `write_silver`)
- Test: `tests/test_cli.py`, `tests/test_integration_sources.py`

**Interfaces:**
- Consumes: everything from Tasks 2-13.
- Produces: `write_silver(resolver, table, df) -> str`; `SOURCES: dict[str, Source]`;
  `run_ingest(settings, assets, source_names, start, end) -> dict[str, dict]`;
  `main(argv) -> int` backing `python -m cryptoval.cli {ingest,backfill,validate,reconcile}`.
  Plan 2 imports nothing from this module — it reads the silver panel — so the CLI stays the only
  wiring layer.

- [ ] **Step 1: Write the silver writer and the failing CLI tests**

```python
# tests/test_cli.py
import datetime as dt
import pathlib

import pandas as pd
import pytest

from cryptoval.cli import SOURCES, main, run_ingest
from cryptoval.config import Asset, Settings
from cryptoval.io.bronze import write_silver
from cryptoval.paths import PathResolver


def test_all_four_sources_are_registered():
    assert set(SOURCES) == {"coinmetrics", "defillama", "coingecko", "equities"}


def test_write_silver_round_trips(tmp_path):
    r = PathResolver(str(tmp_path))
    df = pd.DataFrame({"asset": ["btc"], "metric_date": pd.to_datetime(["2026-10-11"], utc=True)})
    path = write_silver(r, "asset_metrics", df)
    assert pathlib.Path(path).exists()
    assert len(pd.read_parquet(path)) == 1


def test_write_silver_overwrites_rather_than_appending(tmp_path):
    r = PathResolver(str(tmp_path))
    df = pd.DataFrame({"asset": ["btc"], "metric_date": pd.to_datetime(["2026-10-11"], utc=True)})
    write_silver(r, "asset_metrics", df)
    write_silver(r, "asset_metrics", df)
    files = list(pathlib.Path(r.table("silver", "asset_metrics")).glob("*.parquet"))
    assert len(files) == 1


def test_run_ingest_records_a_failure_without_raising(tmp_path, monkeypatch):
    class Boom:
        name = "coinmetrics"
        contract_path = "config/contracts/coinmetrics_asset_metrics.yml"

        def fetch(self, start, end, assets, client):
            raise RuntimeError("HTTP 503")

    monkeypatch.setitem(SOURCES, "coinmetrics", Boom())
    settings = Settings(data_root=str(tmp_path))
    assets = [Asset(symbol="BTC", cm_id="btc", gecko_id="bitcoin",
                    sector="pow_monetary", consensus="pow")]
    report = run_ingest(
        settings, assets, ["coinmetrics"], dt.date(2026, 10, 1), dt.date(2026, 10, 11)
    )
    assert report["coinmetrics"]["status"] == "failed"
    assert "503" in report["coinmetrics"]["error"]


def test_failed_fetch_leaves_no_bronze_partition(tmp_path, monkeypatch):
    class Boom:
        name = "coinmetrics"
        contract_path = "config/contracts/coinmetrics_asset_metrics.yml"

        def fetch(self, start, end, assets, client):
            raise RuntimeError("boom")

    monkeypatch.setitem(SOURCES, "coinmetrics", Boom())
    settings = Settings(data_root=str(tmp_path))
    run_ingest(settings, [], ["coinmetrics"], dt.date(2026, 10, 1), dt.date(2026, 10, 11))
    bronze = pathlib.Path(tmp_path, "bronze")
    written = list(bronze.rglob("*.parquet")) if bronze.exists() else []
    assert written == []


def test_cli_rejects_an_unknown_source():
    assert main(["ingest", "--sources", "nope"]) == 2


def test_cli_rejects_an_end_before_start():
    assert main(["backfill", "--start", "2026-10-11", "--end", "2026-10-01"]) == 2


def test_cli_help_returns_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cryptoval.cli'`.

- [ ] **Step 3: Add the silver writer**

```python
# append to cryptoval/io/bronze.py
def write_silver(resolver: PathResolver, table: str, df: pd.DataFrame) -> str:
    """Write a silver table as a single compacted Parquet file.

    Yearly or finer partitioning is unnecessary at this volume and would
    create the small-file problem the spec warns about (section 7.4); a
    single file per table keeps DuckDB scans fast and the layout simple.
    """
    if resolver.is_remote:
        raise NotImplementedError("silver writes to object storage are added in Plan 5")
    out_dir = pathlib.Path(resolver.table("silver", table))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    path = out_dir / "part-0.parquet"
    df.to_parquet(path, index=False)
    return str(path)
```

- [ ] **Step 4: Write the CLI**

```python
# cryptoval/cli.py
"""Command-line entry point: the only module that wires the pieces together."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import httpx
import pandas as pd

from cryptoval.config import Asset, Settings, load_config
from cryptoval.contracts.checks import run_checks
from cryptoval.contracts.identities import run_identity_checks
from cryptoval.contracts.loader import load_contract
from cryptoval.io.bronze import write_bronze, write_silver
from cryptoval.manifest import RunManifest
from cryptoval.scd2 import merge_scd2, reconcile_report
from cryptoval.sources.coinmetrics import CoinMetricsSource
from cryptoval.sources.defillama import DefiLlamaSource
from cryptoval.sources.prices import CoinGeckoSource, EquitySource

SOURCES = {
    "coinmetrics": CoinMetricsSource(),
    "defillama": DefiLlamaSource(),
    "coingecko": CoinGeckoSource(),
    "equities": EquitySource(),
}

SILVER_TABLE = {
    "coinmetrics": "asset_metrics",
    "defillama": "protocol_revenue",
    "coingecko": "market_chart",
    "equities": "equity_prices",
}

KEYS = {
    "coinmetrics": ["asset", "metric_date"],
    "defillama": ["protocol", "metric_date"],
    "coingecko": ["asset", "metric_date"],
    "equities": ["asset", "metric_date"],
}


def _read_silver(settings: Settings, table: str) -> pd.DataFrame:
    path = pathlib.Path(settings.resolver.table("silver", table)) / "part-0.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def run_ingest(
    settings: Settings,
    assets: list[Asset],
    source_names: list[str],
    start: dt.date,
    end: dt.date,
) -> dict[str, dict]:
    manifest = RunManifest(settings.resolver)
    report: dict[str, dict] = {}
    as_of = dt.datetime.now(dt.UTC)

    with httpx.Client(timeout=settings.request_timeout_s) as client:
        for name in source_names:
            source = SOURCES[name]
            record = manifest.start(name, {"start": start.isoformat(), "end": end.isoformat()})
            try:
                result = source.fetch(start, end, assets, client)
                if result.frame.empty:
                    raise RuntimeError("source returned no rows")

                write_bronze(
                    settings.resolver,
                    name,
                    end,
                    result.frame,
                    {"start": start.isoformat(), "end": end.isoformat()},
                )

                table = SILVER_TABLE[name]
                key = KEYS[name]
                value_cols = [c for c in result.frame.columns
                              if c not in key and not c.startswith("_")]
                merged = merge_scd2(
                    _read_silver(settings, table), result.frame, key, value_cols, as_of
                )
                write_silver(settings.resolver, table, merged)

                manifest.finish(
                    record, result.rows_in, len(result.frame), "success"
                )
                report[name] = {
                    "status": "success",
                    "rows_in": result.rows_in,
                    "rows_out": len(result.frame),
                }
            except Exception as exc:  # recorded, not swallowed: see the manifest
                manifest.finish(record, 0, 0, "failed", error=str(exc))
                report[name] = {"status": "failed", "error": str(exc)}
    return report


def run_validate(settings: Settings) -> list[str]:
    failures: list[str] = []
    for name, table in SILVER_TABLE.items():
        df = _read_silver(settings, table)
        if df.empty:
            failures.append(f"{table}: no data ingested")
            continue
        contract = load_contract(SOURCES[name].contract_path)
        for result in run_checks(df, contract, now=dt.datetime.now(dt.UTC)):
            if not result.passed:
                failures.append(f"{table}/{result.name}: {result.detail}")
        for result in run_identity_checks(df):
            if not result.passed:
                failures.append(f"{table}/{result.name}: {result.detail}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cryptoval")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("ingest", "backfill", "reconcile"):
        p = sub.add_parser(cmd)
        p.add_argument("--sources", default=",".join(SOURCES))
        p.add_argument("--start")
        p.add_argument("--end")
        p.add_argument("--trailing-days", type=int)
    sub.add_parser("validate")

    args = parser.parse_args(argv)
    settings, assets = load_config()

    if args.command == "validate":
        failures = run_validate(settings)
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        print("validation passed" if not failures else f"{len(failures)} check(s) failed")
        return 1 if failures else 0

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        print(f"unknown source(s): {unknown}; known: {sorted(SOURCES)}", file=sys.stderr)
        return 2

    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    if args.trailing_days:
        start = end - dt.timedelta(days=args.trailing_days)
    elif args.start:
        start = dt.date.fromisoformat(args.start)
    else:
        start = end - dt.timedelta(days=settings.trailing_refetch_days)
    if start > end:
        print(f"start {start} is after end {end}", file=sys.stderr)
        return 2

    if args.command == "reconcile":
        with httpx.Client(timeout=settings.request_timeout_s) as client:
            for name in names:
                result = SOURCES[name].fetch(start, end, assets, client)
                table, key = SILVER_TABLE[name], KEYS[name]
                value_cols = [c for c in result.frame.columns
                              if c not in key and not c.startswith("_")]
                print(name, reconcile_report(
                    _read_silver(settings, table), result.frame, key, value_cols
                ))
        return 0

    report = run_ingest(settings, assets, names, start, end)
    for name, outcome in report.items():
        print(name, outcome)
    return 0 if all(o["status"] == "success" for o in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 6: Write the integration tests**

These hit live APIs and are excluded from the default run.

```python
# tests/test_integration_sources.py
import datetime as dt

import httpx
import pytest

from cryptoval.config import load_config
from cryptoval.contracts.identities import check_mvrv_identity
from cryptoval.sources.coinmetrics import CoinMetricsSource
from cryptoval.sources.defillama import DefiLlamaSource

pytestmark = pytest.mark.integration

END = dt.date.today() - dt.timedelta(days=2)
START = END - dt.timedelta(days=5)


def test_coinmetrics_free_tier_still_serves_mvrv():
    _, assets = load_config()
    btc = [a for a in assets if a.symbol == "BTC"]
    with httpx.Client(timeout=30) as client:
        result = CoinMetricsSource().fetch(START, END, btc, client)
    assert not result.frame.empty
    assert result.frame["CapMVRVCur"].notna().any(), (
        "CapMVRVCur returned no values: the free tier may have changed "
        "and the realized-cap derivation is no longer available (spec section 6.2)"
    )


def test_derived_realized_cap_satisfies_the_identity_on_live_data():
    _, assets = load_config()
    btc = [a for a in assets if a.symbol == "BTC"]
    with httpx.Client(timeout=30) as client:
        result = CoinMetricsSource().fetch(START, END, btc, client)
    check = check_mvrv_identity(result.frame)
    assert check.passed, check.detail


def test_defillama_still_exposes_fees_and_methodology():
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)
    assert len(result.frame) > 100
    assert result.frame["methodology_hash"].str.len().eq(64).all()
```

- [ ] **Step 7: Run the integration tests against live APIs**

Run: `.venv/bin/python -m pytest tests/test_integration_sources.py -v -m integration`
Expected: PASS, 3 tests. A failure here means a free tier changed — read the assertion message,
which names the consequence.

- [ ] **Step 8: Run the full pipeline end to end**

```bash
make backfill            # full history from 2015-01-01
make dbt                 # builds marts and runs dbt tests
make validate            # contract + identity checks, non-zero exit on failure
```

Expected: `backfill` prints a success line per source; `dbt build` completes with all tests
passing; `validate` prints "validation passed".

- [ ] **Step 9: Verify point-in-time reconstruction on real data**

```bash
.venv/bin/python - <<'EOF'
import datetime as dt
import pandas as pd
from cryptoval.config import load_config
from cryptoval.scd2 import as_of_view, current_view

settings, _ = load_config()
panel = pd.read_parquet(f"{settings.resolver.table('silver','asset_metrics')}/part-0.parquet")
print("rows:", len(panel), "| current:", len(current_view(panel)))
past = as_of_view(panel, dt.datetime.now(dt.UTC) - dt.timedelta(days=1))
print("as-of yesterday:", len(past))
assert int(panel.groupby(["asset","metric_date"])["is_current"].sum().max()) == 1, \
    "more than one current row for some key"
print("OK: exactly one current row per key")
EOF
```

Expected: prints row counts and `OK: exactly one current row per key`.

- [ ] **Step 10: Write the runbooks**

Spec §9.5 requires these, and they cover the failures that will actually happen rather than
hypothetical ones.

````bash
mkdir -p docs/runbooks
cat > docs/runbooks/free-tier-metric-withdrawn.md <<'EOF'
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
EOF

cat > docs/runbooks/defillama-methodology-change.md <<'EOF'
# Runbook: a DefiLlama protocol is renamed or its methodology changes

**Symptom:** `methodology_hash` changes for a protocol, or a volume-anomaly
check flags a step change in `total_fees_24h` / `net_revenue_24h`.

**Why it matters:** a methodology revision redefines the revenue series, so
history before and after are not comparable and any DCF built across the break
is wrong (spec §9.2).

**Diagnose:**
```
curl -s "https://api.llama.fi/summary/fees/<protocol>" \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('methodology'))"
```
Compare against the stored `methodology_hash` in `silver/protocol_revenue`.

**Act:**
1. Record the break date in the protocol's method card.
2. Decide explicitly whether to truncate the series at the break or model the
   two regimes separately. Do not silently splice them.
3. For a rename, map the old and new names in config and keep both in the
   SCD2 history so the old vintage is still reconstructable.
EOF

cat > docs/runbooks/coingecko-rate-limit.md <<'EOF'
# Runbook: CoinGecko 429 rate limiting

**Symptom:** `make backfill` fails with `HTTP 429` from
`api.coingecko.com`, or the run takes far longer than expected.

**Why it matters:** the Demo tier is capped (roughly 10k calls/month, with a
per-minute ceiling), and a full backfill across 24 assets is call-heavy.

**Act:**
1. The adapter already retries with exponential backoff and sleeps
   `sleep_between_assets_s` between assets. Raise that value first.
2. Backfill in slices: `make backfill` accepts `--start`/`--end`, so run a year
   at a time rather than the whole history in one pass.
3. Coin Metrics covers price and market cap for everything except SOL, so a
   CoinGecko outage degrades velocity coverage rather than blocking the panel.
EOF

cat > docs/runbooks/bigquery-budget-exceeded.md <<'EOF'
# Runbook: BigQuery byte budget exceeded

**Symptom:** `BudgetExceeded: query would scan N bytes but only M bytes remain`.

**Why it matters:** this is the gate working as designed (spec §6.3), not a
failure. The free-only decision means the query must wait, not be paid for.

**Act:**
1. Check the ledger: `bytes_scanned_in_month` in `gold/run_manifest.jsonl`.
2. Narrow the query — tighter `block_timestamp_month` pruning, fewer columns —
   and re-run the dry run.
3. If it still will not fit, stage the remaining epochs into next calendar month.
   Phase 3b is explicitly allowed to span months.
4. Never enable billing to get past this. That requires a new explicit decision
   recorded in the spec.
EOF

cat > docs/runbooks/asset-rebrand.md <<'EOF'
# Runbook: an asset is rebranded or re-ticker'd (e.g. MATIC to POL)

**Symptom:** an asset's series stops updating; `dim_asset.last_observed_on`
goes stale while the asset is plainly still trading.

**Act:**
1. Update `cm_id` / `gecko_id` in `config/assets.yml`; keep the historical
   symbol as the stable `symbol` so the SCD2 history stays joinable.
2. Run `make reconcile --start <first-affected-date>` to see how many rows
   changed before writing anything.
3. If the sector also changed, note the date — Plan 4 needs point-in-time
   sector labels and a silent reclassification is look-ahead bias (spec §8.3).
EOF
````

- [ ] **Step 11: Commit**

```bash
git add cryptoval/cli.py cryptoval/io/bronze.py tests/test_cli.py \
        tests/test_integration_sources.py docs/runbooks
git commit -m "feat: add CLI with ingest, backfill, validate and reconcile commands"
```

---

## Done criteria for Plan 1

- `make lint && make test` green; ~100 unit tests, no network.
- `make test-integration` green against live free APIs.
- `make backfill && make dbt && make validate` produces a populated silver panel and marts with all
  contract, identity and dbt tests passing.
- Exactly one `is_current` row per `(asset, metric_date)`, verified on real data.
- `as_of_view` reconstructs a past vintage, so Plan 4's backtests can be honest.
- CI runs lint, unit tests and `dbt parse` on every push.

**Plan 2 starts from:** `fct_asset_metrics_daily` plus the bi-temporal `silver/asset_metrics`
panel, `Asset.consensus` for cost-of-production applicability, `velocity` and `holding_time_days`
for the equation-of-exchange family, and `net_revenue_24h` for the DCF family.


---

## Appendix: spec coverage

Checked against the spec section by section. Everything in Phase 0 and Phase 1 scope is claimed by
a task; everything else is named with the plan that owns it.

| Spec section | Covered by | Note |
|---|---|---|
| §5 Asset scope | Task 3 | 24 crypto (23 + SOL as a documented gap) + 5 equities |
| §6.1 Verified sources | Tasks 8, 9, 10 | All four adapters |
| §6.2 Realized-cap derivation | Task 8 | Plus the blocking identity check in Task 6 |
| §6.3 Byte gate, monthly ledger | Task 12 | The extraction itself is Plan 3 |
| §6.4 Freshness | Task 5 | `check_freshness`, 48h SLA |
| §7.1 Decision record | Task 1 | No Spark/Kafka/GE/Prometheus appears anywhere |
| §7.2 Stack | Task 1 | Pins verified to resolve on Python 3.14.5 |
| §7.3 Portability contract | Tasks 2, 13 | Path resolver; dual dbt targets |
| §7.4 Layers and partitioning | Tasks 7, 14 | Bronze per-run; silver compacted to one file |
| §8.1 Star schema | Task 13 | `dim_asset`, `dim_date`, `fct_asset_metrics_daily`. `dim_method` → Plan 2; `fct_valuation_estimate` → Plan 2; `fct_model_score` → Plan 4; `obt_asset_valuation_wide` → Plan 5 |
| §8.2 Bi-temporal panel | Task 11 | The core of this plan |
| §8.3 Point-in-time dimensions | Task 13 | SCD2-on-sector deliberately deferred to Plan 4; rationale recorded in the task |
| §8.4 Idempotency | Tasks 7, 11 | Atomic partitions; hash-based merge |
| §9.1 Contracts | Tasks 5, 9, 10 | Three contract files |
| §9.2 Domain guards | Tasks 5, 6, 9, 14 | Identities, volume anomaly, methodology hash, free-tier availability |
| §9.3 dbt tests primary | Task 13 | Great Expectations deliberately absent |
| §9.4 Observability | Task 4 | `run_manifest`; `dbt docs` to Pages → Plan 5 |
| §9.5 Runbooks | Task 14 | Five runbooks for the failures that will actually occur |
| §10 Method registry | — | Plan 2 |
| §11 Statistical layer | — | Plan 4. The no-shuffle constraint is enforced from Task 1 |
| §12 Deliverables | — | Plan 5 |
| §13 Phase 6 cloud | — | Plan 5. Portability seams are in place from Task 2 |
