PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup test test-integration lint fmt ingest backfill validate reconcile seed cards estimate score dbt all

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

seed:
	$(PY) -c "from cryptoval.config import load_config; from cryptoval.seeds import write_asset_seeds; _, a = load_config(); print('\n'.join(write_asset_seeds(a, 'dbt/seeds')))"
	$(PY) -c "from cryptoval.methods.registry import load_method_specs; from cryptoval.seeds import write_method_seed; print(write_method_seed(load_method_specs(), 'dbt/seeds'))"

cards:
	$(PY) -c "from cryptoval.methods.registry import load_method_specs; from cryptoval.cards import load_card_extras, write_cards; print('\n'.join(write_cards(load_method_specs(), 'docs/methods', load_card_extras())))"

estimate:
	$(PY) -m cryptoval.cli estimate

score:
	$(PY) -m cryptoval.cli score

dbt:
	# full-refresh: a seed CSV that gained a column fails to load otherwise
	cd dbt && ../.venv/bin/dbt seed --full-refresh --profiles-dir . && ../.venv/bin/dbt build --profiles-dir .

# A recipe, not a prerequisite list: make runs a repeated prerequisite once, so
# `all: ... dbt estimate dbt ...` skipped the second dbt and the marts never saw the gold tables.
all:
	$(MAKE) lint test ingest
	$(MAKE) dbt
	$(MAKE) estimate
	$(MAKE) dbt
	$(MAKE) score
	$(MAKE) dbt
	$(MAKE) validate
