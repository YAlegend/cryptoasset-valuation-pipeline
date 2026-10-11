PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup test test-integration lint fmt ingest backfill validate reconcile dbt all

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
