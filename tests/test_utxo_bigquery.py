import datetime as dt

import pytest

from cryptoval.config import load_config
from cryptoval.utxo.bigquery import SQL_BUILDERS, BigQueryClient

START, END = dt.date(2026, 9, 1), dt.date(2026, 9, 30)


def test_every_query_is_bounded_to_the_range_end_exclusive():
    for sql in (b(START, END) for b in SQL_BUILDERS.values()):
        assert "'2026-09-01'" in sql and "'2026-10-01'" in sql


def test_spent_joins_inputs_to_deduplicated_transactions():
    sql = SQL_BUILDERS["btc_utxo_spent"](START, END)
    assert "crypto_bitcoin.inputs" in sql and "crypto_bitcoin.transactions" in sql
    assert "crypto_bitcoin.outputs" not in sql  # the 37% dearer join (spec Ruling 1)
    assert "GROUP BY `hash`" in sql  # BIP30 duplicate hashes collapse to one row
    for col in ("created_date", "spent_date", "value_sat", "coinblocks_destroyed", "n_inputs"):
        assert col in sql


def test_no_query_selects_a_wide_column():
    for sql in (b(START, END) for b in SQL_BUILDERS.values()):
        for wide in ("script_hex", "script_asm", "addresses", "SELECT *"):
            assert wide not in sql


def test_an_unconfigured_project_is_refused_before_any_client_is_built():
    with pytest.raises(ValueError, match="bigquery_project"):
        BigQueryClient("")


def test_settings_name_the_sandbox_project():
    settings, _ = load_config()
    assert settings.bigquery_project == "cryptoval-sandbox-3141"
