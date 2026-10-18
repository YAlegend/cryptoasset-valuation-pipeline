import pathlib

import duckdb
import pytest
from dashboard_fixture import build
from streamlit.testing.v1 import AppTest

APP = str(pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
DISCLAIMER = "not investment advice"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    con = duckdb.connect(str(tmp_path / "cryptoval.duckdb"))
    build(con)
    con.close()
    monkeypatch.setenv("CRYPTOVAL_DATA_ROOT", str(tmp_path))
    return tmp_path


def _markdown(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_app_renders_every_tab_against_a_fixture_database(data_root):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert DISCLAIMER in _markdown(at)
    assert len(at.tabs) == 4


def test_app_without_a_database_says_how_to_build_one(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTOVAL_DATA_ROOT", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("make all" in e.value for e in at.error)


def test_app_holds_no_connection_open_between_reads(data_root):
    AppTest.from_file(APP, default_timeout=60).run()
    # a read-write open fails while any other connection holds the file
    duckdb.connect(str(data_root / "cryptoval.duckdb")).close()


def test_dcf_tab_prices_the_row_and_refuses_an_inverted_gordon(data_root):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert [m.label for m in at.metric] == [
        "Fair value per token",
        "Latest price",
        "Terminal share of value",
    ]
    assert not at.error
    at.slider[1].set_value(0.0)  # terminal multiple 0 -> Gordon growth
    at.slider[0].set_value(0.02)  # discount at stage-3 growth
    at.run()
    assert any("Gordon" in e.value for e in at.error)


def test_a_stored_dcf_setting_outside_the_slider_range_is_clamped(data_root):
    con = duckdb.connect(str(data_root / "cryptoval.duckdb"))
    con.execute(
        "update fct_valuation_estimate set inputs_used = replace(inputs_used, "
        "'\"discount_rate\": 0.2', '\"discount_rate\": 0.55') "
        "where method_id = 'defi_net_revenue_dcf'"
    )
    con.close()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.slider[0].value == 0.40
    assert any("outside the slider" in c.value for c in at.caption)
