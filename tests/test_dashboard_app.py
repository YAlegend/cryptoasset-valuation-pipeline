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


def _text(at) -> str:
    return " ".join(m.value for m in [*at.markdown, *at.caption])


def _asset(symbol: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["asset"] = symbol
    return at.run()


def test_screener_lists_every_asset_against_a_fixture_database(data_root):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert DISCLAIMER in _text(at)
    assert list(at.dataframe[0].value["symbol"]) == ["USDC", "BTC"]
    assert at.metric[2].value == "2"  # assets


def test_asset_page_shows_price_models_indicators_and_dcf(data_root):
    at = _asset("btc")
    assert not at.exception, at.exception
    assert at.title[0].value == "BTC"
    assert [m.label for m in at.metric][:4] == [
        "Price",
        "Market cap",
        "Model fair value",
        "Models above price",
    ]
    assert [h.value for h in at.subheader] == [
        "Valuation models",
        "On-chain cycle indicators",
        "DCF calculator",
    ]
    assert DISCLAIMER in _text(at)


def test_an_asset_with_no_models_says_so(data_root):
    at = _asset("USDC")
    assert not at.exception, at.exception
    assert any("No valuation model" in i.value for i in at.info)
    assert at.metric[3].value == "none apply"


def test_an_unknown_asset_warns_instead_of_crashing(data_root):
    at = _asset("NOPE")
    assert not at.exception
    assert any("NOPE" in w.value for w in at.warning)


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
    at = _asset("BTC")
    assert [m.label for m in at.metric][4:] == [
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
    at = _asset("BTC")
    assert not at.exception
    assert at.slider[0].value == 0.40
    assert any("outside the slider" in c.value for c in at.caption)


def test_turning_a_model_family_off_recomputes_the_cards(data_root):
    at = _asset("BTC")
    assert at.metric[3].value == "0 of 2"
    at.pills(key="families").set_value(["Cash flow"]).run()
    assert not at.exception, at.exception
    assert at.metric[3].value == "0 of 1"  # realized_price (cost basis) left out
    at.pills(key="families").set_value([]).run()
    assert at.metric[3].value == "none apply"
    assert any("No valuation model" in i.value for i in at.info)


def test_a_clicked_model_opens_its_detail_and_close_returns(data_root):
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params.update(asset="BTC", model="defi_net_revenue_dcf")
    at.run()
    assert not at.exception, at.exception
    assert "Defi Net Revenue DCF" in [h.value for h in at.subheader]
    assert any("A paper, section 1" in c.value for c in at.caption)
    at.button(key="close_model").click().run()
    assert "model" not in at.query_params
    assert "Defi Net Revenue DCF" not in [h.value for h in at.subheader]


def test_compare_tab_lines_up_the_first_three_assets(data_root):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.multiselect[0].value == ["USDC", "BTC"]
    assert list(at.dataframe[-1].value["symbol"]) == ["USDC", "BTC"]
