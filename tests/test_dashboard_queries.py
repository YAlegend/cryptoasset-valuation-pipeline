import datetime as dt
import math

import duckdb
import pytest
from dashboard_fixture import DCF_INPUTS, LAST, PREV, build

from cryptoval.dashboard import queries as q
from cryptoval.methods.cashflow import three_stage_pv


@pytest.fixture
def con():
    con = duckdb.connect()
    build(con)
    yield con
    con.close()


def test_freshness_flags_a_stale_panel(con):
    fresh = q.freshness(con, LAST + dt.timedelta(days=1))
    assert fresh["panel_through"] == LAST and fresh["coin_age_through"] == LAST
    assert fresh["panel_lag_days"] == 1 and not fresh["stale"]
    assert q.freshness(con, LAST + dt.timedelta(days=5))["stale"]


def test_football_field_is_the_grid_range_on_each_methods_latest_applicable_date(con):
    field = q.football_field(con, "BTC").set_index("method_id")
    assert set(field.index) == {"realized_price", "defi_net_revenue_dcf"}
    rp, dcf = field.loc["realized_price"], field.loc["defi_net_revenue_dcf"]
    assert rp["metric_date"] == LAST and rp["low"] == rp["high"] == 60.0 and rp["is_circular"]
    # the DCF refused on LAST, so its latest APPLICABLE date is PREV, with a two-cell grid
    assert dcf["metric_date"] == PREV and (dcf["low"], dcf["high"], dcf["mid"]) == (40, 80, 60)
    assert dcf["n_cells"] == 2 and not dcf["is_circular"]


def test_signals_never_enter_the_football_field_or_heatmap(con):
    assert "slrv" not in set(q.football_field(con, "BTC")["method_id"])
    assert "slrv" not in set(q.agreement(con)["method_id"])


def test_refusals_list_methods_that_never_apply(con):
    assert q.refusals(con, "BTC").to_dict("records") == [
        {"method_id": "never_applies", "reason": "nope"}
    ]
    assert set(q.refusals(con, "USDC")["method_id"]) == {"never_applies", "realized_price"}
    assert q.football_field(con, "USDC").empty


def test_indicator_series_reads_panel_coin_age_and_signal_diagnostics(con):
    assert (q.indicator_series(con, "mvrv")["value"] == 2.0).all()
    assert (q.indicator_series(con, "liveliness")["value"] == 0.6).all()
    slrv = q.indicator_series(con, "slrv")
    assert slrv.to_dict("records") == [{"metric_date": LAST, "value": 0.05}]
    assert q.indicator_series(con, "puell_multiple")["value"].tolist() == pytest.approx([1.0] * 36)
    with pytest.raises(ValueError, match="unknown indicator"):
        q.indicator_series(con, "astrology")


def test_mayer_multiple_is_price_over_its_200_day_mean(con):
    mayer = q.indicator_series(con, "mayer_multiple")
    assert len(mayer) == 400 - 199
    assert mayer["value"].iloc[-1] == pytest.approx(499 / 399.5)


def test_agreement_is_log10_of_median_fair_value_over_that_days_price(con):
    got = q.agreement(con).set_index(["symbol", "method_id"])["log_ratio"]
    assert got[("BTC", "realized_price")] == pytest.approx(math.log10(60 / 499))
    assert got[("BTC", "defi_net_revenue_dcf")] == pytest.approx(math.log10(60 / 498))
    assert len(got) == 2


def test_dcf_inputs_come_from_the_latest_applicable_row(con):
    [row] = q.dcf_inputs(con).to_dict("records")
    assert row["symbol"] == "BTC" and row["metric_date"] == PREV
    assert row["annual_cash_flow_usd"] == 1e6 and row["supply"] == 1e4
    assert row["growth"] == [0.1, 0.05, 0.02] and row["terminal_multiple"] == 10.0


def test_dcf_value_matches_three_stage_pv_per_token():
    d = DCF_INPUTS
    pv, share = three_stage_pv(1e6, (0.1, 0.05, 0.02), 0.2, (3, 3, 4), 10.0)
    got = q.dcf_value(1e6, 1e4, d["growth"], 0.2, d["stage_years"], 10.0)
    assert got == pytest.approx((pv / 1e4, share))


def test_dcf_value_refuses_an_inverted_gordon():
    with pytest.raises(ValueError, match="Gordon"):
        q.dcf_value(1e6, 1e4, [0.1, 0.05, 0.03], 0.03, [3, 3, 4], None)


def test_freshness_on_an_empty_panel_is_stale_not_a_crash(con):
    con.execute("delete from fct_asset_metrics_daily")
    fresh = q.freshness(con, LAST)
    assert fresh["panel_through"] is None and fresh["stale"]


def test_overview_has_one_row_per_asset_with_changes_and_method_votes(con):
    got = q.overview(con).set_index("symbol")
    assert list(got.index) == ["USDC", "BTC"]  # largest market cap first
    btc, usdc = got.loc["BTC"], got.loc["USDC"]
    assert btc["price_date"] == LAST and btc["price_usd"] == 499.0
    assert btc["change_7d"] == pytest.approx(499 / 492 - 1)
    assert btc["change_30d"] == pytest.approx(499 / 469 - 1)
    # both BTC fair values (60/499, 60/498) sit below price: median upside is their median ratio
    assert btc["methods"] == 2 and btc["above"] == 0
    assert btc["median_upside"] == pytest.approx(
        10 ** ((math.log10(60 / 499) + math.log10(60 / 498)) / 2) - 1
    )
    assert btc["market_cap_usd"] == 4990.0  # today's price x yesterday's supply
    assert usdc["market_cap_usd"] == 5e5  # no supply: the stored market cap
    assert btc["sector"] == "pow_monetary" and len(btc["last_90d"]) == 90
    assert btc["fair_value"] == pytest.approx(499 * (1 + btc["median_upside"]))
    assert usdc["methods"] == 0 and math.isnan(usdc["median_upside"])


def test_overview_market_cap_falls_back_to_total_supply_when_the_panel_is_stale(con):
    con.execute("update fct_asset_metrics_daily set supply = null, market_cap_usd = null")
    got = q.overview(con).set_index("symbol")["market_cap_usd"]
    assert got["BTC"] == 499.0 * 20 and got["USDC"] == 6e5


def test_price_series_is_the_positive_price_history(con):
    s = q.price_series(con, "BTC")
    assert len(s) == 400 and s["price_usd"].iloc[-1] == 499.0 and s["metric_date"].iloc[-1] == LAST
