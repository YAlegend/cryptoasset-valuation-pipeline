import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cryptoval.utxo.derive import (
    COINTIME_COLUMNS,
    derive,
    empty_cointime,
    supply_before,
    vendor_mismatches,
)

D0 = dt.date(2009, 1, 3)
DAYS = [D0 + dt.timedelta(days=i) for i in range(4)]
SAT = 100_000_000


def _toy(price=(np.nan, 10.0, 20.0, 40.0), spent_c1=100):
    blocks = pd.DataFrame(
        {"height": [0, 1, 2, 3, 4], "block_date": [DAYS[0], DAYS[1], DAYS[1], DAYS[2], DAYS[3]]}
    )
    created = pd.DataFrame(
        {"created_date": DAYS, "value_sat": [50 * SAT, 100 * SAT, 50 * SAT, 50 * SAT]}
    )
    spent = pd.DataFrame(
        {
            "created_date": [DAYS[0], DAYS[1]],
            "spent_date": [DAYS[2], DAYS[3]],
            "value_sat": [20 * SAT, spent_c1 * SAT],
            "coinblocks_destroyed": [60.0, 250.0],
        }
    )
    prices = pd.Series(price, index=DAYS)
    return derive(spent, created, blocks, prices, DAYS[-1], start=D0).set_index("metric_date")


def test_subsidy_schedule_halves_at_210000():
    got = supply_before(np.array([0, 1, 210_000, 210_001]))
    assert got.tolist() == [0, 50 * SAT, 210_000 * 50 * SAT, 210_000 * 50 * SAT + 25 * SAT]


def test_cointime_primitives_on_a_toy_chain():
    t = _toy()
    assert t["coinblocks_created"].tolist() == [0.0, 150.0, 150.0, 200.0]
    assert t["coinblocks_destroyed"].tolist() == [0.0, 0.0, 60.0, 250.0]
    assert np.isnan(t["liveliness"].iloc[0])
    assert t["liveliness"].iloc[1:].tolist() == pytest.approx([0.0, 0.2, 0.62])
    assert (t["liveliness"] + t["vaultedness"]).iloc[1:].tolist() == pytest.approx([1, 1, 1])
    assert t["utxo_supply"].tolist() == [50.0, 150.0, 180.0, 130.0]
    assert t["active_supply"].iloc[-1] == pytest.approx(130 * 0.62)


def test_realized_cap_values_each_coin_at_its_creation_price_and_pre_price_coins_at_zero():
    assert _toy()["utxo_realized_cap"].tolist() == [0.0, 1000.0, 2000.0, 3000.0]


def test_hodl_waves_by_supply_and_by_realized_cap():
    last = _toy().iloc[-1]
    assert last["hodl_0d"] == pytest.approx(50 / 130)
    assert last["hodl_1d_1w"] == pytest.approx(80 / 130)
    assert last["hodl_rc_0d"] == pytest.approx(2000 / 3000)
    assert last["hodl_rc_1d_1w"] == pytest.approx(1000 / 3000)
    assert last["hodl_6m_1y"] == 0.0
    bands = [c for c in COINTIME_COLUMNS if c.startswith("hodl_") and not c.startswith("hodl_rc_")]
    assert last[bands].sum() == pytest.approx(1.0)


def test_short_term_holders_and_realized_profit():
    t = _toy()
    assert t["sth_supply"].iloc[-1] == 130.0 and t["sth_realized_cap"].iloc[-1] == 3000.0
    # day 2 spends 20 BTC bought at 0 for 20 each; day 3 spends 100 bought at 10 for 40 each
    assert t["realized_profit_usd"].tolist() == [0.0, 0.0, 400.0, 3000.0]


def test_negative_unspent_raises():
    with pytest.raises(ValueError, match="negative unspent"):
        _toy(spent_c1=101)


def test_a_price_gap_after_the_first_price_raises():
    with pytest.raises(ValueError, match="price"):
        _toy(price=(np.nan, 10.0, np.nan, 40.0))


def test_empty_cointime_has_every_column():
    assert list(empty_cointime().columns) == COINTIME_COLUMNS and empty_cointime().empty


def test_vendor_mismatches_name_the_breach():
    days = pd.to_datetime(["2012-01-01", "2012-01-02"]).date
    ours = pd.DataFrame(
        {"metric_date": days, "utxo_supply": [100.0, 100.0], "utxo_realized_cap": [10.0, 10.0]}
    )
    vendor = pd.DataFrame(
        {"metric_date": days, "supply": [100.0, 101.0], "realized_cap": [10.0, 10.2]}
    )
    found = vendor_mismatches(ours, vendor)
    assert len(found) == 1 and "utxo_supply" in found[0] and "1 day" in found[0]


def test_a_spend_at_a_loss_realizes_no_profit():
    # day 3 spends 100 BTC created at 10 when the price is 5: a loss, not negative profit
    assert _toy(price=(np.nan, 10.0, 20.0, 5.0))["realized_profit_usd"].iloc[-1] == 0.0


def test_short_term_holders_exclude_coins_aged_155_days_or_more():
    days = [D0 + dt.timedelta(days=i) for i in range(200)]
    blocks = pd.DataFrame({"height": [0, 1], "block_date": [days[0], days[-1]]})
    created = pd.DataFrame({"created_date": [days[0], days[-1]], "value_sat": [50 * SAT, 50 * SAT]})
    spent = pd.DataFrame(
        columns=["created_date", "spent_date", "value_sat", "coinblocks_destroyed"]
    )
    prices = pd.Series(1.0, index=days)
    last = derive(spent, created, blocks, prices, days[-1], start=D0).iloc[-1]
    assert last["utxo_supply"] == 100.0
    assert last["sth_supply"] == 50.0 and last["sth_realized_cap"] == 50.0
