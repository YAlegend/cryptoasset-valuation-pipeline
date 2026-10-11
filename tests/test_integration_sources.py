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
