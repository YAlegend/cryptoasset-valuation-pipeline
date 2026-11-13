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


def test_defillama_serves_net_revenue_distinctly_from_gross_fees():
    """Ruling G: the fees overview has no revenue field, so revenue needs its own call.

    If this fails, net_revenue_24h is silently all-null again and every revenue
    multiple built on it is meaningless (spec section 10.4).
    """
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        result = DefiLlamaSource().fetch(START, END, [], client)
    assert result.errors == ()
    assert result.frame["net_revenue_24h"].notna().sum() > 100
    both = result.frame.dropna(subset=["total_fees_24h", "net_revenue_24h"])
    assert (both["total_fees_24h"] != both["net_revenue_24h"]).any(), (
        "net revenue is identical to gross fees on every protocol: the revenue call "
        "is probably returning the fees payload"
    )
