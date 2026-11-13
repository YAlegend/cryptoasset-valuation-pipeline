import numpy as np
import pandas as pd
import pytest

from cryptoval.config import Asset
from cryptoval.rollup import REVENUE_COLUMNS, assert_no_aggregate_collision, rollup_protocols

UNI = Asset(
    symbol="UNI",
    sector="defi",
    consensus="other",
    cm_id="uni",
    gecko_id="uniswap",
    defillama_slugs=("Uniswap V4", "Uniswap V3", "Uniswap V2"),
)
LDO = Asset(
    symbol="LDO",
    sector="defi",
    consensus="other",
    cm_id="ldo",
    gecko_id="lido-dao",
    defillama_slugs=("Lido",),
)


def _feed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protocol": ["Uniswap V4", "Uniswap V3", "Uniswap V2", "Lido", "Hegic"],
            "metric_date": pd.to_datetime(["2026-10-11"] * 5, utc=True),
            "total_fees_24h": [1370164.0, 450744.0, 39866.0, 1628206.0, 1.36],
            "total_fees_30d": [1.0e7, 1.2e7, 1.0e6, 4.0e7, -71888.64],
            "net_revenue_24h": [0.0, 92267.0, 0.0, 99746.0, 1.36],
        }
    )


def test_versions_are_summed_into_one_symbol():
    out = rollup_protocols(_feed(), [UNI, LDO])
    uni = out.set_index("symbol").loc["UNI"]
    # A DCF on Uniswap V3 alone would be wrong and declining: V4 now carries three times
    # V3's fees and the mix is still shifting (spec 5.1).
    assert uni["total_fees_24h"] == pytest.approx(1370164.0 + 450744.0 + 39866.0)
    assert uni["net_revenue_24h"] == pytest.approx(92267.0)
    assert uni["protocol_count"] == 3


def test_an_unmapped_protocol_is_excluded():
    out = rollup_protocols(_feed(), [UNI, LDO])
    # Hegic maps to no asset we value, so it must not appear.
    assert set(out["symbol"]) == {"UNI", "LDO"}


def test_an_asset_with_no_slugs_is_absent_rather_than_zero():
    btc = Asset(symbol="BTC", sector="pow_monetary", consensus="pow", cm_id="btc")
    out = rollup_protocols(_feed(), [btc])
    assert out.empty
    assert list(out.columns) == ["symbol", "metric_date", *REVENUE_COLUMNS, "protocol_count"]


def test_an_all_null_group_stays_null_rather_than_summing_to_zero():
    feed = _feed()
    feed.loc[feed["protocol"].str.startswith("Uniswap"), "net_revenue_24h"] = np.nan
    out = rollup_protocols(feed, [UNI])
    # Spec global constraint: a missing value is never coerced to 0.0. A zero here would
    # read as "this protocol retains nothing", which is a different claim.
    assert np.isnan(out.set_index("symbol").loc["UNI", "net_revenue_24h"])


def test_a_configured_slug_missing_from_the_feed_lowers_the_count():
    feed = _feed()
    feed = feed[feed["protocol"] != "Uniswap V2"]
    out = rollup_protocols(feed, [UNI])
    assert out.set_index("symbol").loc["UNI", "protocol_count"] == 2


def test_an_aggregate_alongside_its_components_is_refused():
    with pytest.raises(ValueError, match="would double-count"):
        assert_no_aggregate_collision(("Uniswap", "Uniswap V3"), "UNI")


def test_sibling_versions_do_not_collide():
    assert assert_no_aggregate_collision(("Uniswap V4", "Uniswap V3"), "UNI") is None
    assert assert_no_aggregate_collision(("Curve DEX", "Curve LlamaLend V2"), "CRV") is None
    assert assert_no_aggregate_collision(("Sky Lending", "Sky Money"), "MKR") is None


def test_a_prefix_that_is_not_a_word_boundary_does_not_collide():
    # "Aave" is a prefix of "Aavegotchi" as a string but not as a protocol family.
    assert assert_no_aggregate_collision(("Aave", "Aavegotchi"), "AAVE") is None


def test_the_rollup_refuses_a_colliding_config():
    bad = Asset(
        symbol="UNI",
        sector="defi",
        consensus="other",
        cm_id="uni",
        gecko_id="uniswap",
        defillama_slugs=("Uniswap", "Uniswap V3"),
    )
    with pytest.raises(ValueError, match="would double-count"):
        rollup_protocols(_feed(), [bad])
