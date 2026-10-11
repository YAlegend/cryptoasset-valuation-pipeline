import numpy as np

from cryptoval.sources.defillama import (
    DefiLlamaSource,
    methodology_hash,
    parse_fees_overview,
)


def test_parses_protocols(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    assert len(df) == 3
    assert set(df["protocol"]) == {"Curve DEX", "Aave V2", "WBTC"}


def test_keeps_fees_and_net_revenue_as_distinct_columns(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    curve = df[df["protocol"] == "Curve DEX"].iloc[0]
    assert curve["total_fees_24h"] == 7415
    assert curve["net_revenue_24h"] == 1483


def test_missing_net_revenue_is_nan_not_a_fallback_to_gross_fees(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    aave = df[df["protocol"] == "Aave V2"].iloc[0]
    assert np.isnan(aave["net_revenue_24h"])
    assert aave["total_fees_24h"] == 5266


def test_methodology_hash_is_stable_across_key_order():
    a = {"methodology": {"Fees": "x", "Revenue": "y"}}
    b = {"methodology": {"Revenue": "y", "Fees": "x"}}
    assert methodology_hash(a) == methodology_hash(b)


def test_methodology_hash_changes_when_the_text_changes():
    a = {"methodology": {"Fees": "Trading fees paid by users"}}
    b = {"methodology": {"Fees": "Trading fees plus incentives"}}
    assert methodology_hash(a) != methodology_hash(b)


def test_empty_methodology_still_hashes(load_fixture):
    df = parse_fees_overview(load_fixture("defillama_fees_overview.json"))
    wbtc = df[df["protocol"] == "WBTC"].iloc[0]
    assert len(wbtc["methodology_hash"]) == 64


def test_source_metadata():
    s = DefiLlamaSource()
    assert s.name == "defillama"
    assert s.contract_path.endswith("defillama_fees.yml")
