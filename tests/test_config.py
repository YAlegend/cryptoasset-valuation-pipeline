import pytest

from cryptoval.config import METRIC_GROUPS, SECTORS, Asset, _validate, load_config


def test_loads_settings():
    settings, _ = load_config("config")
    assert settings.trailing_refetch_days == 30
    assert settings.bigquery_monthly_byte_budget == 900_000_000_000
    assert settings.resolver.layer("bronze").endswith("/bronze")


def test_loads_full_universe():
    _, assets = load_config("config")
    crypto = [a for a in assets if not a.is_equity]
    equities = [a for a in assets if a.is_equity]
    assert len(crypto) == 24  # 23 valuation targets plus SOL, which has no CM id
    # five single-name equities plus gold and the dollar index (spec 15 Q1, resolved)
    assert len(equities) == 7


def test_sol_has_no_coinmetrics_id_recording_the_documented_gap():
    _, assets = load_config("config")
    sol = next(a for a in assets if a.symbol == "SOL")
    assert sol.cm_id is None
    assert sol.gecko_id == "solana"


def test_every_sector_is_known():
    _, assets = load_config("config")
    for a in assets:
        if not a.is_equity:
            assert a.sector in SECTORS, f"{a.symbol} has unknown sector {a.sector}"


def test_consensus_drives_cost_of_production_applicability():
    _, assets = load_config("config")
    btc = next(a for a in assets if a.symbol == "BTC")
    eth = next(a for a in assets if a.symbol == "ETH")
    assert btc.consensus == "pow"
    assert eth.consensus == "pos"


def test_usdc_is_flagged_as_the_stablecoin_control():
    _, assets = load_config("config")
    usdc = next(a for a in assets if a.symbol == "USDC")
    assert usdc.is_stablecoin is True


def test_unknown_sector_is_rejected(tmp_path):
    (tmp_path / "settings.yml").write_text("data_root: ./data\n")
    (tmp_path / "assets.yml").write_text(
        "assets:\n  - {symbol: FOO, cm_id: foo, gecko_id: foo, sector: nonsense, consensus: pow}\n"
    )
    with pytest.raises(ValueError, match="unknown sector"):
        load_config(str(tmp_path))


def test_asset_without_any_source_id_is_rejected(tmp_path):
    (tmp_path / "settings.yml").write_text("data_root: ./data\n")
    (tmp_path / "assets.yml").write_text(
        "assets:\n  - {symbol: FOO, cm_id: null, gecko_id: null, sector: defi, consensus: other}\n"
    )
    with pytest.raises(ValueError, match="no usable source id"):
        load_config(str(tmp_path))


def test_empty_data_root_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTOVAL_DATA_ROOT", raising=False)
    (tmp_path / "settings.yml").write_text('data_root: ""\n')
    (tmp_path / "assets.yml").write_text("assets: []\n")
    with pytest.raises(ValueError, match="data_root"):
        load_config(str(tmp_path))


def test_trx_draws_network_metrics_from_the_real_tron_series():
    _, assets = load_config()
    trx = next(a for a in assets if a.symbol == "TRX")
    # trx_eth is a near-dead ERC-20 wrapper: market cap NULL, volume 0, $165K credible max
    # against real TRON's $254M/day (spec 5.1).
    assert trx.source_for("network") == "cm:trx"
    assert trx.source_for("volume") == "cm:trx"
    # Coin Metrics serves no free market cap for `trx`, so it comes from CoinGecko.
    assert trx.source_for("valuation") == "coingecko"
    assert trx.cm_ids == ("trx",)


def test_avax_takes_volume_from_a_different_coin_metrics_id():
    _, assets = load_config()
    avax = next(a for a in assets if a.symbol == "AVAX")
    # avaxc is the genuine C-Chain, but serves no reported spot volume (0% of the panel).
    assert avax.source_for("network") == "cm:avaxc"
    assert avax.source_for("valuation") == "cm:avaxc"
    assert avax.source_for("volume") == "cm:avax"
    assert avax.cm_ids == ("avax", "avaxc")


def test_an_unmapped_asset_defaults_every_group_to_its_primary_id():
    _, assets = load_config()
    btc = next(a for a in assets if a.symbol == "BTC")
    assert [btc.source_for(g) for g in METRIC_GROUPS] == ["cm:btc"] * 3
    assert btc.cm_ids == ("btc",)


def test_an_asset_with_no_coin_metrics_id_falls_back_to_coingecko():
    _, assets = load_config()
    sol = next(a for a in assets if a.symbol == "SOL")
    assert sol.source_for("network") == "none"
    assert sol.source_for("valuation") == "coingecko"
    assert sol.cm_ids == ()


def test_an_unknown_metric_group_is_rejected():
    with pytest.raises(ValueError, match="unknown metric group"):
        _validate(Asset(symbol="X", sector="defi", cm_id="x", metric_sources={"nope": "cm:x"}))


def test_a_malformed_metric_source_is_rejected():
    with pytest.raises(ValueError, match="unusable metric source"):
        _validate(Asset(symbol="X", sector="defi", cm_id="x", metric_sources={"volume": "CM:X"}))


def test_a_valuation_group_pointing_at_coingecko_requires_a_gecko_id():
    with pytest.raises(ValueError, match="names coingecko but has no gecko_id"):
        _validate(
            Asset(symbol="X", sector="defi", cm_id="x", metric_sources={"valuation": "coingecko"})
        )


def test_a_bare_string_slug_list_is_rejected(tmp_path):
    """`defillama_slugs: Lido` would become ('L','i','d','o') and roll up to no revenue."""
    (tmp_path / "settings.yml").write_text("data_root: ./data\n")
    (tmp_path / "assets.yml").write_text(
        "assets:\n  - {symbol: LDO, cm_id: ldo, gecko_id: lido-dao, sector: defi, "
        "consensus: other, defillama_slugs: Lido}\n"
    )
    with pytest.raises(ValueError, match="defillama_slugs must be a list"):
        load_config(str(tmp_path))


def test_a_list_of_slugs_is_still_accepted(tmp_path):
    (tmp_path / "settings.yml").write_text("data_root: ./data\n")
    (tmp_path / "assets.yml").write_text(
        "assets:\n  - {symbol: LDO, cm_id: ldo, gecko_id: lido-dao, sector: defi, "
        "consensus: other, defillama_slugs: [Lido]}\n"
    )
    _, assets = load_config(str(tmp_path))
    assert assets[0].defillama_slugs == ("Lido",)


def test_an_empty_slug_is_rejected():
    with pytest.raises(ValueError, match="empty entry"):
        _validate(Asset(symbol="X", sector="defi", cm_id="x", defillama_slugs=("Lido", "  ")))


def _load_one(tmp_path, genesis: str):
    (tmp_path / "settings.yml").write_text("data_root: ./data\n")
    (tmp_path / "assets.yml").write_text(
        "assets:\n  - {symbol: X, cm_id: x, gecko_id: x, sector: defi, consensus: other, "
        f"genesis_date: {genesis}}}\n"
    )
    return load_config(str(tmp_path))[1][0]


def test_genesis_date_loads_as_a_date_and_btc_has_one():
    import datetime as dt

    btc = next(a for a in load_config()[1] if a.symbol == "BTC")
    assert btc.genesis_date == dt.date(2009, 1, 3)
    # Absent where the launch is not a matter of record, so the power law refuses it.
    assert next(a for a in load_config()[1] if a.symbol == "BNB").genesis_date is None


# 2009-13-45 is rejected by the YAML parser itself, the other two by _validate.
@pytest.mark.parametrize("bad", ["2009-13-45", "not-a-date", "2009-01-03T10:00:00"])
def test_a_malformed_genesis_date_fails_config_load(tmp_path, bad):
    with pytest.raises(ValueError, match="genesis_date|month"):
        _load_one(tmp_path, bad)
