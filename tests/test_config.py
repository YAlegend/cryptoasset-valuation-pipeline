import pytest

from cryptoval.config import SECTORS, load_config


def test_loads_settings():
    settings, _ = load_config("config")
    assert settings.trailing_refetch_days == 30
    assert settings.bigquery_monthly_byte_budget == 900_000_000_000
    assert settings.resolver.layer("bronze").endswith("/bronze")


def test_loads_full_universe():
    _, assets = load_config("config")
    crypto = [a for a in assets if not a.is_equity]
    equities = [a for a in assets if a.is_equity]
    assert len(crypto) == 24      # 23 valuation targets plus SOL, which has no CM id
    assert len(equities) == 5


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
