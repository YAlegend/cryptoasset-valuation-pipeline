import csv
import pathlib

from cryptoval.config import Asset, load_config
from cryptoval.seeds import write_asset_seeds


def _rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_asset_seed_carries_one_row_per_symbol(tmp_path):
    _, assets = load_config()
    paths = write_asset_seeds(assets, str(tmp_path))
    rows = _rows(paths[0])
    assert len(rows) == len(assets)
    assert len({r["symbol"] for r in rows}) == len(assets)


def test_the_metric_source_seed_is_long_not_wide(tmp_path):
    assets = [
        Asset(symbol="AVAX", sector="smart_contract_l1", consensus="pos", cm_id="avaxc",
              gecko_id="avalanche-2", metric_sources={"volume": "cm:avax"}),
    ]
    rows = _rows(write_asset_seeds(assets, str(tmp_path))[1])
    # One row per (symbol, group), so a vendor id serving two groups appears twice and the
    # join in staging stays a simple equi-join.
    assert len(rows) == 3
    by_group = {r["metric_group"]: r for r in rows}
    assert by_group["network"]["source_id"] == "avaxc"
    assert by_group["valuation"]["source_id"] == "avaxc"
    assert by_group["volume"]["source_id"] == "avax"
    assert {r["source_kind"] for r in rows} == {"cm"}


def test_a_coingecko_valuation_source_is_recorded_as_such(tmp_path):
    assets = [
        Asset(symbol="TRX", sector="smart_contract_l1", consensus="pos", cm_id="trx",
              gecko_id="tron", metric_sources={"valuation": "coingecko"}),
    ]
    rows = {r["metric_group"]: r for r in _rows(write_asset_seeds(assets, str(tmp_path))[1])}
    assert rows["valuation"]["source_kind"] == "coingecko"
    assert rows["valuation"]["source_id"] == "tron"
    assert rows["network"]["source_kind"] == "cm"
    assert rows["network"]["source_id"] == "trx"


def test_slugs_are_serialised_as_a_pipe_delimited_field(tmp_path):
    assets = [
        Asset(symbol="UNI", sector="defi", consensus="other", cm_id="uni", gecko_id="uniswap",
              defillama_slugs=("Uniswap V4", "Uniswap V3")),
    ]
    row = _rows(write_asset_seeds(assets, str(tmp_path))[0])[0]
    # Pipe, not comma: these are CSV files and the slugs contain spaces but never pipes.
    assert row["defillama_slugs"] == "Uniswap V4|Uniswap V3"


def test_a_network_source_of_none_is_omitted_rather_than_written_as_a_join_key(tmp_path):
    sol = Asset(symbol="SOL", sector="smart_contract_l1", consensus="pos", cm_id=None,
                gecko_id="solana")
    rows = _rows(write_asset_seeds([sol], str(tmp_path))[1])
    # SOL has no network source at all. A row with source_id "none" would join to nothing
    # and read as a configuration error later; absence is the honest encoding.
    assert {r["metric_group"] for r in rows} == {"valuation", "volume"}


def test_the_seeds_are_deterministic(tmp_path):
    _, assets = load_config()
    first = [pathlib.Path(p).read_text(encoding="utf-8")
             for p in write_asset_seeds(assets, str(tmp_path / "a"))]
    second = [pathlib.Path(p).read_text(encoding="utf-8")
              for p in write_asset_seeds(assets, str(tmp_path / "b"))]
    # Committed generated files must not churn, or every run dirties the tree.
    assert first == second


def test_the_committed_seeds_match_what_config_generates(tmp_path):
    _, assets = load_config()
    fresh = write_asset_seeds(assets, str(tmp_path))
    repo = pathlib.Path(__file__).parent.parent
    for path in fresh:
        committed = repo / "dbt" / "seeds" / pathlib.Path(path).name
        assert committed.read_bytes() == pathlib.Path(path).read_bytes(), (
            f"{committed} is stale against config/assets.yml; run `make seed` and commit it"
        )
