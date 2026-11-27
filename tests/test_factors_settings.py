import pytest

from cryptoval.factors.settings import load_factor_config


def test_the_repository_config_loads():
    config = load_factor_config()
    assert config.exclude["wbtc"] == "wrapper"
    assert config.exclude["usdt"] == "pegged"
    assert (config.min_assets, config.deciles, config.newey_west_lags) == (50, 10, 4)


def test_an_id_excluded_twice_is_refused(tmp_path):
    path = tmp_path / "f.yml"
    path.write_text("exclude:\n  pegged: [usdt]\n  wrapper: [usdt]\n")
    with pytest.raises(ValueError, match="usdt"):
        load_factor_config(str(path))


def test_an_unknown_key_is_refused(tmp_path):
    path = tmp_path / "f.yml"
    path.write_text("exclude: {}\nmin_asets: 3\n")
    with pytest.raises(TypeError):
        load_factor_config(str(path))
