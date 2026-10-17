import pytest

import cryptoval.econometrics  # noqa: F401 -- loads the registry
from cryptoval.econometrics.registry import PAIRS, load_pair_specs
from cryptoval.econometrics.series import LEVEL_SERIES


def test_the_three_designed_pairs_are_registered():
    assert set(PAIRS) == {"mvrv_anchor", "liquidity_anchor", "metcalfe_anchor"}


def test_every_pair_names_series_the_levels_panel_actually_builds():
    for pair_id, spec in PAIRS.items():
        assert spec.dependent in LEVEL_SERIES, pair_id
        assert spec.independent in LEVEL_SERIES, pair_id


def test_metcalfe_forces_two_and_expects_two():
    """The law states value is proportional to the SQUARE of the users, so the
    coefficient-1 ratio is not Metcalfe's law and must not be labelled as it."""
    spec = PAIRS["metcalfe_anchor"]
    assert spec.forced_coefficient == 2.0
    assert spec.expected_coefficient == 2.0


def test_the_ratio_pairs_force_one_and_predict_no_particular_coefficient():
    for pair_id in ("mvrv_anchor", "liquidity_anchor"):
        assert PAIRS[pair_id].forced_coefficient == 1.0
        assert PAIRS[pair_id].expected_coefficient is None


def test_every_pair_carries_a_citation():
    for pair_id, spec in PAIRS.items():
        assert spec.citation.strip(), pair_id


def test_a_duplicate_pair_id_is_refused(tmp_path):
    for name in ("a.yml", "b.yml"):
        (tmp_path / name).write_text(
            "pair_id: same\ndependent: log_market_cap\n"
            "independent: log_volume\nforced_coefficient: 1.0\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError) as exc:
        load_pair_specs(str(tmp_path))
    assert "same" in str(exc.value)
