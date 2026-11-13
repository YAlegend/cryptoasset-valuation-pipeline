from cryptoval.cards import load_card_extras, render_card, write_cards
from cryptoval.methods.registry import load_method_specs


def test_a_card_states_the_formula_citation_and_failure_modes():
    spec = load_method_specs()["metcalfe_estimated"]
    card = render_card(spec, {"failure_modes": ["an active address is not a user"]})
    assert "metcalfe_estimated" in card
    assert spec.citation in card
    assert "an active address is not a user" in card
    assert "market_cap_usd" in card  # data dependencies are listed


def test_a_circular_method_says_so_prominently():
    """Spec 14: such metrics are labelled diagnostics, not independent fair values."""
    spec = load_method_specs()["crypto_qtm_panel_velocity"]
    card = render_card(spec, {})
    assert "Circular" in card
    assert "velocity_total" in card
    assert "not independent evidence" in card


def test_a_non_circular_method_does_not_claim_circularity():
    card = render_card(load_method_specs()["metcalfe_estimated"], {})
    assert "Circular" not in card


def test_a_card_is_written_per_method(tmp_path):
    specs = load_method_specs()
    paths = write_cards(specs, str(tmp_path))
    assert len(paths) == len(specs)
    assert all(p.endswith(".md") for p in paths)


def test_the_hand_written_sections_live_in_config_so_regeneration_cannot_destroy_them(tmp_path):
    """`make cards` overwrites every card, so the judgment cannot live in the card."""
    extras = load_card_extras()
    specs = load_method_specs()
    assert set(extras) == set(specs)
    missing = [m for m in specs if not (extras[m]["failure_modes"] and extras[m]["assumptions"])]
    assert not missing, f"method(s) with no hand-written card sections: {missing}"
    cards = write_cards(specs, str(tmp_path), extras)
    assert not any("(to be written)" in open(p, encoding="utf-8").read() for p in cards)


def test_every_registered_method_has_a_config_and_vice_versa():
    """A method with no config cannot be run; a config with no method is a dead card."""
    import cryptoval.methods  # noqa: F401 -- import registers every method family
    from cryptoval.methods.registry import METHODS

    assert set(METHODS) == set(load_method_specs())


def test_every_method_has_a_card_on_disk_and_the_registry_holds_all_41():
    """`make cards` must have been re-run: a config with no card is an unpublished method.

    41 = 14 from Plan 2 + 21 from Plan 3 + 6 from Plan 3b (Family E(b)). A count that drifts
    without a plan entry is a method someone added without a card or a config.
    """
    import pathlib

    import cryptoval.methods  # noqa: F401 -- import registers every method family
    from cryptoval.methods.registry import METHODS

    cards = {p.stem for p in pathlib.Path("docs/methods").glob("*.md")}
    missing = sorted(set(load_method_specs()) - cards)
    assert not missing, f"method(s) with no card on disk -- run `make cards`: {missing}"
    assert len(METHODS) == 41
