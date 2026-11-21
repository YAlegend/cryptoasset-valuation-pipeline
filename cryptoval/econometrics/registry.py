"""Pair discovery (spec 11.4).

The same shape as cryptoval/scoring/registry.py and cryptoval/methods/registry.py: a
dict keyed by id, specs loaded from one YAML per pair. A pair is DATA, so adding the
next fundamental is a file, not a branch.
"""

from __future__ import annotations

import pathlib

import yaml

from cryptoval.econometrics.base import PairSpec

PAIRS: dict[str, PairSpec] = {}


def load_pair_specs(config_dir: str = "config/econometrics") -> dict[str, PairSpec]:
    specs: dict[str, PairSpec] = {}
    for path in sorted(pathlib.Path(config_dir).glob("*.yml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        expected = raw.get("expected_coefficient")
        spec = PairSpec(
            pair_id=raw["pair_id"],
            dependent=raw["dependent"],
            independent=raw["independent"],
            forced_coefficient=float(raw["forced_coefficient"]),
            expected_coefficient=None if expected is None else float(expected),
            citation=raw.get("citation", ""),
            params=dict(raw.get("params") or {}),
        )
        if spec.pair_id in specs:
            raise ValueError(f"duplicate pair_id {spec.pair_id} in {path}")
        specs[spec.pair_id] = spec
    return specs


PAIRS.update(load_pair_specs())
