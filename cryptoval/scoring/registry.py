"""Model discovery and registration (spec 11.3).

Deliberately the same shape as cryptoval/methods/registry.py: a dict keyed by id, a class
decorator that registers an INSTANCE, and specs loaded from one YAML per model. An
executor who has read the method registry has read this one.
"""

from __future__ import annotations

import pathlib

import yaml

from cryptoval.scoring.base import Model, ModelSpec

# The model every other model is tested against (spec 11.1). Named once so the runner,
# the multiple-testing correction and the report cannot disagree about what "the
# baseline" means.
BASELINE_MODEL_ID = "naive_persistence"

MODELS: dict[str, Model] = {}


def register_instance(model: Model) -> Model:
    if model.spec.model_id in MODELS:
        raise ValueError(f"model id already registered: {model.spec.model_id}")
    MODELS[model.spec.model_id] = model
    return model


def register(cls: type) -> type:
    """Class decorator: instantiate and register, returning the class unchanged."""
    register_instance(cls())
    return cls


def load_model_specs(config_dir: str = "config/models") -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    for path in sorted(pathlib.Path(config_dir).glob("*.yml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = ModelSpec(
            model_id=raw["model_id"],
            family=raw["family"],
            params=dict(raw.get("params") or {}),
            citation=raw.get("citation", ""),
            uses_features=bool(raw.get("uses_features", True)),
        )
        if spec.model_id in specs:
            raise ValueError(f"duplicate model_id {spec.model_id} in {path}")
        specs[spec.model_id] = spec
    return specs
