import json
import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolate_cryptoval_env(monkeypatch):
    monkeypatch.delenv("CRYPTOVAL_DATA_ROOT", raising=False)


@pytest.fixture
def load_fixture():
    def _load(name: str):
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    return _load
