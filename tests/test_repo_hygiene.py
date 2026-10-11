import pathlib

import cryptoval

REPO = pathlib.Path(__file__).resolve().parent.parent
FORBIDDEN = ("train_test_split", ".sample(frac=", "shuffle=True")


def test_version_is_exposed():
    assert cryptoval.__version__ == "0.1.0"


def test_no_shuffled_splits_anywhere_in_package():
    offenders = []
    for path in (REPO / "cryptoval").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{path.relative_to(REPO)}: {token}")
    assert offenders == [], (
        "Time-ordered data must never be shuffled or randomly split "
        f"(spec section 11.2). Offenders: {offenders}"
    )
