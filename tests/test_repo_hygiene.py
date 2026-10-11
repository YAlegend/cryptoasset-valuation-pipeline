import ast
import pathlib

import cryptoval

REPO = pathlib.Path(__file__).resolve().parent.parent
EXCLUDED_DIRS = {".venv", ".superpowers", ".git", "dbt_packages", "node_modules", "build"}
FORBIDDEN_CALLS = {
    "train_test_split",
    "shuffle",
    "permutation",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
}


def _is_literal_false(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _python_files(root: pathlib.Path):
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel_parts):
            continue
        yield path


def find_shuffle_violations(root: pathlib.Path) -> tuple[int, list[str]]:
    """Scan Python files under root; return (files_scanned, ["path: what matched"])."""
    scanned = 0
    offenders: list[str] = []
    for path in _python_files(root):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name in FORBIDDEN_CALLS:
                offenders.append(f"{rel}:{node.lineno}: call to {name}()")
            for kw in node.keywords:
                if kw.arg == "shuffle" and not _is_literal_false(kw.value):
                    offenders.append(f"{rel}:{node.lineno}: shuffle= is not literal False")
                if kw.arg == "frac" and name == "sample":
                    offenders.append(f"{rel}:{node.lineno}: .sample(frac=...)")
            if (
                name == "sample"
                and isinstance(node.func, ast.Attribute)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, float)
            ):
                offenders.append(f"{rel}:{node.lineno}: .sample(<float fraction>)")
    return scanned, offenders


def test_version_is_exposed():
    assert cryptoval.__version__ == "0.1.0"


def test_no_shuffled_splits_anywhere_in_repository():
    scanned, offenders = find_shuffle_violations(REPO)
    assert scanned > 0, "hygiene scan visited no files; the scan root is wrong"
    assert offenders == [], (
        "Time-ordered data must never be shuffled or randomly split "
        f"(spec section 11.2). Offenders: {offenders}"
    )


def _scan_snippet(tmp_path: pathlib.Path, code: str) -> list[str]:
    (tmp_path / "snippet.py").write_text(code, encoding="utf-8")
    scanned, offenders = find_shuffle_violations(tmp_path)
    assert scanned == 1
    return offenders


def test_detector_flags_violations(tmp_path):
    snippets = [
        "train_test_split(X, y)",
        "sklearn.model_selection.train_test_split(X, y)",
        "np.random.shuffle(a)",
        "random.shuffle(a)",
        "np.random.permutation(10)",
        "sklearn.utils.shuffle(a)",
        "KFold(shuffle=True)",
        "KFold(shuffle = True)",
        "KFold(shuffle=1)",
        "ShuffleSplit(n_splits=3)",
        "StratifiedShuffleSplit(n_splits=3)",
        "df.sample(frac=1)",
        "df.sample(frac = 1.0)",
        "df.sample(1.0)",
    ]
    for code in snippets:
        assert _scan_snippet(tmp_path, code + "\n"), f"detector missed: {code}"


def test_detector_allows_safe_code_and_ignores_strings(tmp_path):
    safe = (
        "KFold(shuffle=False)\n"
        "df.sample(n=5)\n"
        'text = "train_test_split(X, y)  # shuffle=True"\n'
        "# np.random.shuffle(a)\n"
    )
    assert _scan_snippet(tmp_path, safe) == []


def test_detector_skips_excluded_directories(tmp_path):
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "bad.py").write_text("train_test_split(X, y)\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    scanned, offenders = find_shuffle_violations(tmp_path)
    assert (scanned, offenders) == (1, [])
