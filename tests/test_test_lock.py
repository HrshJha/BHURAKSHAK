from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_locked_test_path_is_referenced_only_by_final_eval_reader() -> None:
    """Grep guard: production readers must not mention the locked corpus path."""
    matches = []
    for folder in (ROOT / "src", ROOT / "scripts", ROOT / "notebooks"):
        for path in folder.rglob("*"):
            if not path.is_file() or path.name == "final_eval.py":
                continue
            if path.suffix not in {".py", ".ipynb"}:
                continue
            if "data/heldout_locked" in path.read_text(encoding="utf-8", errors="ignore"):
                matches.append(path.relative_to(ROOT).as_posix())
    assert matches == []
