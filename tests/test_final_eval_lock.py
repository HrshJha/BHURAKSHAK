from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.final_eval import claim_eval


def test_final_eval_claim_is_one_use_and_override_is_recorded(tmp_path: Path) -> None:
    lock = tmp_path / "test_lock.json"
    lock.write_text(json.dumps({"evals_run": 0}), encoding="utf-8")
    assert claim_eval(lock, False)["evals_run"] == 1
    with pytest.raises(SystemExit, match="already evaluated"):
        claim_eval(lock, False)
    burned_again = claim_eval(lock, True)
    assert burned_again["evals_run"] == 2
    assert burned_again["burn_override_used"] is True
