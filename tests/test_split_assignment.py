"""split-assignment producer tests.

Proves the audit CRITICAL is closed: data/features/split_assignment.csv has a
single reproducible producer (scripts/make_split_assignment.py), the
committed file equals the regenerated file, and both named splits satisfy
 unit-exclusivity with real class support.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SPLIT_CSV = REPO / "data" / "features" / "split_assignment.csv"

spec = importlib.util.spec_from_file_location("make_split_assignment", REPO / "scripts" / "make_split_assignment.py")
msa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msa)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def split_csv() -> pd.DataFrame:
    return pd.read_csv(SPLIT_CSV)


def test_regenerating_twice_gives_identical_bytes(tmp_path: Path) -> None:
    out_a, out_b = tmp_path / "a.csv", tmp_path / "b.csv"
    for out in (out_a, out_b):
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "make_split_assignment.py"), "--out", str(out)],
            check=True, capture_output=True,
        )
    assert _sha256(out_a) == _sha256(out_b), "the producer must be byte-reproducible"


def test_committed_csv_equals_regenerated_csv(tmp_path: Path) -> None:
    out = tmp_path / "regen.csv"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_split_assignment.py"), "--out", str(out)],
        check=True, capture_output=True,
    )
    committed = _sha256(SPLIT_CSV)
    assert committed == _sha256(out), "committed split_assignment.csv is stale — re-run make_split_assignment.py"


def test_both_split_columns_present_and_complete(split_csv: pd.DataFrame) -> None:
    assert {"event_id", "scenario_family", "split", "split_family_balanced"} <= set(split_csv.columns)
    for col in ("split", "split_family_balanced"):
        assert set(split_csv[col].unique()) == {"train", "validation", "test"}


def test_regime_holdout_has_zero_group_leakage(split_csv: pd.DataFrame) -> None:
    from src.evaluation.splits import assert_no_leakage

    assert_no_leakage(split_csv, split_csv["split"], "event_id")


def test_legacy_balanced_has_zero_group_leakage(split_csv: pd.DataFrame) -> None:
    from src.evaluation.splits import assert_no_leakage

    assert_no_leakage(split_csv, split_csv["split_family_balanced"], "event_id")


def test_regime_test_split_holds_out_whole_families(split_csv: pd.DataFrame) -> None:
    """: the test regimes must not appear in train/validation at all."""
    test_fams = set(split_csv.loc[split_csv["split"] == "test", "scenario_family"])
    assert test_fams == {"E_rapid_subsidence", "E_accelerating_subsidence", "E_stable_ground"}
    other = split_csv[split_csv["split"] != "test"]
    assert not other["scenario_family"].isin(test_fams).any()


def test_regime_test_split_has_all_three_classes(split_csv: pd.DataFrame) -> None:
    """Class support: the holdout must exercise NORMAL, WARNING and CRITICAL."""
    store = pd.read_parquet(REPO / "data" / "features" / "features_v2.parquet", columns=["event_id", "risk_label"])
    merged = store.merge(split_csv[["event_id", "split"]], on="event_id", validate="many_to_one")
    test_counts = merged[merged["split"] == "test"]["risk_label"].value_counts().to_dict()
    assert set(test_counts) == {"NORMAL", "WARNING", "CRITICAL"}
    assert all(v >= 1000 for v in test_counts.values()), f"thin class support in regime test: {test_counts}"


def test_regime_split_is_disjoint_by_generation_parameters(split_csv: pd.DataFrame) -> None:
    from src.evaluation.splits import assert_no_leakage

    assert "generation_parameter_id" in split_csv
    assert_no_leakage(split_csv, split_csv["split"], "generation_parameter_id")


def test_manifest_records_the_producer() -> None:
    import json

    manifest = json.loads((REPO / "data" / "synthetic" / "dataset_manifest.json").read_text())
    sd = manifest["split_definition"]
    assert sd["producer"] == "scripts/make_split_assignment.py"
    assert sd["default_split"] == "regime_holdout"
    assert "assigned in " not in json.dumps(sd)
