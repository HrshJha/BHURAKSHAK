"""T-021 acceptance tests — manifest carries the exact §10.1 field list."""

from __future__ import annotations

import json

import pytest

from src.config import feature_schema_config
from src.simulator.manifest import (
    REQUIRED_FIELDS,
    build_manifest,
    validate_manifest,
    write_manifest,
)


def _manifest() -> dict:
    return build_manifest(
        dataset_version="v0.1.0",
        random_seed=42,
        counts_per_scenario={"stable_ground": 10, "slow_subsidence": 8},
    )


def test_required_fields_exact() -> None:
    expected = [
        "dataset_version",
        "generator_version",
        "random_seed",
        "physics_parameters",
        "noise_parameters",
        "fault_parameters",
        "scenario_parameters",
        "source_data_versions",
        "feature_schema_version",
        "split_definition",
    ]
    assert REQUIRED_FIELDS == expected, "§10.1 defines exactly these manifest fields"


def test_build_manifest_has_all_fields() -> None:
    m = _manifest()
    assert validate_manifest(m) == [], "built manifest must satisfy §10.1"
    assert m["random_seed"] == 42
    assert m["feature_schema_version"] == feature_schema_config()["feature_schema_version"]
    assert len(m["physics_parameters"]) == 10, "§10 ten parameters"
    assert set(m["fault_parameters"]["fault_types"]) >= {"BIAS", "STUCK", "DROPOUT", "SPIKE", "DRIFT"}
    assert m["source_data_versions"] == {"sentinel1": None, "dgps": None, "hardware": None}


def test_validate_detects_missing_fields() -> None:
    m = _manifest()
    del m["random_seed"]
    del m["split_definition"]
    missing = validate_manifest(m)
    assert set(missing) == {"random_seed", "split_definition"}


def test_write_refuses_incomplete_manifest(tmp_path) -> None:
    m = _manifest()
    del m["generator_version"]
    with pytest.raises(ValueError, match="§10.1"):
        write_manifest(m, tmp_path / "manifest.json")


def test_write_manifest_roundtrip(tmp_path) -> None:
    m = _manifest()
    out = write_manifest(m, tmp_path / "sub" / "dataset_manifest.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == m
    assert out.name == "dataset_manifest.json"


def test_split_definition_default_is_parameter_holdout() -> None:
    m = _manifest()
    assert m["split_definition"]["type"] == "synthetic_parameter_holdout"
