""" tests — / model registry with dataset-version resolution."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.risk.model_registry import (
    DEFAULT_MANIFEST_PATHS,
    FR14_FIELDS,
    ModelRegistry,
    ModelRegistryError,
    REPO_ROOT,
)

REAL_SCHEMA_VERSION = __import__("src.config", fromlist=["feature_schema_config"]).feature_schema_config()["feature_schema_version"]
REAL_DATASET_VERSION = json.loads((REPO_ROOT / "data" / "synthetic" / "dataset_manifest.json").read_text())["dataset_version"]


@pytest.fixture()
def registry(tmp_path) -> ModelRegistry:
    """Registry pointed at the REAL manifest but a temp registry file."""
    return ModelRegistry(
        path=tmp_path / "registry.json",
        manifest_paths=DEFAULT_MANIFEST_PATHS,
        predictions_path=tmp_path / "predictions.jsonl",
    )


def entry_kwargs(**kw) -> dict:
    base = dict(
        model_name="xgboost_risk",
        model_version="0.1.0",
        feature_version=REAL_SCHEMA_VERSION,
        training_dataset_version=REAL_DATASET_VERSION,
    )
    base.update(kw)
    return base


def test_real_manifest_exists_and_is_the_default_resolution_target() -> None:
    assert all(p.exists() for p in DEFAULT_MANIFEST_PATHS), ": the dataset manifest is a hard requirement"
    reg = ModelRegistry(path=REPO_ROOT / "models" / "registry.json") if False else None
    # (constructed lazily in other tests; here just assert the default path layout)
    assert (REPO_ROOT / "data" / "synthetic" / "dataset_manifest.json").is_file()


def test_registered_models_persist_in_the_registry_file(registry: ModelRegistry) -> None:
    entry = registry.register_model(**entry_kwargs())
    data = json.loads(registry.path.read_text(encoding="utf-8"))
    assert data["models"][0]["model_name"] == "xgboost_risk"
    assert entry.model_version == "0.1.0"


def test_provenance_hash_is_persisted_with_registered_artifact(registry: ModelRegistry) -> None:
    entry = registry.register_model(**entry_kwargs(), provenance_hash="a" * 64, split_name="train", seed=42)
    reloaded = ModelRegistry(path=registry.path, manifest_paths=DEFAULT_MANIFEST_PATHS,
                             predictions_path=registry.predictions_path)
    assert entry.provenance_hash == "a" * 64
    loaded = reloaded.get_entry(entry.model_name, entry.model_version)
    assert loaded.provenance_hash == "a" * 64
    assert loaded.split_name == "train"
    assert loaded.seed == 42


def test_dataset_version_must_resolve_to_an_existing_manifest(registry: ModelRegistry, tmp_path) -> None:
    with pytest.raises(ModelRegistryError, match="does not resolve"):
        registry.register_model(**entry_kwargs(training_dataset_version="v999.0.0"))
    # a version whose manifest file is absent fails even if well-formed:
    empty = ModelRegistry(
        path=tmp_path / "r2.json",
        manifest_paths=[tmp_path / "missing.json"],
        predictions_path=tmp_path / "p2.jsonl",
    )
    with pytest.raises(ModelRegistryError, match="does not resolve"):
        empty.resolve_training_dataset(REAL_DATASET_VERSION)


def test_feature_version_must_match_the_manifest_schema_version(registry: ModelRegistry) -> None:
    with pytest.raises(ModelRegistryError, match="feature_schema_version"):
        registry.register_model(**entry_kwargs(feature_version="vX-wrong"))


def test_every_logged_prediction_carries_the_five_fr14_fields(registry: ModelRegistry) -> None:
    registry.register_model(**entry_kwargs())
    rec = registry.log_prediction(
        "xgboost_risk",
        "0.1.0",
        {"node_id": "n3", "alert_level": "WARNING", "P_NORMAL": 0.1, "P_WARNING": 0.6, "P_CRITICAL": 0.3},
        now=datetime(2026, 9, 27, 13, 30, tzinfo=timezone.utc),
    )
    for field in FR14_FIELDS:
        assert field in rec, f" field {field} missing"
    assert rec["timestamp"] == "2026-09-27T13:30:00+00:00"
    assert rec["training_dataset_version"] == REAL_DATASET_VERSION
    assert rec["feature_version"] == REAL_SCHEMA_VERSION
    assert rec["model_name"] == "xgboost_risk"


def test_predictions_cannot_be_logged_for_unregistered_models(registry: ModelRegistry) -> None:
    with pytest.raises(ModelRegistryError, match="not registered"):
        registry.log_prediction("ghost_model", "1.0", {"node_id": "n1"})


def test_prediction_log_is_append_only_jsonl(registry: ModelRegistry) -> None:
    registry.register_model(**entry_kwargs())
    registry.log_prediction("xgboost_risk", "0.1.0", {"node_id": "n1"})
    registry.log_prediction("xgboost_risk", "0.1.0", {"node_id": "n2"})
    lines = registry.predictions_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert [json.loads(line)["node_id"] for line in lines] == ["n1", "n2"]
    assert len(registry.read_predictions()) == 2


def test_registry_reloads_entries_from_disk(tmp_path) -> None:
    path = tmp_path / "registry.json"
    reg1 = ModelRegistry(path=path, manifest_paths=DEFAULT_MANIFEST_PATHS, predictions_path=tmp_path / "p.jsonl")
    reg1.register_model(**entry_kwargs(model_version="0.2.0"))
    reg2 = ModelRegistry(path=path, manifest_paths=DEFAULT_MANIFEST_PATHS, predictions_path=tmp_path / "p.jsonl")
    entry = reg2.get_entry("xgboost_risk", "0.2.0")  # survived the reload
    assert entry.training_dataset_version == REAL_DATASET_VERSION
    assert len(reg2.list_models()) == 1


def test_registry_records_registration_timestamp(registry: ModelRegistry) -> None:
    ts = datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc)
    entry = registry.register_model(**entry_kwargs(), now=ts)
    assert entry.registered_at == "2026-09-27T09:00:00+00:00"


def test_blank_names_are_rejected(registry: ModelRegistry) -> None:
    with pytest.raises(ModelRegistryError, match="required"):
        registry.register_model(**entry_kwargs(model_name="  "))
