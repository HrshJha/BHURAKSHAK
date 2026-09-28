"""Fixall Phase 2 — artifact integrity, FR-14 traceability, credential hygiene.

Covers: versioned save/load with sha256 sidecars (tamper refused),
preprocessing declared-and-applied per bundle, registry-backed FR-14
prediction records, and the env-only CDSE credential path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.risk.artifacts import ArtifactError, load_model_artifact, save_model_artifact

REPO = Path(__file__).resolve().parent.parent
REAL_DATASET_VERSION = json.loads((REPO / "data" / "synthetic" / "dataset_manifest.json").read_text())["dataset_version"]


@pytest.fixture()
def tiny_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"displacement": rng.normal(10, 2, 64), "velocity": rng.normal(0.1, 0.01, 64)})


def _save(tmp_path: Path, **overrides):
    kwargs = dict(
        path=tmp_path / "m.joblib",
        model={"stub": "model"},
        features=["displacement", "velocity"],
        model_name="test_model",
        model_version="1.0.0",
        training_dataset_version=REAL_DATASET_VERSION,
        split_name="regime_holdout",
        seed=42,
    )
    kwargs.update(overrides)
    return save_model_artifact(**kwargs)


def test_save_writes_sidecar_and_registers(tmp_path: Path) -> None:
    fr14 = _save(tmp_path)
    assert (tmp_path / "m.joblib.sha256").is_file()
    assert set(fr14) == {"model_name", "model_version", "feature_version",
                         "training_dataset_version", "timestamp", "split_name", "seed"}
    from src.risk.model_registry import ModelRegistry

    entries = {(e.model_name, e.model_version) for e in ModelRegistry().list_models()}
    assert ("test_model", "1.0.0") in entries


def test_load_round_trip_and_transform_applies_declared_preprocessing(tmp_path: Path, tiny_frame: pd.DataFrame) -> None:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(tiny_frame[["displacement", "velocity"]])
    _save(tmp_path, scaler=scaler, preprocessing="standardise")
    art = load_model_artifact(tmp_path / "m.joblib")
    assert art["preprocessing"] == "standardise"
    out = art["transform"](tiny_frame)
    np.testing.assert_allclose(out["displacement"].to_numpy(), scaler.transform(tiny_frame[["displacement", "velocity"]])[:, 0])


def test_load_refuses_tampered_artifact(tmp_path: Path) -> None:
    _save(tmp_path)
    path = tmp_path / "m.joblib"
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # flip one byte
    path.write_bytes(bytes(raw))
    with pytest.raises(ArtifactError, match="integrity failure"):
        load_model_artifact(path)


def test_load_refuses_missing_sidecar(tmp_path: Path) -> None:
    _save(tmp_path)
    (tmp_path / "m.joblib.sha256").unlink()
    with pytest.raises(ArtifactError, match="sidecar"):
        load_model_artifact(tmp_path / "m.joblib")


def test_load_refuses_schema_mismatch(tmp_path: Path) -> None:
    _save(tmp_path)
    with pytest.raises(ArtifactError, match="schema"):
        load_model_artifact(tmp_path / "m.joblib", expected_schema_version="v0")


def test_preprocessing_declaration_must_match_payload(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="declare"):
        _save(tmp_path, preprocessing="none", scaler=object())
    with pytest.raises(ArtifactError, match="standardise"):
        _save(tmp_path, preprocessing="standardise", scaler=None)


def test_pipeline_prediction_log_carries_the_five_fr14_fields() -> None:
    """One real prediction path (the pipeline run's log) must carry FR-14 fields."""
    from src.risk.model_registry import ModelRegistry

    log_path = REPO / "models" / "predictions.jsonl"
    if not log_path.is_file():
        pytest.skip("predictions.jsonl not produced yet — run scripts/run_pipeline.py")
    first = json.loads(log_path.read_text().splitlines()[0])
    for field in ("model_name", "model_version", "feature_version", "training_dataset_version", "timestamp"):
        assert field in first, f"FR-14 field {field} missing from the prediction record"
    registry = ModelRegistry()
    entry = registry.get_entry(first["model_name"], first["model_version"])
    assert first["training_dataset_version"] == entry.training_dataset_version
    assert first["feature_version"] == entry.feature_version


def test_downloader_takes_no_password_argv() -> None:
    """The CDSE password must be env-only (not an argv flag)."""
    import ast

    tree = ast.parse((REPO / "scripts" / "download_sentinel1.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            arg = node.args[0].value if node.args else ""
            assert "--password" not in str(arg), "download_sentinel1.py still accepts --password on argv"
