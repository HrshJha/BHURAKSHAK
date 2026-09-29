"""schema drift guard tests (both directions + manifest)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import feature_schema_config
from src.features.build_feature_store import build_feature_store
from src.features.schema_guard import (
    SchemaDriftError,
    assert_manifest_schema_version,
    assert_schema_drift,
)

REPO = Path(__file__).resolve().parent.parent


def _current_store() -> pd.DataFrame:
    n = 144
    interval = 10.0 / 60.0
    frames = []
    coords = []
    for i in range(3):
        node = f"V{i:04d}"
        coords.append({"node_id": node, "x": 25.0 * i, "y": 0.0})
        t = interval * pd.RangeIndex(n).to_numpy(dtype=float)
        d = i * 0.1 + pd.RangeIndex(n).to_numpy(dtype=float) * (i + 1) * 0.01
        frames.append(pd.DataFrame({
            "event_id": "E1", "node_id": node, "timestamp": t,
            "tilt_x": d * 1e-4, "tilt_y": d * -1e-4, "tilt_magnitude": abs(d) * 1e-4,
            "displacement": d, "strain": d * 1e-6, "vibration_rms": abs(d) + 0.02,
            "vibration_peak": abs(d) + 0.05, "battery": 4.0 - d * 1e-3,
            "RSSI": -95.0, "SNR": 4.0, "packet_loss": 0.0,
            "anomaly_label": int(i == 2), "risk_label": "WARNING" if i == 2 else "NORMAL",
            "progression_label": "SLOW" if i == 2 else "STABLE", "fault_label": "NONE",
        }))
    store, _ = build_feature_store(pd.concat(frames, ignore_index=True), pd.DataFrame(coords))
    return store


def test_real_store_passes_the_drift_guard() -> None:
    store = _current_store()
    assert_schema_drift(store.columns)  # must not raise


def test_undeclared_emitted_column_fails() -> None:
    store = _current_store()
    bad = store.assign(totally_new_feature=1.0)
    with pytest.raises(SchemaDriftError, match="totally_new_feature"):
        assert_schema_drift(bad.columns)


def test_ungated_absent_declared_column_fails() -> None:
    schema = feature_schema_config()
    # drop one ungated (A_physical) column from the frame
    name = schema["feature_groups"]["A_physical"][0]
    store = _current_store()
    assert name in store.columns
    with pytest.raises(SchemaDriftError, match=name):
        assert_schema_drift(store.drop(columns=[name]).columns)


def test_gated_absent_group_passes() -> None:
    schema = feature_schema_config()
    store = _current_store()
    # Group C values can exist in a multi-node fixture, but the corpus-level
    # gate allows them to be absent when no co-temporal graph is available.
    columns_without_c = [c for c in store.columns if c not in schema["feature_groups"]["C_spatial"]]
    assert_schema_drift(columns_without_c)
    trimmed = {**schema, "gates": {k: v for k, v in schema["gates"].items() if k != "C_spatial"}}
    with pytest.raises(SchemaDriftError, match="neighbor_mean"):
        assert_schema_drift(columns_without_c, schema=trimmed)


def test_gate_without_reason_fails() -> None:
    schema = feature_schema_config()
    broken = {**schema, "gates": {**schema["gates"], "G_dgps": {"gated": True, "gate_reason": ""}}}
    with pytest.raises(SchemaDriftError, match="gate_reason"):
        assert_schema_drift(_current_store().columns, schema=broken)


def test_labels_and_keys_are_exempt() -> None:
    store = _current_store()
    assert "risk_label" in store.columns and "event_id" in store.columns
    assert_schema_drift(store.columns)  # labels/keys never count as drift


def test_manifest_schema_version_matches_live(tmp_path: Path) -> None:
    manifest = json.loads((REPO / "data" / "synthetic" / "dataset_manifest.json").read_text())
    manifest["feature_schema_version"] = feature_schema_config()["feature_schema_version"]
    p = tmp_path / "dataset_manifest.json"
    p.write_text(json.dumps(manifest))
    assert_manifest_schema_version(p) == "v2"


def test_manifest_schema_mismatch_fails(tmp_path: Path) -> None:
    manifest = json.loads((REPO / "data" / "synthetic" / "dataset_manifest.json").read_text())
    manifest["feature_schema_version"] = "v999"
    p = tmp_path / "dataset_manifest.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(SchemaDriftError, match="v999"):
        assert_manifest_schema_version(p)
