"""T-025 acceptance tests — §11 dataset schema module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.schema import (
    NODE_METADATA_FIELDS,
    RAW_NODE_LABEL_FIELDS,
    RAW_NODE_TABLE_FIELDS,
    SchemaError,
    dataset_schema_fields,
    schema_violations,
    validate_node_metadata,
    validate_raw_node_table,
    validate_schema,
)


def _df11() -> pd.DataFrame:
    """A minimal dataframe with exactly the §11 columns."""
    n = 3
    data: dict[str, object] = {
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "node_id": ["V0001", "V0002", "V0003"],
        "x": np.zeros(n),
        "y": np.zeros(n),
        "tilt_x": np.zeros(n),
        "tilt_y": np.zeros(n),
        "tilt_magnitude": np.zeros(n),
        "displacement": np.zeros(n),
        "strain": np.zeros(n),
        "tilt_velocity": np.zeros(n),
        "tilt_acceleration": np.zeros(n),
        "displacement_velocity": np.zeros(n),
        "displacement_acceleration": np.zeros(n),
        "vibration_rms": np.zeros(n),
        "vibration_peak": np.zeros(n),
        "neighbor_mean": np.zeros(n),
        "neighbor_std": np.zeros(n),
        "neighbor_anomaly_fraction": np.zeros(n),
        "spatial_coherence": np.zeros(n),
        "battery": np.full(n, 4.0),
        "RSSI": np.full(n, -90.0),
        "SNR": np.full(n, 5.0),
        "packet_loss": np.zeros(n),
        "DGPS_displacement": np.zeros(n),
        "DGPS_velocity": np.zeros(n),
        "InSAR_displacement": np.zeros(n),
        "InSAR_velocity": np.zeros(n),
        "InSAR_coherence": np.zeros(n),
        "physics_displacement": np.zeros(n),
        "physics_residual": np.zeros(n),
        "anomaly_score": np.zeros(n),
        "progression_label": ["STABLE"] * n,
        "risk_label": ["NORMAL"] * n,
    }
    return pd.DataFrame(data)


def test_schema_fields_are_the_exact_prd_list() -> None:
    fields = dataset_schema_fields()
    assert len(fields) == 33, "§11 enumerates exactly 33 fields"
    # spot-check first/last per the PRD text block
    assert fields[0] == "timestamp"
    assert fields[-1] == "risk_label"
    assert "anomaly_score" in fields and "physics_residual" in fields


def test_valid_schema_dataframe_passes() -> None:
    validate_schema(_df11())


def test_missing_column_rejected() -> None:
    df = _df11().drop(columns=["physics_residual"])
    with pytest.raises(SchemaError, match="physics_residual"):
        validate_schema(df)


def test_unknown_column_rejected() -> None:
    df = _df11()
    df = df.assign(hacker_field=1.0)
    with pytest.raises(SchemaError, match="hacker_field"):
        validate_schema(df)


def test_missing_and_unknown_reported_together() -> None:
    df = _df11().drop(columns=["anomaly_score"]).assign(extra=1.0)
    with pytest.raises(SchemaError) as err:
        validate_schema(df)
    msg = str(err.value)
    assert "anomaly_score" in msg and "extra" in msg


def test_relaxed_modes_allow_subsets() -> None:
    df = _df11()[["timestamp", "node_id"]]
    report = schema_violations(df, require_all=False, forbid_unknown=False)
    assert report.ok


def test_dtype_contract_rejects_non_string_node_id() -> None:
    df = _df11()
    df["node_id"] = df.index.astype(float) + 1.0  # numeric instead of string id
    with pytest.raises(SchemaError, match="node_id"):
        validate_schema(df)


def test_dtype_contract_rejects_non_string_risk_label() -> None:
    df = _df11()
    df["risk_label"] = 0  # collapsed binary flag — forbidden by §12
    with pytest.raises(SchemaError, match="risk_label"):
        validate_schema(df)


def test_datetime_timestamp_accepted_numeric_rejected_by_contract() -> None:
    df = _df11()
    df["timestamp"] = "not-a-time"
    report = schema_violations(df)
    assert any("timestamp" in p for p in report.dtype_problems)


def test_raw_node_table_layout_accepted() -> None:
    n = 2
    df = pd.DataFrame(
        {
            "event_id": ["E1", "E1"],
            "timestamp": [0.0, 0.1667],
            "node_id": ["V0001", "V0001"],
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "tilt_x": [0.0, 0.0],
            "tilt_y": [0.0, 0.0],
            "tilt_magnitude": [0.0, 0.0],
            "displacement": [0.0, 0.0],
            "strain": [0.0, 0.0],
            "vibration_rms": [0.0, 0.0],
            "vibration_peak": [0.0, 0.0],
            "battery": [4.0, 4.0],
            "RSSI": [-90.0, -90.0],
            "SNR": [5.0, 5.0],
            "packet_loss": [0, 0],
            "anomaly_label": [0, 0],
            "fault_label": ["NONE", "NONE"],
            "progression_label": ["STABLE", "STABLE"],
            "risk_label": ["NORMAL", "NORMAL"],
        }
    )
    validate_raw_node_table(df)


def test_raw_node_table_rejects_missing_label_column() -> None:
    n = 2
    df = pd.DataFrame(
        {
            "event_id": ["E1", "E1"],
            "timestamp": [0.0, 0.1667],
            "node_id": ["V0001", "V0001"],
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "tilt_x": [0.0, 0.0],
            "tilt_y": [0.0, 0.0],
            "tilt_magnitude": [0.0, 0.0],
            "displacement": [0.0, 0.0],
            "strain": [0.0, 0.0],
            "vibration_rms": [0.0, 0.0],
            "vibration_peak": [0.0, 0.0],
            "battery": [4.0, 4.0],
            "RSSI": [-90.0, -90.0],
            "SNR": [5.0, 5.0],
            "packet_loss": [0, 0],
            "anomaly_label": [0, 0],
            "fault_label": ["NONE", "NONE"],
            "progression_label": ["STABLE", "STABLE"],
        }
    )
    with pytest.raises(SchemaError, match="risk_label"):
        validate_raw_node_table(df)


def test_raw_node_fields_superset_of_s11_labels() -> None:
    assert set(RAW_NODE_LABEL_FIELDS) <= set(RAW_NODE_TABLE_FIELDS)


def test_node_metadata_requires_calibration_fields() -> None:
    meta = pd.DataFrame(
        {
            "node_id": ["V0001", "V0002"],
            "calibration_date": ["2026-01-01", "2026-01-02"],
            "tilt_x_offset": [0.0, 0.1],
            "drift_history": [[], [0.01]],
        }
    )
    validate_node_metadata(meta)


def test_node_metadata_rejects_missing_calibration_date() -> None:
    meta = pd.DataFrame({"node_id": ["V0001"], "tilt_x_offset": [0.0]})
    with pytest.raises(SchemaError, match="calibration_date"):
        validate_node_metadata(meta)


def test_node_metadata_rejects_unknown_columns() -> None:
    meta = pd.DataFrame({"node_id": ["V0001"], "calibration_date": ["2026-01-01"], "bogus": [1]})
    with pytest.raises(SchemaError, match="bogus"):
        validate_node_metadata(meta)


def test_node_metadata_fields_documented() -> None:
    # §8.4: per-node offsets, calibration date, drift history must all exist
    for field in ("tilt_x_offset", "displacement_offset", "calibration_date", "drift_history"):
        assert field in NODE_METADATA_FIELDS
