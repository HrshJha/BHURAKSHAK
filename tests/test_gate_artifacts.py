"""Gate-artifact regression tests — the three §10 deliverables stay valid.

Lightweight on purpose: reads the events CSV and manifest fully, and the
nodes CSV by column projection only (the raw file is ~220 MB).

Skipped automatically when the gate artifacts have not been generated yet
(run `python scripts/generate_synthetic_nodes.py` first).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.simulator.manifest import validate_manifest

SYNTH_DIR = Path("data/synthetic")

EVENT_COLUMNS = [
    "id", "start_time", "end_time", "type", "severity",
    "center", "max_deformation", "rate",
]

pytestmark = pytest.mark.skipif(
    not (SYNTH_DIR / "synthetic_nodes.csv").is_file(),
    reason="gate artifacts not generated yet (scripts/generate_synthetic_nodes.py)",
)


@pytest.fixture(scope="module")
def events_df() -> pd.DataFrame:
    return pd.read_csv(SYNTH_DIR / "synthetic_events.csv")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((SYNTH_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))


def test_nodes_csv_exists_with_min_columns() -> None:
    header = pd.read_csv(SYNTH_DIR / "synthetic_nodes.csv", nrows=0)
    required = {
        "event_id", "timestamp", "node_id", "x", "y",
        "tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain",
        "vibration_rms", "vibration_peak", "battery", "RSSI", "SNR", "packet_loss",
        "anomaly_label", "fault_label", "progression_label", "risk_label",
    }
    missing = required - set(header.columns)
    assert not missing, f"nodes CSV missing §11-aligned columns: {sorted(missing)}"


def test_events_csv_has_exact_section10_columns(events_df: pd.DataFrame) -> None:
    assert list(events_df.columns) == EVENT_COLUMNS
    assert len(events_df) >= 10_000, "§10: ≥10,000 generated sequences"


def test_referential_integrity_nodes_to_events(events_df: pd.DataFrame) -> None:
    nodes_ids = pd.read_csv(SYNTH_DIR / "synthetic_nodes.csv", usecols=["event_id"])
    unresolved = set(nodes_ids["event_id"].unique()) - set(events_df["id"].unique())
    assert not unresolved, f"{len(unresolved)} event_ids unresolved"


def test_manifest_valid_and_consistent(manifest: dict, events_df: pd.DataFrame) -> None:
    assert validate_manifest(manifest) == [], "manifest must satisfy §10.1"
    rc = manifest["row_counts"]
    assert rc["synthetic_nodes_rows"] == rc["synthetic_events_rows"] * 144, (
        "each sequence contributes exactly one event and 144 timestep rows"
    )
    assert rc["sequences_generated"] >= 10_000


def test_gate_fault_and_scenario_diversity(events_df: pd.DataFrame) -> None:
    assert set(events_df["type"].unique()) >= {
        "stable_ground", "slow_subsidence", "accelerating_subsidence",
        "rapid_subsidence", "irregular_subsidence", "multiple_zones",
        "vibration_only", "packet_loss", "communication_failure",
    }, "all major §10 scenario families present"
