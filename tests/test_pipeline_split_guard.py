"""Fixall Phase 1.3 — the pipeline must fail with a named producer when the
split file is missing, not with a bare FileNotFoundError."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _tiny_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["E_stable_ground_0000_0000"],
            "node_id": ["V0000"],
            "timestamp": [0.0],
            "x": [0.0],
            "y": [0.0],
            "tilt_x": [0.0],
            "tilt_y": [0.0],
            "tilt_magnitude": [0.0],
            "displacement": [0.0],
            "strain": [0.0],
            "vibration_rms": [0.0],
            "vibration_peak": [0.0],
            "battery": [4.0],
            "RSSI": [-95.0],
            "SNR": [0.0],
            "packet_loss": [0.0],
            "anomaly_label": [0],
            "risk_label": ["NORMAL"],
            "progression_label": ["STABLE"],
            "fault_label": ["NONE"],
        }
    )


def test_pipeline_names_the_split_producer_when_csv_missing() -> None:
    import src.pipeline as pl

    tiny = _tiny_raw()
    coords = tiny[["node_id", "x", "y"]].drop_duplicates()
    real = Path("data/features/split_assignment.csv")
    backup = Path("data/features/split_assignment.csv.fixall-bak")
    existed = real.exists()
    assert existed, "the committed split file must exist for this test to be meaningful"
    real.rename(backup)
    try:
        with pytest.raises(pl.PipelineError, match="make_split_assignment"):
            pl.run_pipeline(tiny, coords, verbose=False)
    finally:
        backup.rename(real)
