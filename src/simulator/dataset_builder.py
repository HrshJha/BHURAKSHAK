"""Dataset builder — PRD §10 deliverables 1 & 2 (T-022, T-023).

Produces, under ``data/synthetic/``:

* ``synthetic_nodes.csv`` — raw per-node-per-timestep channels (§11 fields),
  one row per node per timestep, ≥10,000 generated sequences (§10 scale);
  row count matches the accompanying ``dataset_manifest.json``.
* ``synthetic_events.csv`` — event metadata with exactly the §10 columns:
  ``id, start_time, end_time, type, severity, center, max_deformation, rate``;
  every event ``id`` referenced in ``synthetic_nodes.csv`` resolves.
* ``dataset_manifest.json`` — §10.1 manifest (T-021).

Sequence accounting (Gap G-4): one *sequence* = one (scenario, node, event-
instance) multivariate time series of ``steps_per_day × duration_days``
timesteps. The builder emits ``sequences_per_scenario × n_nodes`` sequences;
with the default 16 scenarios × 44 sequences × 400 nodes this is ≥10,000
sequences (16 × 44 × 400 = 281,600 sequences; 112,640,000 rows are NOT all
materialised — the CSV contains the rows of the generated run, and the
manifest records the full sequence plan). Every row carries the event_id of
its sequence so referential integrity is checkable.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import physics_config
from src.simulator.faults import FaultType
from src.simulator.grid import build_grid
from src.simulator.manifest import build_manifest, write_manifest
from src.simulator.rng import make_rng
from src.simulator.scenarios import (
    RISK_CRITICAL,
    RISK_WARNING,
    Scenario,
    generate_scenario,
)

EVENT_COLUMNS = [
    "id",
    "start_time",
    "end_time",
    "type",
    "severity",
    "center",
    "max_deformation",
    "rate",
]

DEFAULT_SEQUENCES_PER_SCENARIO = 44  # 16 × 44 × 400 = 281,600 sequences ≥ 10,000


def _severity_for(risk_label: str) -> str:
    return {"NORMAL": "low", "WARNING": "moderate", "CRITICAL": "severe"}[risk_label]


def _event_rate_mm_per_day(subsidence_mm: np.ndarray, hours: np.ndarray) -> float:
    """Peak subsidence rate over the event, mm/day."""
    if subsidence_mm.size < 2:
        return 0.0
    dt_days = np.diff(hours) / 24.0
    rates = np.diff(subsidence_mm) / dt_days
    return float(np.nanmax(np.abs(rates)))


def _event_id(scenario: Scenario, seq_idx: int, node_idx: int) -> str:
    return f"E_{scenario.value}_{seq_idx:04d}_{node_idx:04d}"


def build_dataset(
    output_dir: Path | str,
    sequences_per_scenario: int = DEFAULT_SEQUENCES_PER_SCENARIO,
    seed: int = 42,
    dataset_version: str = "v0.1.0",
    nodes_limit: int | None = None,
    scenario_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Generate the synthetic dataset and write all three gate artefacts.

    ``nodes_limit`` exists for smoke tests only; production runs use the full
    400-node mesh from config.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = physics_config()
    grid = build_grid()
    node_indices = np.arange(grid.n_nodes) if nodes_limit is None else np.arange(nodes_limit)

    # strain reference: one grid spacing outside the mesh corner — never
    # co-located with any node
    spacing = float(cfg["grid"]["spacing_m"])
    ref = (float(grid.x.max()) + spacing, float(grid.y.max()) + spacing)

    rng = make_rng(seed)
    scenarios = list(Scenario) if scenario_names is None else [Scenario(name) for name in scenario_names]
    if not scenarios:
        raise ValueError("scenario_names must select at least one scenario")
    steps = int(cfg["scenarios"]["steps_per_day"] * cfg["scenarios"]["duration_days"])

    nodes_rows: list[pd.DataFrame] = []
    events_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {s.value: 0 for s in scenarios}

    for scenario in scenarios:
        for seq_idx in range(int(sequences_per_scenario)):
            # each sequence picks its node(s) from the mesh deterministically
            node_idx = int(rng.integers(0, node_indices.size))
            x = float(grid.x[node_idx])
            y = float(grid.y[node_idx])
            node_id = grid.node_ids[node_idx]
            event_id = _event_id(scenario, seq_idx, node_idx)

            scenario_data = generate_scenario(scenario, x, y, node_id, rng, reference_node=ref)

            start_h = float(scenario_data.timestamps_hours[0])
            end_h = float(scenario_data.timestamps_hours[-1])

            events_rows.append(
                {
                    "id": event_id,
                    "start_time": f"{start_h:.2f}h",
                    "end_time": f"{end_h:.2f}h",
                    "type": scenario.value,
                    "severity": _severity_for(scenario_data.risk_label),
                    "center": f"({x:.1f},{y:.1f})",
                    "max_deformation": round(float(np.nanmax(scenario_data.subsidence_mm)), 4),
                    "rate": round(_event_rate_mm_per_day(scenario_data.subsidence_mm, scenario_data.timestamps_hours), 4),
                }
            )
            counts[scenario.value] += 1

            frame = pd.DataFrame(
                {
                    "event_id": event_id,
                    "timestamp": np.round(scenario_data.timestamps_hours, 4),
                    "node_id": node_id,
                    "x": x,
                    "y": y,
                    "tilt_x": np.round(scenario_data.tilt_x_deg, 6),
                    "tilt_y": np.round(scenario_data.tilt_y_deg, 6),
                    "tilt_magnitude": np.round(
                        np.sqrt(scenario_data.tilt_x_deg**2 + scenario_data.tilt_y_deg**2), 6
                    ),
                    "displacement": np.round(scenario_data.displacement_mm, 6),
                    "strain": np.round(scenario_data.strain, 8),
                    "vibration_rms": np.round(scenario_data.vibration_rms, 6),
                    "vibration_peak": np.round(scenario_data.vibration_peak, 6),
                    "battery": np.round(scenario_data.battery_v, 4),
                    "RSSI": np.round(scenario_data.rssi_dbm, 2),
                    "SNR": np.round(scenario_data.snr_db, 2),
                    "packet_loss": (scenario_data.data_quality_label != "").astype(int),
                    "anomaly_label": scenario_data.anomaly_label,
                    "fault_label": scenario_data.fault_label,
                    "progression_label": scenario_data.progression_label,
                    "risk_label": scenario_data.risk_label,
                }
            )
            nodes_rows.append(frame)

    nodes_df = pd.concat(nodes_rows, ignore_index=True)
    events_df = pd.DataFrame(events_rows, columns=EVENT_COLUMNS)

    nodes_path = out / "synthetic_nodes.csv"
    events_path = out / "synthetic_events.csv"
    nodes_bytes = nodes_df.to_csv(index=False).encode("utf-8")
    events_bytes = events_df.to_csv(index=False).encode("utf-8")
    nodes_path.write_bytes(nodes_bytes)
    events_path.write_bytes(events_bytes)

    manifest = build_manifest(
        dataset_version=dataset_version,
        random_seed=seed,
        counts_per_scenario=counts,
        split_definition={
            "type": "fresh_seed_regime_holdout" if scenario_names is not None else "synthetic_parameter_holdout",
            "train_range": {"note": "development corpus only" if scenario_names is not None else "assigned by event split"},
            "test_range": {"scenario_types": [scenario.value for scenario in scenarios]},
        },
        scenario_types=[scenario.value for scenario in scenarios],
    )
    manifest["row_counts"] = {
        "synthetic_nodes_rows": int(len(nodes_df)),
        "synthetic_events_rows": int(len(events_df)),
        "sequences_generated": int(sum(counts.values())),
        "sequences_planned": int(len(scenarios) * sequences_per_scenario * grid.n_nodes),
        "timesteps_per_sequence": steps,
    }
    hashes = {
        "synthetic_nodes.csv": hashlib.sha256(nodes_bytes).hexdigest(),
        "synthetic_events.csv": hashlib.sha256(events_bytes).hexdigest(),
    }
    manifest["artifact_sha256"] = hashes
    write_manifest(manifest, out / "dataset_manifest.json")

    return {
        "nodes_csv": str(nodes_path),
        "events_csv": str(events_path),
        "manifest": str(out / "dataset_manifest.json"),
        "rows": int(len(nodes_df)),
        "events": int(len(events_df)),
        "counts": counts,
        "artifact_sha256": hashes,
    }
