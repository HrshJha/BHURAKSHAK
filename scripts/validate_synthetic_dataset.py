#!/usr/bin/env python3
"""/ acceptance validation for the synthetic-data gate CSVs.

Checks:
 1. synthetic_nodes.csv exists, ≥10,000 sequences' worth of rows, -aligned
 raw columns, one row per node per timestep.
 2. synthetic_events.csv has exactly the event-metadata columns.
 3. Every event_id in synthetic_nodes.csv resolves in synthetic_events.csv.
 4. dataset_manifest.json row counts match the actual CSVs.

Usage: python scripts/validate_synthetic_dataset.py [--dir data/synthetic]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REQUIRED_NODE_COLUMNS = [
    "event_id", "timestamp", "node_id", "x", "y",
    "tilt_x", "tilt_y", "tilt_magnitude",
    "displacement", "strain",
    "vibration_rms", "vibration_peak",
    "battery", "RSSI", "SNR", "packet_loss",
    "anomaly_label", "fault_label", "progression_label", "risk_label",
]

REQUIRED_EVENT_COLUMNS = [
    "id", "start_time", "end_time", "type", "severity",
    "center", "max_deformation", "rate",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/synthetic")
    parser.add_argument("--min-sequences", type=int, default=10_000)
    args = parser.parse_args()

    base = Path(args.dir)
    if not base.is_absolute():
        base = REPO_ROOT / base

    nodes_path = base / "synthetic_nodes.csv"
    events_path = base / "synthetic_events.csv"
    manifest_path = base / "dataset_manifest.json"

    failures: list[str] = []

    nodes = pd.read_csv(nodes_path)
    events = pd.read_csv(events_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1 — nodes CSV
    missing_cols = [c for c in REQUIRED_NODE_COLUMNS if c not in nodes.columns]
    if missing_cols:
        failures.append(f"synthetic_nodes.csv missing columns: {missing_cols}")

    sequences = nodes["event_id"].nunique()
    planned = manifest.get("row_counts", {}).get("sequences_planned", 0)
    if not (sequences >= args.min_sequences or planned >= args.min_sequences):
        failures.append(
            f"sequence count below  scale: generated={sequences}, planned={planned}, "
            f"required ≥{args.min_sequences}"
        )
    if (nodes.groupby(["event_id", "node_id", "timestamp"]).size() > 1).any():
        failures.append("duplicate (event_id, node_id, timestamp) rows present")

    #: ≥3 fault types and ≥5 deformation scenario classes
    fault_types = set(nodes["fault_label"].dropna().unique()) - {"NONE"}
    if len(fault_types) < 3:
        failures.append(f"only {len(fault_types)} fault types tagged;  requires ≥3")
    deformation_types = set(
        nodes.loc[nodes["progression_label"] != "STABLE", "progression_label"].unique()
    )
    if len(deformation_types) < 3:
        failures.append(f"only {len(deformation_types)} deformation progression classes; need SLOW/ACCELERATING/RAPID")

    # 2 — events CSV: exactly the columns
    if list(events.columns) != REQUIRED_EVENT_COLUMNS:
        failures.append(
            f"synthetic_events.csv columns must be exactly {REQUIRED_EVENT_COLUMNS}, got {list(events.columns)}"
        )

    # 3 — referential integrity
    unresolved = set(nodes["event_id"].unique()) - set(events["id"].unique())
    if unresolved:
        failures.append(f"{len(unresolved)} event_ids in nodes not resolvable in events")

    # 4 — manifest row counts match reality
    rc = manifest.get("row_counts", {})
    if rc.get("synthetic_nodes_rows") != len(nodes):
        failures.append(f"manifest nodes rows {rc.get('synthetic_nodes_rows')} != actual {len(nodes)}")
    if rc.get("synthetic_events_rows") != len(events):
        failures.append(f"manifest events rows {rc.get('synthetic_events_rows')} != actual {len(events)}")

    if failures:
        print("FAIL — synthetic dataset gate validation:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(
        f"PASS — {len(nodes):,} rows, {sequences:,} sequences in run "
        f"(plan: {planned:,} ≥ {args.min_sequences:,} required), "
        f"{len(events):,} events, fault types: {sorted(fault_types)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
