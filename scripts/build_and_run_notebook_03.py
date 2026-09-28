#!/usr/bin/env python3
"""T-042 — build and execute notebooks/03_feature_engineering.ipynb.

Acceptance (TASKS.md T-042): the notebook executes end-to-end and asserts the
assembled first-iteration feature count falls within 40–70 inclusive, failing
loudly if it does not. The run also emits the feature-store artifacts to
data/features/ (features_v2.parquet + feature_store_report.json).

Idempotent: rebuilds and re-executes the notebook in place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "03_feature_engineering.ipynb"

CELLS = [
    new_markdown_cell(
        """# 03 — Feature Engineering: the Feature Store (PRD §13, T-042)

**Purpose:** assemble the first-iteration feature store from the raw synthetic
node table through the Phase-2 pipeline — §10 windowing (60/10) → feature
Groups A–F → §12 majority labels per window — and **assert the §13 budget of
40–70 engineered features**, failing loudly outside it.

**Design notes carried from the modules:**
- Group B rolling statistics are series-internal (no train/test boundary
  crossing — §23, executable check in `tests/test_group_b.py`);
- Group C uses observable values from the current snapshot; single-node
  snapshots receive gated NaNs. Oracle geometry is refused for model builds.
- Group D features come from §8.3 on-node **summarised** vibration only."""
    ),
    new_code_cell(
        """import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent))

from src.features.build_feature_store import build_feature_store, feature_names
from src.features.windowing import windowing_params

REPO = Path.cwd().parent
nodes_path = REPO / "data" / "synthetic" / "synthetic_nodes.csv"
window, stride = windowing_params()
print(f"windowing: window={window} stride={stride} (§10, config-driven)")"""
    ),
    new_code_cell(
        """usecols = ["event_id", "node_id", "timestamp", "x", "y",
           "tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain",
           "vibration_rms", "vibration_peak", "battery", "RSSI", "SNR", "packet_loss",
           "anomaly_label", "risk_label", "progression_label", "fault_label"]
t0 = time.time()
raw = pd.read_csv(nodes_path, usecols=usecols)
coords = raw.groupby("node_id", sort=False)[["x", "y"]].first().reset_index()
print(f"loaded {len(raw):,} rows, {raw.event_id.nunique():,} events in {time.time()-t0:.1f}s")"""
    ),
    new_markdown_cell(
        """## 1 — Assemble the feature store (Groups A–F over §10 windows)

One node per event in the corpus means no co-temporal neighbours. Group C is
gated on the production corpus (spatial behavior is exercised by mesh tests);
§21.1 neighbor confirmation cannot be validated with this data."""
    ),
    new_code_cell(
        """t0 = time.time()
model, report = build_feature_store(raw, coords, center_mode="detected")
print(f"feature store: {report.n_windows:,} windows in {time.time()-t0:.1f}s")
print(f"features: {len(report.features)}")
for group, feats in report.per_group.items():
    print(f"  {group:16s} {len(feats):2d} features")"""
    ),
    new_markdown_cell(
        "## 2 — The §13 budget assertion (40–70 features, inclusive)"
    ),
    new_code_cell(
        """n = len(report.features)
MIN_BUDGET, MAX_BUDGET = 40, 70
print(f"assembled feature count = {n}; §13 budget = [{MIN_BUDGET}, {MAX_BUDGET}]")
assert MIN_BUDGET <= n <= MAX_BUDGET, (
    f"§13 feature budget violated: {n} features, allowed {MIN_BUDGET}-{MAX_BUDGET}"
)
print("budget assertion PASSED")"""
    ),
    new_markdown_cell(
        """## 3 — Sanity: labels stay separate, features are finite, windows are per-series"""
    ),
    new_code_cell(
        """from src.preprocessing.labels import validate_labels, assert_labels_separate

validate_labels(model)          # §12 vocabularies
assert_labels_separate(model)   # no collapse into one flag

feature_cols = report.features
from src.features.provenance import model_input_allowlist
from src.features.schema_guard import assert_schema_drift
assert_schema_drift(model.columns)
checked_cols = sorted(model_input_allowlist(feature_cols))
finite_ok = model[checked_cols].apply(lambda s: np.isfinite(s.to_numpy(dtype=float)).all()).all()
print(f"all {len(checked_cols)} active allow-listed feature columns finite: {bool(finite_ok)}")
assert finite_ok

per_series = model.groupby(["event_id", "node_id"]).size()
print(f"windows per series: min={per_series.min()} max={per_series.max()} (expect 9 = (144-60)/10+1)")
assert (per_series == 9).all()
print(f"risk distribution in the store:\\n{model['risk_label'].value_counts().to_string()}")"""
    ),
    new_markdown_cell("## 4 — Persist the feature-store artifacts (§33 data/features/)"),
    new_code_cell(
        """out_dir = REPO / "data" / "features"
out_dir.mkdir(parents=True, exist_ok=True)
parquet_path = out_dir / "features_v2.parquet"
model.to_parquet(parquet_path, index=False)

report_json = {
    "feature_schema_version": "v2",
    "dataset_version": "synthetic-v2-label-blind",
    "windowing": {"window_steps": window, "stride": stride},
    "n_windows": int(report.n_windows),
    "n_features": len(report.features),
    "features": report.features,
    "per_group": {k: len(v) for k, v in report.per_group.items()},
    "center_mode": report.center_mode,
    "gated_features": report.gated_features,
    "spatial_gate_reason": "single-node events, no co-temporal neighbours",
    "budget": {"min": MIN_BUDGET, "max": MAX_BUDGET, "ok": True},
    "source": "data/synthetic/synthetic_nodes.csv",
}
(out_dir / "feature_store_report.json").write_text(json.dumps(report_json, indent=2))
print(f"wrote {parquet_path} ({parquet_path.stat().st_size/1e6:.1f} MB)")
print(f"wrote {out_dir / 'feature_store_report.json'}")"""
    ),
    new_markdown_cell("## Verdict"),
    new_code_cell(
        """print("T-042 FEATURE STORE")
print(f"  windows           : {report.n_windows:,}")
print(f"  features assembled: {len(report.features)} (budget 40-70: PASS)")
print(f"  §12 labels        : separate columns, vocabularies validated")
print(f"  artifacts         : data/features/features_v2.parquet + report")
verdict = True
assert verdict"""
    ),
]


def main() -> int:
    notebook = new_notebook(
        cells=CELLS,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)

    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT / "notebooks")}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    manifest_path = REPO_ROOT / "data" / "synthetic" / "dataset_manifest.json"
    report_path = REPO_ROOT / "data" / "features" / "feature_store_report.json"
    if manifest_path.exists() and report_path.exists():
        manifest = json.loads(manifest_path.read_text())
        report = json.loads(report_path.read_text())
        manifest["feature_schema_version"] = report["feature_schema_version"]
        manifest["windows_produced"] = int(report["n_windows"])
        manifest["feature_store"] = {
            "path": "data/features/features_v2.parquet",
            "n_windows": int(report["n_windows"]),
            "n_features": int(report["n_features"]),
            "window_steps": int(report["windowing"]["window_steps"]),
            "stride_steps": int(report["windowing"]["stride"]),
            "center_mode": report["center_mode"],
        }
        manifest["sequence_count_note"] = (
            "10,000 generated events produce 90,000 overlapping windows at window=60/stride=10, "
            "below PRD §15's 100,000-500,000 window band. The 4,000,000 planned sequence "
            "capacity is not achieved data; §10's nine windows per event cap forecasting to horizon 1."
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"executed OK → {NOTEBOOK_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
