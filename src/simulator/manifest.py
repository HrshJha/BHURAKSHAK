"""Dataset manifest writer — PRD §10.1 (T-021), NFR-7, FR-14.

Every generated synthetic dataset ships with ``dataset_manifest.json`` so any
dataset used for training is traceable and reproducible — this is what the
model registry (§30) points to as ``training_dataset_version``.

The manifest is a HARD requirement (§10.1): without it the unseen-parameter-
regime test split (§23) and registry traceability (FR-14) cannot be verified
after the fact. ``write_manifest`` refuses to emit a manifest that is missing
any §10.1 field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import feature_schema_config, physics_config
from src.simulator.scenarios import Scenario

GENERATOR_VERSION = "sim-0.1.0"

# The exact §10.1 field list — validated on write
REQUIRED_FIELDS: list[str] = [
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

FAULT_TYPES = ["BIAS", "STUCK", "DROPOUT", "SPIKE", "DRIFT"]


def build_manifest(
    dataset_version: str,
    random_seed: int,
    counts_per_scenario: dict[str, int],
    split_definition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a §10.1 manifest from the live config + generation facts."""
    cfg = physics_config()
    fsc = feature_schema_config()
    return {
        "dataset_version": dataset_version,
        "generator_version": GENERATOR_VERSION,
        "random_seed": int(random_seed),
        "physics_parameters": dict(cfg["physics"]),
        "noise_parameters": {
            "tilt_noise_std": cfg["noise"]["tilt_noise_std_deg"],
            "displacement_noise_std": cfg["noise"]["displacement_noise_std_mm"],
            "vibration_noise_model": cfg["noise"]["vibration_noise_model"],
        },
        "fault_parameters": {
            "fault_types": list(FAULT_TYPES),
            "injection_rate": cfg["scenarios"]["fault_injection_rate"],
        },
        "scenario_parameters": {
            "scenario_types": [s.value for s in Scenario],
            "counts_per_scenario": dict(counts_per_scenario),
            "taxonomy_rows": 12,
        },
        "source_data_versions": {"sentinel1": None, "dgps": None, "hardware": None},
        "feature_schema_version": str(fsc["feature_schema_version"]),
        "split_definition": split_definition
        or {"type": "synthetic_parameter_holdout", "train_range": {}, "test_range": {}},
    }


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return the list of missing §10.1 fields (empty = valid)."""
    return [f for f in REQUIRED_FIELDS if f not in manifest]


def write_manifest(manifest: dict[str, Any], path: Path | str) -> Path:
    """Write the manifest JSON, refusing incomplete manifests (hard requirement)."""
    missing = validate_manifest(manifest)
    if missing:
        raise ValueError(f"manifest missing required §10.1 fields: {missing}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out
