"""Check feature columns against their declared schema."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import feature_schema_config

__all__ = [
    "SchemaDriftError",
    "KEY_COLUMNS",
    "EXEMPT_COLUMNS",
    "LABEL_SUFFIXES",
    "declared_feature_names",
    "assert_schema_drift",
    "assert_manifest_schema_version",
]


class SchemaDriftError(ValueError):
    """Raised when emitted columns and the declared schema disagree."""


#: Frame columns that are keys/labels, never features.
KEY_COLUMNS = ("event_id", "node_id", "window_index", "window_timestamp", "timestamp", "split", "center_mode", "spatial_gate_reason")
LABEL_SUFFIXES = ("_label",)
EXEMPT_COLUMNS = ("physics_residual_z", "physics_residual_tilt_x", "physics_residual_tilt_y",
                  "neighbour_confirmations", "anomaly_score", "if_flag")


def declared_feature_names(schema: dict | None = None) -> set[str]:
    """All feature names declared across every group of the schema."""
    schema = schema or feature_schema_config()
    names: set[str] = set()
    for group in schema["feature_groups"].values():
        names |= set(group)
    return names


def _is_exempt(col: str) -> bool:
    return col in KEY_COLUMNS or col in EXEMPT_COLUMNS or any(col.endswith(s) for s in LABEL_SUFFIXES)


def assert_schema_drift(columns: pd.Index | list[str], schema: dict | None = None) -> None:
    """Fail on undeclared emitted columns and on ungated absent declared columns.

 ``schema`` defaults to the live configs/feature_schema_v1.yaml. The
 allowed-absent set is the union of every group's names listed under
 ``gates:`` with ``gated: true`` — a gate without a reason fails.
 """
    schema = schema or feature_schema_config()
    declared = declared_feature_names(schema)
    gates = schema.get("gates", {})
    gated_names: set[str] = set()
    for group, meta in gates.items():
        if not meta.get("gated"):
            continue
        if not str(meta.get("gate_reason", "")).strip():
            raise SchemaDriftError(f"schema gate for {group} has no gate_reason")
        gated_names |= set(schema["feature_groups"].get(group, []))

    emitted = {str(c) for c in columns if not _is_exempt(str(c))}

    undeclared = sorted(emitted - declared)
    if undeclared:
        raise SchemaDriftError(
            f"feature-schema drift: {len(undeclared)} emitted column(s) not declared in "
            f"feature_schema_v{schema.get('feature_schema_version', '?')}: {undeclared[:10]}"
        )

    # declared-but-absent is a failure only when the group is NOT gated
    absent_ungated = sorted(declared - gated_names - emitted)
    if absent_ungated:
        raise SchemaDriftError(
            f"feature-schema drift: {len(absent_ungated)} declared column(s) absent without a gate: "
            f"{absent_ungated[:10]}"
        )


def assert_manifest_schema_version(manifest_path: str | Path) -> str:
    """The manifest's feature_schema_version must equal the live schema."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_version = str(manifest.get("feature_schema_version", ""))
    live_version = str(feature_schema_config()["feature_schema_version"])
    if manifest_version != live_version:
        raise SchemaDriftError(
            f"dataset manifest claims feature_schema_version {manifest_version!r} "
            f"but the live schema is {live_version!r} — regenerate or re-version the dataset"
        )
    return live_version
