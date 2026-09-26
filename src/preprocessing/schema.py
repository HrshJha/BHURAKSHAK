"""§11 dataset schema module — PRD §11 + §8.4 fold-in (T-025).

The PRD defines one canonical dataset schema (§11). This module is the single
authority for it:

- ``dataset_schema_fields()`` returns the exact §11 field list, read from
  ``configs/feature_schema_v1.yaml`` (config-driven, NFR-6/NFR-7) — never
  re-typed in code.
- ``validate_schema`` accepts a dataframe only when its columns are exactly
  the §11 fields (modulo the ``require_all``/``forbid_unknown`` switches) and
  checks the coarse dtypes that downstream stages rely on.
- ``validate_raw_node_table`` validates the raw per-timestep node table the
  preprocessing pipeline consumes (the synthetic_nodes.csv layout: raw §8/§9
  channels + the four §12 label columns).
- §8.4 calibration output fields (per-node offsets, calibration date, drift
  history) are folded into node metadata here — they must exist in the
  node-metadata table even though the physical calibration protocol itself is
  out of workstream scope (§8.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import feature_schema_config

__all__ = [
    "SchemaError",
    "SchemaReport",
    "dataset_schema_fields",
    "validate_schema",
    "schema_violations",
    "RAW_NODE_TABLE_FIELDS",
    "RAW_NODE_LABEL_FIELDS",
    "validate_raw_node_table",
    "NODE_METADATA_FIELDS",
    "NODE_METADATA_REQUIRED_FIELDS",
    "validate_node_metadata",
    "is_string_like",
]


class SchemaError(ValueError):
    """Raised when a dataframe does not conform to a SubSense schema."""


@dataclass(frozen=True)
class SchemaReport:
    """Structural schema check result (does not raise)."""

    missing: list[str]
    unknown: list[str]
    dtype_problems: list[str]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unknown or self.dtype_problems)


def dataset_schema_fields() -> list[str]:
    """The exact §11 field list, in PRD order, from configs/feature_schema_v1.yaml."""
    fields = feature_schema_config()["dataset_fields"]
    if not isinstance(fields, list) or not fields:
        raise SchemaError("feature_schema_v1.yaml:dataset_fields must be a non-empty list")
    return [str(f) for f in fields]


#: §12 label columns carried alongside the §11 fields (never collapsed — T-026).
RAW_NODE_LABEL_FIELDS = ("anomaly_label", "fault_label", "progression_label", "risk_label")

#: Exact column list of the raw per-timestep node table (synthetic_nodes.csv
#: layout): §9 raw stream + coordinates/event key + §12 labels.
RAW_NODE_TABLE_FIELDS = (
    "event_id",
    "timestamp",
    "node_id",
    "x",
    "y",
    "tilt_x",
    "tilt_y",
    "tilt_magnitude",
    "displacement",
    "strain",
    "vibration_rms",
    "vibration_peak",
    "battery",
    "RSSI",
    "SNR",
    "packet_loss",
    *RAW_NODE_LABEL_FIELDS,
)

#: §8.4 calibration record folded into node metadata (T-025).
NODE_METADATA_FIELDS = (
    "node_id",
    "x",
    "y",
    "lat",
    "lon",
    "tilt_x_offset",
    "tilt_y_offset",
    "displacement_offset",
    "temperature_coefficient",
    "calibration_date",
    "drift_history",
)

#: Node metadata cannot be traced without these (§8.4, §10.1 traceability).
NODE_METADATA_REQUIRED_FIELDS = ("node_id", "calibration_date")


def is_string_like(series: pd.Series) -> bool:
    """True if a series holds strings (object/string dtype, or all-str values)."""
    if isinstance(series.dtype, pd.StringDtype) or series.dtype == object:
        non_null = series.dropna()
        return bool(non_null.map(lambda v: isinstance(v, str)).all())
    return False


def _dtype_problems(df: pd.DataFrame, fields: tuple[str, ...]) -> list[str]:
    """Coarse dtype contract for the fields downstream stages rely on."""
    problems: list[str] = []
    for field in ("node_id", "event_id"):
        if field in fields and field in df.columns and not is_string_like(df[field]):
            problems.append(f"{field} must be a string identifier")
    if "timestamp" in fields and "timestamp" in df.columns:
        ts = df["timestamp"]
        if not (pd.api.types.is_numeric_dtype(ts) or pd.api.types.is_datetime64_any_dtype(ts)):
            problems.append("timestamp must be numeric or datetime")
    for field in ("risk_label", "progression_label"):
        if field in fields and field in df.columns and not is_string_like(df[field]):
            problems.append(f"{field} must be a string label (§12)")
    return problems


def schema_violations(
    df: pd.DataFrame,
    *,
    fields: tuple[str, ...] | None = None,
    require_all: bool = True,
    forbid_unknown: bool = True,
) -> SchemaReport:
    """Structural check against ``fields`` (default: the §11 list) without raising."""
    expected = list(fields) if fields is not None else dataset_schema_fields()
    present = list(df.columns)
    missing = [c for c in expected if c not in present] if require_all else []
    unknown = [c for c in present if c not in set(expected)] if forbid_unknown else []
    return SchemaReport(
        missing=missing,
        unknown=unknown,
        dtype_problems=_dtype_problems(df, tuple(expected)),
    )


def validate_schema(
    df: pd.DataFrame,
    *,
    require_all: bool = True,
    forbid_unknown: bool = True,
) -> None:
    """Validate a dataframe against the exact §11 field list.

    Raises ``SchemaError`` listing every missing and/or unknown column; the
    message must be actionable because §9 makes data-quality states
    first-class, never silently repaired.
    """
    report = schema_violations(
        df, require_all=require_all, forbid_unknown=forbid_unknown
    )
    problems: list[str] = []
    if report.missing:
        problems.append(f"missing §11 fields: {report.missing}")
    if report.unknown:
        problems.append(f"unknown columns (not in §11): {report.unknown}")
    problems.extend(report.dtype_problems)
    if problems:
        raise SchemaError("schema validation failed — " + "; ".join(problems))


def validate_raw_node_table(df: pd.DataFrame) -> None:
    """Validate the raw per-timestep node table (preprocessing input layout).

    This is the table every T-028…T-033 stage consumes; the §11 feature-store
    schema (``validate_schema``) applies *after* feature generation.
    """
    report = schema_violations(
        df, fields=RAW_NODE_TABLE_FIELDS, require_all=True, forbid_unknown=False
    )
    if report.missing:
        raise SchemaError(f"raw node table is missing required columns: {report.missing}")
    if report.dtype_problems:
        raise SchemaError("raw node table dtype violations — " + "; ".join(report.dtype_problems))


def validate_node_metadata(meta: pd.DataFrame) -> None:
    """Validate the §8.4 per-node calibration/metadata record.

    Requires ``node_id`` and ``calibration_date`` (§8.4 traceability) and
    rejects unknown calibration fields so the record stays auditable.
    """
    missing = [c for c in NODE_METADATA_REQUIRED_FIELDS if c not in meta.columns]
    unknown = [c for c in meta.columns if c not in set(NODE_METADATA_FIELDS)]
    problems: list[str] = []
    if missing:
        problems.append(f"missing §8.4 calibration fields: {missing}")
    if unknown:
        problems.append(f"unknown node-metadata columns: {unknown}")
    if "calibration_date" in meta.columns:
        if meta["calibration_date"].isna().any():
            problems.append("calibration_date must be present for every node (§8.4)")
    if problems:
        raise SchemaError("node metadata validation failed — " + "; ".join(problems))
