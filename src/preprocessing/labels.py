"""§12 label schema — separated labels, never collapsed (T-026).

PRD §12: labels are kept **separate**, never collapsed into one binary flag —
collapsing loses the distinction between "sensor is broken" and "ground is
moving", the single most important design decision in the labelling strategy.

This module is the runtime authority for that rule:

- the four §12 label fields (``anomaly_label``, ``fault_label``,
  ``progression_label``, ``risk_label``) plus the continuous targets are
  declared and validated independently;
- ``assert_labels_separate`` raises if a table is missing any §12 label field
  or if two label fields are re-encodings of one another (a collapse);
- vocabularies mirror the simulator (§10/T-016/T-018): ``fault_label`` carries
  the five injected fault modes (incl. ``SPIKE``) plus ``NONE``; ``risk_label``
  uses the 3-class MVP vocabulary NORMAL/WARNING/CRITICAL with the full
  4-class vocabulary available for the future expansion.

  G-1 mapping (resolved, T-050 shipped): the §12 GREEN slot is filled by
  NORMAL — ``FULL4_TO_MVP3`` below is the authoritative rename table
  (GREEN→NORMAL, WATCH→WARNING, WARNING→WARNING, CRITICAL→CRITICAL), and the
  §10 scenario taxonomy is mapped to ``risk_label`` at generation time in
  ``src/simulator/scenarios.py`` (per-scenario, documented there and in
  docs/label_mapping.md). The §10-only labels (SENSOR_FAULT, DATA_QUALITY,
  LOCAL_ANOMALY, NON_SUBSIDENCE, COMMUNICATION_FAILURE, MIXED) are carried by
  the separate ``fault_label`` / ``anomaly_label`` / ``data_quality_label``
  fields — never collapsed into ``risk_label`` (§12).
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing.schema import RAW_NODE_LABEL_FIELDS, is_string_like

__all__ = [
    "LabelError",
    "LABEL_COLUMNS",
    "CONTINUOUS_TARGETS",
    "ANOMALY_VALUES",
    "FAULT_VALUES",
    "PROGRESSION_VALUES",
    "RISK_VOCAB_MVP3",
    "RISK_VOCAB_FULL4",
    "FULL4_TO_MVP3",
    "validate_labels",
    "assert_labels_separate",
    "validate_continuous_targets",
]


class LabelError(ValueError):
    """Raised on label-schema violations (§12)."""


LABEL_COLUMNS: tuple[str, ...] = RAW_NODE_LABEL_FIELDS

#: §12 continuous targets — stored as their own numeric fields, never as labels.
CONTINUOUS_TARGETS = ("deformation", "deformation_velocity", "deformation_acceleration")

ANOMALY_VALUES = (0, 1)

#: NONE + the five injectable fault modes from §10/T-016.
FAULT_VALUES = ("NONE", "BIAS", "STUCK", "DROPOUT", "SPIKE", "DRIFT")

PROGRESSION_VALUES = ("STABLE", "SLOW", "ACCELERATING", "RAPID")

#: §12: start with 3 classes (MVP); the 4-class expansion path is kept ready.
#: G-1 mapping is RESOLVED (see module docstring + docs/label_mapping.md).
RISK_VOCAB_MVP3 = ("NORMAL", "WARNING", "CRITICAL")
RISK_VOCAB_FULL4 = ("GREEN", "WATCH", "WARNING", "CRITICAL")

#: Explicit, documented mapping for the future expansion (not a collapse:
#: it is a rename within one vocabulary, applied to one column only).
FULL4_TO_MVP3 = {"GREEN": "NORMAL", "WATCH": "WARNING", "WARNING": "WARNING", "CRITICAL": "CRITICAL"}


def _check_vocabulary(df: pd.DataFrame, column: str, allowed: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    if column not in df.columns:
        return [f"missing §12 label field: {column}"]
    if not is_string_like(df[column]):
        return [f"{column} must be a string label (binary/numeric encoding is a §12 collapse)"]
    values = set(df[column].dropna().unique())
    unexpected = values - set(allowed)
    if unexpected:
        problems.append(f"{column} has values outside {list(allowed)}: {sorted(unexpected)}")
    if df[column].isna().any():
        problems.append(f"{column} contains NaN")
    return problems


def validate_labels(
    df: pd.DataFrame,
    *,
    risk_vocab: tuple[str, ...] = RISK_VOCAB_MVP3,
    columns: tuple[str, ...] = LABEL_COLUMNS,
) -> None:
    """Validate the §12 label fields on a raw node table. Raises ``LabelError``."""
    problems: list[str] = []
    for column in columns:
        if column == "anomaly_label":
            if column not in df.columns:
                problems.append("missing §12 label field: anomaly_label")
                continue
            if not pd.api.types.is_integer_dtype(df[column]):
                problems.append("anomaly_label must be integer 0/1, not a boolean or float flag")
            else:
                bad = set(df[column].dropna().unique()) - set(ANOMALY_VALUES)
                if bad:
                    problems.append(f"anomaly_label values outside {list(ANOMALY_VALUES)}: {sorted(bad)}")
                if df[column].isna().any():
                    problems.append("anomaly_label contains NaN")
        elif column == "fault_label":
            problems.extend(_check_vocabulary(df, "fault_label", FAULT_VALUES))
        elif column == "progression_label":
            problems.extend(_check_vocabulary(df, "progression_label", PROGRESSION_VALUES))
        elif column == "risk_label":
            problems.extend(_check_vocabulary(df, "risk_label", risk_vocab))
        else:  # defensive — future label fields must declare a vocabulary
            problems.append(f"unknown label column {column!r} (no §12 vocabulary declared)")
    if problems:
        raise LabelError("label validation failed — " + "; ".join(problems))


def assert_labels_separate(df: pd.DataFrame) -> None:
    """Assert the four §12 label fields exist independently (no collapse).

    A collapse is any state where a §12 label field is missing *or* two label
    fields are re-encodings of one another (equal after factorising values) —
    e.g. ``risk_label`` overwritten with a copy of ``fault_label``, or labels
    squeezed into a single binary flag column.
    """
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise LabelError(f"§12 collapse detected — label fields missing: {missing}")
    encoded = {c: df[c].astype("category").cat.codes for c in LABEL_COLUMNS}
    names = list(LABEL_COLUMNS)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if encoded[a].equals(encoded[b]):
                raise LabelError(
                    f"§12 collapse detected — {a} and {b} are re-encodings of one another"
                )


def validate_continuous_targets(
    df: pd.DataFrame,
    *,
    required: tuple[str, ...] = CONTINUOUS_TARGETS,
    column_for: dict[str, str] | None = None,
) -> None:
    """Assert §12 continuous targets exist as their own numeric fields.

    ``column_for`` maps a §12 target name onto the carrying column when the
    series is stored under its raw channel name (e.g. ``deformation`` is
    carried by the ``displacement`` channel in the raw node table).
    """
    column_for = {"deformation": "displacement", **(column_for or {})}
    problems: list[str] = []
    for target in required:
        if target in df.columns:
            column = target
        else:
            column = column_for.get(target, target)
        if column not in df.columns:
            problems.append(f"missing continuous target {target!r} (column {column!r})")
        elif not pd.api.types.is_numeric_dtype(df[column]):
            problems.append(f"continuous target {target!r} (column {column!r}) must be numeric")
    if problems:
        raise LabelError("continuous-target validation failed — " + "; ".join(problems))
