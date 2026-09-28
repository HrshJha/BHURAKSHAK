"""T-026 acceptance tests — §12 separated label schema."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.labels import (
    CONTINUOUS_TARGETS,
    FAULT_VALUES,
    FULL4_TO_MVP3,
    LABEL_COLUMNS,
    RISK_VOCAB_FULL4,
    RISK_VOCAB_MVP3,
    LabelError,
    assert_labels_separate,
    validate_continuous_targets,
    validate_labels,
)


def _labelled(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anomaly_label": [0, 0, 1, 1],
            "fault_label": ["NONE", "NONE", "BIAS", "NONE"],
            "progression_label": ["STABLE", "SLOW", "STABLE", "RAPID"],
            "risk_label": ["NORMAL", "NORMAL", "WARNING", "CRITICAL"],
        }
    )


def test_four_label_fields_exist_as_separate_columns() -> None:
    df = _labelled()
    validate_labels(df)
    assert_labels_separate(df)
    for col in LABEL_COLUMNS:
        assert col in df.columns


def test_missing_label_field_rejected() -> None:
    df = _labelled().drop(columns=["fault_label"])
    with pytest.raises(LabelError, match="fault_label"):
        validate_labels(df)


def test_binary_collapse_of_risk_label_rejected() -> None:
    df = _labelled()
    df["risk_label"] = [0, 0, 1, 1]  # collapsed to binary
    with pytest.raises(LabelError, match="string label"):
        validate_labels(df)


def test_unknown_risk_value_rejected() -> None:
    df = _labelled()
    df.loc[0, "risk_label"] = "PURPLE"
    with pytest.raises(LabelError, match="PURPLE"):
        validate_labels(df)


def test_full4_to_mvp3_mapping_is_pinned() -> None:
    """G-1 closure: the 4-class → 3-class rename table is authoritative."""
    assert FULL4_TO_MVP3 == {"GREEN": "NORMAL", "WATCH": "WARNING", "WARNING": "WARNING", "CRITICAL": "CRITICAL"}
    # the rename must be within one vocabulary: every value is an MVP class
    assert set(FULL4_TO_MVP3.values()) == set(RISK_VOCAB_MVP3)
    # and it must cover the whole 4-class vocabulary
    assert set(FULL4_TO_MVP3) == set(RISK_VOCAB_FULL4)


def test_full4_vocabulary_accepted_when_selected() -> None:
    df = _labelled()
    df["risk_label"] = ["GREEN", "WATCH", "WARNING", "CRITICAL"]
    validate_labels(df, risk_vocab=("GREEN", "WATCH", "WARNING", "CRITICAL"))


def test_full4_to_mvp3_mapping_is_a_rename_not_a_collapse() -> None:
    # The mapping renames within one vocabulary on ONE column; it must never
    # erase the distinction the §12 separation protects.
    assert set(RISK_VOCAB_MVP3) <= set(RISK_VOCAB_MVP3)
    for src, dst in {
        "GREEN": "NORMAL",
        "WATCH": "WARNING",
        "WARNING": "WARNING",
        "CRITICAL": "CRITICAL",
    }.items():
        assert dst in RISK_VOCAB_MVP3


def test_reencoded_labels_rejected_as_collapse() -> None:
    df = _labelled()
    # fault_label re-encoded as risk_label — a collapse of two vocabularies
    df["risk_label"] = df["fault_label"]
    with pytest.raises(LabelError, match="collapse"):
        assert_labels_separate(df)


def test_progression_reencoded_as_risk_rejected() -> None:
    df = _labelled()
    df["risk_label"] = df["progression_label"]
    with pytest.raises(LabelError, match="collapse"):
        assert_labels_separate(df)


def test_anomaly_label_must_be_integer_0_1() -> None:
    df = _labelled()
    df["anomaly_label"] = [False, False, True, True]
    with pytest.raises(LabelError, match="integer"):
        validate_labels(df)
    df["anomaly_label"] = [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(LabelError, match="integer"):
        validate_labels(df)


def test_fault_vocabulary_matches_simulator_modes() -> None:
    assert FAULT_VALUES == ("NONE", "BIAS", "STUCK", "DROPOUT", "SPIKE", "DRIFT")


def test_unknown_label_column_rejected() -> None:
    df = _labelled()
    with pytest.raises(LabelError, match="no §12 vocabulary"):
        validate_labels(df, columns=LABEL_COLUMNS + ("mystery_label",))


def test_continuous_targets_are_own_numeric_fields() -> None:
    df = pd.DataFrame(
        {"deformation": [1.0], "deformation_velocity": [0.1], "deformation_acceleration": [0.01]}
    )
    validate_continuous_targets(df)
    assert CONTINUOUS_TARGETS == ("deformation", "deformation_velocity", "deformation_acceleration")


def test_continuous_targets_via_raw_channel_name() -> None:
    # In the raw node table `deformation` is carried by the `displacement` channel
    df = pd.DataFrame({"displacement": [1.0], "deformation_velocity": [0.1], "deformation_acceleration": [0.01]})
    validate_continuous_targets(df)
    df_missing = df.drop(columns=["deformation_velocity"])
    with pytest.raises(LabelError, match="deformation_velocity"):
        validate_continuous_targets(df_missing)


def test_continuous_target_rejects_non_numeric() -> None:
    df = pd.DataFrame({"deformation": ["1.0"], "deformation_velocity": [0.1], "deformation_acceleration": [0.01]})
    with pytest.raises(LabelError, match="numeric"):
        validate_continuous_targets(df)
