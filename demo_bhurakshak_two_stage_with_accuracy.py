#!/usr/bin/env python3
"""Demonstrate v3 two-stage inference and report window-level validation accuracy.

Run from the BHURAKSHAK repository root. Uses development observations for the
live example and saved, labelled reserved-validation predictions for metrics.
It does not read or reuse the locked independent-test examples.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.features.causal_v3 import build_features
from src.risk.severity_v3 import TwoStageClassifier

CLASSES = ["NORMAL", "WARNING", "CRITICAL"]


def show_accuracy(prediction_path: Path) -> None:
    if not prediction_path.exists():
        raise SystemExit(
            f"Validation predictions not found: {prediction_path}\n"
            "Use the saved v3 reserved-validation predictions; do not use the "
            "locked independent-test data for an ad-hoc demo."
        )

    df = pd.read_csv(prediction_path)
    required = {"truth", "prediction"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV must contain columns: {sorted(required)}")
    if df.empty or df[list(required)].isna().any().any():
        raise SystemExit("Validation predictions are empty or contain missing labels.")

    y_true = df["truth"].astype(str)
    y_pred = df["prediction"].astype(str)
    unknown = (set(y_true) | set(y_pred)) - set(CLASSES)
    if unknown:
        raise SystemExit(f"Unexpected labels: {sorted(unknown)}")

    print("\n=== RESERVED DEVELOPMENT VALIDATION METRICS ===")
    print(f"Samples (windows): {len(df):,}")
    print(f"Accuracy:          {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred) * 100:.2f}%")
    print("\nPer-class report:")
    print(classification_report(
        y_true, y_pred, labels=CLASSES, target_names=CLASSES,
        digits=4, zero_division=0
    ))
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print("Confusion matrix (rows=true, columns=predicted; NORMAL, WARNING, CRITICAL):")
    print(cm)
    print(
        "\nNote: metrics are window-level results on reserved synthetic validation data, "
        "not independent-test or real-mine accuracy."
    )


def show_two_stage(bundle_path: Path, input_path: Path, event_id: str | None) -> None:
    if not bundle_path.exists():
        raise SystemExit(f"Model bundle not found: {bundle_path}")
    if not input_path.exists():
        raise SystemExit(f"Development observations not found: {input_path}")

    artifacts = joblib.load(bundle_path)
    bundle = artifacts["selected"]
    pipeline = bundle.model
    estimator = pipeline.named_steps["estimator"]
    if not isinstance(estimator, TwoStageClassifier):
        raise SystemExit("The selected saved artifact is not a TwoStageClassifier.")

    raw = (
        pd.read_parquet(input_path)
        if input_path.suffix.lower() == ".parquet"
        else pd.read_csv(input_path)
    )
    if event_id is None:
        event_id = str(raw["event_id"].iloc[0])
        print(f"\nDemo event: first development event ({event_id})")
    raw = raw[raw["event_id"].astype(str) == str(event_id)].copy()
    if raw.empty:
        raise SystemExit(f"Event {event_id!r} was not found in development observations.")

    frame = build_features(raw)
    if frame.empty:
        raise SystemExit("Event is too short to produce a complete v3 feature window.")

    x = pipeline.named_steps["imputer"].transform(
        frame[bundle.features].to_numpy(dtype=float)
    )

    # Stage 1: Critical versus non-Critical.
    p_critical = estimator.critical_.predict_proba(x)[:, 1]
    # Stage 2: Warning versus Normal, conditional on not being Critical.
    p_warning_given_noncritical = estimator.warning_.predict_proba(x)[:, 1]

    # Final calibrated probabilities and threshold-based decisions.
    final_p = bundle.predict_proba(frame)
    final_decision = bundle.predict(frame)

    result = pd.DataFrame({
        "window": frame["window_index"].to_numpy(),
        "displacement_mm": frame["displacement_endpoint"].to_numpy(),
        "stage1_critical_%": 100 * p_critical,
        "stage2_warning_given_noncritical_%": 100 * p_warning_given_noncritical,
        "final_normal_%": 100 * final_p[:, 0],
        "final_warning_%": 100 * final_p[:, 1],
        "final_critical_%": 100 * final_p[:, 2],
        "decision": final_decision,
    })
    print("\n=== TWO-STAGE INFERENCE DEMO (DEVELOPMENT EVENT) ===")
    print("Stage 1: Critical vs. non-Critical")
    print("Stage 2: Warning vs. Normal, conditional on not being Critical")
    print(f"Decision thresholds: {bundle.thresholds}")
    print(result.tail(8).to_string(index=False, formatters={
        col: (lambda v: f"{v:.2f}") for col in result.columns if col.endswith("%") or col == "displacement_mm"
    }))
    print("These are actual model outputs for the selected development event, not an accuracy score.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="reports/generalization_v3/candidates.joblib")
    parser.add_argument("--input", default="reports/generalization_v3/development_observations.parquet")
    parser.add_argument("--validation-predictions", default="reports/generalization_v3/validation_predictions.csv")
    parser.add_argument("--event-id", help="Optional development event ID for the inference demo")
    parser.add_argument("--metrics-only", action="store_true", help="Show only saved validation metrics")
    args = parser.parse_args()

    show_accuracy(Path(args.validation_predictions))
    if not args.metrics_only:
        show_two_stage(Path(args.bundle), Path(args.input), args.event_id)


if __name__ == "__main__":
    main()
