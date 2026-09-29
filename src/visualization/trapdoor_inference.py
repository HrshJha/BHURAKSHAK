"""Feature and inference adapter for the interactive one-trapdoor demo.

The model is the repository's frozen legacy tabletop Random Forest and
Isolation Forest. This adapter extracts the same 2 s / 10 Hz features used by
``scripts/generate_tabletop_dataset.py`` from simulated sensor samples. It does
not train, tune, or write model/evaluation artifacts.
"""
from __future__ import annotations

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WINDOW_SAMPLES = 20
SAMPLE_HZ = 10.0
FEATURES = [
    "roll", "pitch", "vibration_rms", "accel_peak", "tof_delta", "tof_rate",
    "displacement_mm", "displacement_rate", "relative_tilt", "relative_vibration",
]
CLASS_NAMES = {0: "NORMAL", 1: "WARNING", 2: "WARNING", 3: "CRITICAL"}
CLASS_DETAIL = {0: "Stable", 1: "Initiation", 2: "Progressive", 3: "Critical"}


def _slope(values: np.ndarray) -> float:
    t = np.arange(len(values), dtype=float) / SAMPLE_HZ
    return float(np.polyfit(t, values, 1)[0]) if len(values) > 1 else 0.0


def extract_trapdoor_features(samples: list[dict]) -> dict:
    """Extract one active-node window using the saved corpus feature contract.

    Payload readings are quantized sensor outputs in degrees, mm, and g.
    The first complete 2 s is the local sensor baseline, matching the data
    generator's initial-window baseline convention. N1/N3/N4 are reference
    nodes in this one-door visualization; they never represent other doors.
    """
    if len(samples) < WINDOW_SAMPLES:
        raise ValueError(f"need {WINDOW_SAMPLES} samples for a 2 s feature window")
    if not all({"N1", "N2", "N3", "N4"} <= set(row.get("nodes", {})) for row in samples):
        raise ValueError("each sample must include N1, N2, N3, and N4")

    baseline = samples[:WINDOW_SAMPLES]
    window = samples[-WINDOW_SAMPLES:]
    active = [row["nodes"]["N2"] for row in window]
    references = {
        node: [row["nodes"][node] for row in window]
        for node in ("N1", "N3", "N4")
    }

    def values(series: list[dict], key: str) -> np.ndarray:
        return np.asarray([float(point[key]) for point in series if point.get(key) is not None], dtype=float)

    active_pitch = values(active, "pitch_deg")
    active_roll = values(active, "roll_deg")
    active_accel = values(active, "accel_g")
    active_tof = values(active, "tof_mm")
    active_ultra = values(active, "ultrasonic_mm")
    reference_pitch = np.mean([
        np.mean(values(references[node], "pitch_deg")) for node in references
    ])
    reference_vibration = np.mean([
        np.std(values(references[node], "accel_g"), ddof=0) for node in references
    ])
    base_tof = float(np.mean(values([row["nodes"]["N2"] for row in baseline], "tof_mm")))
    base_ultra = float(np.mean(values([row["nodes"]["N2"] for row in baseline], "ultrasonic_mm")))

    return {
        "roll": round(float(np.mean(active_roll)), 3),
        "pitch": round(float(np.mean(active_pitch)), 3),
        "vibration_rms": round(float(np.std(active_accel, ddof=0)), 5),
        "accel_peak": round(float(np.max(active_accel - np.median(active_accel))), 5),
        "tof_delta": round(float(np.mean(active_tof) - base_tof), 3),
        "tof_rate": round(_slope(active_tof), 4),
        "displacement_mm": round(float(np.mean(active_ultra) - base_ultra), 3),
        "displacement_rate": round(_slope(active_ultra), 4),
        "relative_tilt": round(float(np.mean(active_pitch) - reference_pitch), 3),
        "relative_vibration": round(float(np.std(active_accel, ddof=0) - reference_vibration), 5),
    }


class TrapdoorInference:
    """Read-only access to frozen tabletop models."""

    def __init__(self, repo_root: Path = ROOT):
        self.repo_root = Path(repo_root)
        rf_artifact = joblib.load(self.repo_root / "models/tabletop/random_forest.joblib")
        if_artifact = joblib.load(self.repo_root / "models/tabletop/isolation_forest.joblib")
        self.rf = rf_artifact["model"]
        self.rf_features = rf_artifact["features"]
        self.if_model = if_artifact["model"]
        self.if_scaler = if_artifact["scaler"]
        self.if_features = if_artifact["features"]
        self.if_threshold = float(if_artifact["threshold"])

    def predict(self, samples: list[dict]) -> dict:
        features = extract_trapdoor_features(samples)
        vector = np.asarray([[features[name] for name in self.rf_features]], dtype=float)
        probabilities = self.rf.predict_proba(vector)[0]
        class_id = int(self.rf.predict(vector)[0])
        if_values = pd.DataFrame([features], columns=self.if_features)
        scaled = self.if_scaler.transform(if_values)
        anomaly_score = float(-self.if_model.score_samples(scaled)[0])
        anomaly_flag = anomaly_score > self.if_threshold
        # Match the saved tabletop pipeline: a strong IF anomaly promotes an
        # otherwise-stable RF result to Critical. The raw RF class is retained.
        alert_class = 3 if class_id == 0 and anomaly_flag else class_id
        probability_summary = {name: 0.0 for name in ("NORMAL", "WARNING", "CRITICAL")}
        for label, probability in zip(self.rf.classes_, probabilities):
            probability_summary[CLASS_NAMES[int(label)]] += float(probability)
        return {
            "features": features,
            "class_id": class_id,
            "class_detail": CLASS_DETAIL[class_id],
            "prediction": CLASS_NAMES[alert_class],
            "rf_prediction": CLASS_NAMES[class_id],
            "probabilities": probability_summary,
            "anomaly_score": anomaly_score,
            "anomaly_threshold": self.if_threshold,
            "anomaly_flag": bool(anomaly_flag),
            "source": "frozen tabletop RF + IF; generated stand-in training data",
            "safety_alert": False,
            "sampled_at": time.time(),
        }
