from __future__ import annotations

import numpy as np
import pytest

from src.visualization.trapdoor_inference import TrapdoorInference, extract_trapdoor_features


def sample_series(*, displacement: float = 0.0, slope: float = 0.0, count: int = 20):
    rows = []
    for index in range(count):
        active = {
            "pitch_deg": slope,
            "roll_deg": 0.0,
            "accel_g": 1.0 + 0.001 * (-1) ** index,
            "tof_mm": 120.0 + displacement + slope * index / 10,
            "ultrasonic_mm": 220.0 + displacement + slope * index / 10,
        }
        refs = {
            name: {
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "accel_g": 1.0 + 0.0005 * (-1) ** index,
                "tof_mm": 120.0,
                "ultrasonic_mm": 220.0,
            }
            for name in ("N1", "N3", "N4")
        }
        rows.append({"timestamp_s": index / 10, "nodes": {"N2": active, **refs}})
    return rows


def test_requires_a_complete_two_second_window():
    with pytest.raises(ValueError, match="20 samples"):
        extract_trapdoor_features(sample_series(count=19))


def test_feature_extraction_uses_distance_delta_and_relative_tilt():
    rows = sample_series(count=40)
    for offset, row in enumerate(rows[20:]):
        travel = 8.25 * offset / 19
        row["nodes"]["N2"]["pitch_deg"] = 0.42
        row["nodes"]["N2"]["tof_mm"] += travel
        row["nodes"]["N2"]["ultrasonic_mm"] += travel
    features = extract_trapdoor_features(rows)
    assert features["displacement_mm"] == pytest.approx(4.125)
    assert features["tof_delta"] == pytest.approx(4.125)
    assert features["relative_tilt"] == pytest.approx(0.42)
    assert features["displacement_rate"] > 0


def test_frozen_model_predicts_from_extracted_sensor_features():
    result = TrapdoorInference().predict(sample_series(count=40))
    assert result["prediction"] in {"NORMAL", "WARNING", "CRITICAL"}
    assert result["rf_prediction"] in {"NORMAL", "WARNING", "CRITICAL"}
    assert result["safety_alert"] is False
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert set(result["features"]) == {
        "roll", "pitch", "vibration_rms", "accel_peak", "tof_delta", "tof_rate",
        "displacement_mm", "displacement_rate", "relative_tilt", "relative_vibration",
    }
