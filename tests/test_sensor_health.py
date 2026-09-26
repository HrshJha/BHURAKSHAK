"""T-033 acceptance tests — sensor-health state detection (§9, §8.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.sensor_health import (
    DEGRADED,
    DRIFT,
    HEALTHY,
    OFFSET,
    STUCK,
    SeriesHealth,
    detect_series_health,
    flag_series_health,
    sensor_health_params,
)

INTERVAL = 10.0 / 60.0
P = sensor_health_params()


def _ts(n: int) -> np.ndarray:
    return INTERVAL * np.arange(n, dtype=float)


def test_healthy_noisy_series_is_healthy() -> None:
    rng = np.random.default_rng(0)
    values = 0.01 * rng.standard_normal(144)  # tilt-scale noise around 0
    health = detect_series_health(values, P)
    assert health.state == HEALTHY, health.detail


def test_stuck_values_produce_stuck_state() -> None:
    rng = np.random.default_rng(1)
    values = 0.01 * rng.standard_normal(144)
    values[60:80] = 0.42  # 20 identical readings ≥ stuck_min_repeats (5)
    health = detect_series_health(values, P)
    assert health.state == STUCK
    assert health.stuck_run_len >= 15


def test_drift_produces_drift_state() -> None:
    n = 144
    values = np.linspace(0.0, 2.0, n) + 0.01 * np.random.default_rng(2).standard_normal(n)
    # slope 2/144 per step vs scale (median |v|) ≈ 1 → drift score well above 0.05
    health = detect_series_health(values, P)
    assert health.state == DRIFT
    assert health.drift_score > float(P["drift_score_threshold"])


def test_sudden_offset_produces_offset_state() -> None:
    rng = np.random.default_rng(3)
    values = 0.01 * rng.standard_normal(144)
    values[72:] += 1.0  # one-step 1.0 mm/deg jump ≈ 100σ for 0.01-scale noise
    health = detect_series_health(values, P)
    assert health.state == OFFSET
    assert health.offset_jump_sigma > float(P["offset_jump_sigma"])


def test_mild_drift_is_drift_not_degraded() -> None:
    """A ramp-shaped change IS drift, whatever its amplitude (§9's distinction
    is shape/separation, not severity; DEGRADED is for mild conditions but a
    clean ramp above the trend threshold is a genuine drift signature)."""
    n = 144
    values = np.linspace(0.0, 0.35, n) + 0.01 * np.random.default_rng(4).standard_normal(n)
    health = detect_series_health(values, P)
    assert health.state == DRIFT
    assert health.drift_score > float(P["drift_score_threshold"])
    assert health.offset_jump_sigma < 6.0, "no step-like jump"


def test_mild_offset_is_degraded() -> None:
    """A mid-series step changes BOTH halves' means, so the fitted trend is
    large too; shape (jump 4.4σ < 6σ hard threshold) makes it DEGRADED."""
    rng = np.random.default_rng(5)
    values = 0.01 * rng.standard_normal(144)
    values[72:] += 0.05  # ≈5σ jump: below the 6σ hard threshold, above 4σ
    health = detect_series_health(values, P)
    assert health.state == DEGRADED
    assert health.offset_jump_sigma > 4.0 and health.offset_jump_sigma <= 6.0


def test_injected_sensor_fault_separates_from_ground_movement() -> None:
    """§9 core requirement: T-016 SENSOR_FAULT is separable from movement.

    A BIAS fault is a STEP: the detector answers OFFSET with jump > 6σ. Real
    movement of the same total magnitude is a RAMP: the detector answers DRIFT
    with no step-like jump. (T-037's spatial coherence + §10 fault_label then
    finish the separation: movement is multi-node coherent, faults are not.)
    """
    rng = np.random.default_rng(6)
    noise = 0.01 * rng.standard_normal(144)
    step = noise.copy()
    step[72:] += 0.08
    ramp = np.linspace(0.0, 0.08, 144) + noise
    step_health = detect_series_health(step, P)
    ramp_health = detect_series_health(ramp, P)
    assert step_health.state == OFFSET
    assert step_health.offset_jump_sigma > float(P["offset_jump_sigma"])
    assert ramp_health.state == DRIFT
    assert ramp_health.offset_jump_sigma < float(P["offset_jump_sigma"]), (
        "a smooth ramp of equal magnitude must not look like a sudden sensor offset"
    )
    assert ramp_health.drift_score > float(P["drift_score_threshold"])


def test_empty_series_is_healthy_not_crash() -> None:
    health = detect_series_health(np.array([]), P)
    assert health.state == HEALTHY


def test_constant_zero_series_is_not_stuck_forever_edge() -> None:
    values = np.zeros(30)
    health = detect_series_health(values, P)
    assert health.state == STUCK, "constant output is the definition of stuck"


def test_flag_series_health_groups_by_event_node_channel() -> None:
    rng = np.random.default_rng(7)
    n = 144
    rows = []
    for node, fault in (("V0001", 0.0), ("V0002", 0.5)):
        for i in range(n):
            rows.append(
                {
                    "event_id": "E1",
                    "node_id": node,
                    "timestamp": _ts(n)[i],
                    "tilt_x": 0.01 * rng.standard_normal(),
                    "tilt_y": 0.01 * rng.standard_normal(),
                    "displacement": (0.02 * rng.standard_normal()) + (fault if i >= 100 else 0.0),
                }
            )
    df = pd.DataFrame(rows)
    out = flag_series_health(df, channels=("displacement",), params=P)
    assert len(out) == 2
    by_node = out.set_index("node_id")
    assert by_node.loc["V0001", "health_state"] == HEALTHY
    # +0.5 mm at i=100 with 0.02 mm noise ≈ 25σ jump → clear OFFSET (a T-016 BIAS fault)
    assert by_node.loc["V0002", "health_state"] == OFFSET
    assert by_node.loc["V0002", "offset_jump_sigma"] > float(P["offset_jump_sigma"])


def test_flag_series_health_preserves_labels_and_never_mutates() -> None:
    rng = np.random.default_rng(8)
    n = 20
    df = pd.DataFrame(
        {
            "event_id": ["E1"] * n,
            "node_id": ["V0001"] * n,
            "timestamp": _ts(n),
            "tilt_x": 0.01 * rng.standard_normal(n),
            "fault_label": ["NONE"] * n,
        }
    )
    out = flag_series_health(df, channels=("tilt_x",), params=P)
    assert "fault_label" not in out.columns
    assert "fault_label" in df.columns
    assert list(df["fault_label"]) == ["NONE"] * n


def test_params_come_from_config() -> None:
    from src.config import load_config

    assert sensor_health_params() == dict(load_config("preprocessing")["sensor_health"])


def test_missing_config_keys_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.preprocessing.sensor_health as sh

    real = sh.load_config

    def fake(name: str):
        data = real(name)
        if name == "preprocessing":
            data = {**data, "sensor_health": {"drift_score_threshold": 0.05}}
        return data

    monkeypatch.setattr(sh, "load_config", fake)
    with pytest.raises(sh.SensorHealthError):
        sh.sensor_health_params()
