"""Acceptance tests for the Phase 0 config files (T-003, T-004, T-005, T-006).

These encode the PRD values verbatim so that any config drift from §8.3,
§21.1, §10 and §11/§13 fails loudly.
"""

from __future__ import annotations

import pytest

from src.config import (
    alerts_config,
    deescalation_multiplier,
    escalation_thresholds,
    feature_schema_config,
    physics_config,
    physics_parameters,
    sampling_config,
)


# ---------------------------------------------------------------------------
# T-003 — configs/sampling.yaml must match PRD §8.3 exactly
# ---------------------------------------------------------------------------


def test_sampling_matches_prd() -> None:
    cfg = sampling_config()
    ch = cfg["channels"]

    tilt = ch["tilt"]
    assert tilt["default_hz"] == 2.0, "§8.3: tilt default 2 Hz"
    assert tilt["min_hz"] == 1.0 and tilt["max_hz"] == 10.0, "§8.3: tilt range 1–10 Hz"

    assert ch["displacement"]["default_hz"] == 1.0, "§8.3: displacement 1 Hz"

    vib = ch["vibration"]
    assert vib["on_node_sampling_hz"] >= 100.0, "§8.3: vibration sampled on-node at ≥100 Hz"
    assert vib["min_on_node_hz"] >= 100.0
    assert vib["transmitted_summary_hz"] == 1.0, "§8.3: vibration summarised to 1 Hz uplink"

    assert ch["crack_width"]["default_hz"] == 0.1, "§8.3: crack width 0.1 Hz"
    assert ch["temperature"]["default_hz"] == 0.1, "§8.3: temperature 0.1 Hz"

    adaptive = cfg["adaptive_sampling"]
    assert adaptive["rate_multiplier"] == 3.0, "§8.3: adaptive-sampling multiplier default 3×"


def test_sampling_node_metadata_piggybacks() -> None:
    """§8.3: battery/RSSI/SNR/packet_loss attach to every uplink, no separate schedule."""
    cfg = sampling_config()
    assert cfg["node_metadata"]["attach_to"] == "every_uplink_packet"


# ---------------------------------------------------------------------------
# T-004 — configs/alerts.yaml must match PRD §21.1 exactly
# ---------------------------------------------------------------------------


def test_alerts_match_prd() -> None:
    cfg = alerts_config()
    esc = escalation_thresholds()

    g2w = esc["GREEN_to_WATCH"]
    assert g2w["class_threshold"] == 0.5, "§21.1: GREEN→WATCH threshold 0.5"
    assert g2w["persistence_windows"] == 3, "§21.1: GREEN→WATCH persistence 3 windows"

    w2w = esc["WATCH_to_WARNING"]
    assert w2w["class_threshold"] == 0.6, "§21.1: WATCH→WARNING threshold 0.6"
    assert w2w["persistence_windows"] == 3, "§21.1: WATCH→WARNING persistence 3 windows"
    assert "spatial_coherence_above_threshold" in w2w["requires"]
    assert "displacement_trend_positive" in w2w["requires"]

    c = esc["WARNING_to_CRITICAL"]
    assert c["class_threshold"] == 0.7, "§21.1: WARNING→CRITICAL threshold 0.7"
    assert c["persistence_windows"] == 2, "§21.1: WARNING→CRITICAL persistence 2 windows"
    assert c["min_confirming_neighbours"] == 2, "§21.1: CRITICAL needs ≥2 neighbouring nodes"
    assert "physics_residual_low" in c["requires"]

    assert deescalation_multiplier() == 1.5, "§21.1: de-escalation hysteresis 1.5×"

    assert cfg["spatial_aggregation"]["rule"] == "max", "§21.1: region state = max over nodes"
    assert cfg["manual_override"]["suppresses_model_output"] is False, (
        "§21.1: override never suppresses the recorded model output"
    )


# ---------------------------------------------------------------------------
# T-005 — configs/physics.yaml must carry the full §10 parameter set
# ---------------------------------------------------------------------------


def test_physics_params() -> None:
    required = [
        "panel_center_x",
        "panel_center_y",
        "panel_width",
        "panel_length",
        "mine_depth",
        "extraction_height",
        "subsidence_factor",
        "influence_radius",
        "time_coefficient",
        "maximum_subsidence",
    ]
    params = physics_parameters()
    for name in required:
        assert name in params, f"§10 requires physics parameter {name!r}"
        assert isinstance(params[name], float), f"{name} must be numeric"

    raw = physics_config()
    assert raw["grid"]["nodes_per_side"] == 20, "§10: 20×20 = 400 virtual nodes"


# ---------------------------------------------------------------------------
# T-006 — configs/feature_schema_v1.yaml must enumerate §11 fields and §13 groups
# ---------------------------------------------------------------------------


def test_feature_schema() -> None:
    cfg = feature_schema_config()
    assert cfg["feature_schema_version"] in ("v1", "v2"), "§10.1 manifest field: feature_schema_version (live schema)"

    # §11 dataset schema — exact field list
    expected_fields = [
        "timestamp", "node_id", "x", "y",
        "tilt_x", "tilt_y", "tilt_magnitude",
        "displacement", "strain",
        "tilt_velocity", "tilt_acceleration",
        "displacement_velocity", "displacement_acceleration",
        "vibration_rms", "vibration_peak",
        "neighbor_mean", "neighbor_std", "neighbor_anomaly_fraction", "spatial_coherence",
        "battery", "RSSI", "SNR", "packet_loss",
        "DGPS_displacement", "DGPS_velocity",
        "InSAR_displacement", "InSAR_velocity", "InSAR_coherence",
        "physics_displacement", "physics_residual",
        "anomaly_score",
        "progression_label", "risk_label",
    ]
    assert list(cfg["dataset_fields"]) == expected_fields, (
        "§11 dataset schema fields must be enumerated exactly"
    )

    # §13 feature groups A–J
    expected_groups = {
        "A_physical", "B_temporal", "C_spatial", "D_vibration", "E_sensor_health",
        "F_physics", "G_dgps", "H_insar", "I_terrain", "J_environmental",
    }
    groups = cfg["feature_groups"]
    assert set(groups.keys()) == expected_groups, "§13 groups A–J must all be enumerated"

    assert groups["A_physical"] == [
        "tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain",
    ], "§13 Group A names must match the PRD"

    # §13: Group J only via ablation
    assert groups["J_environmental"]["enabled_by_default"] is False
    assert groups["J_environmental"]["gate"] == "ablation_required"

    # §13 first-iteration budget: ~40–70 engineered features
    budget = cfg["model_input_budget"]
    assert budget["min"] == 40 and budget["max"] == 70


@pytest.mark.parametrize(
    "name",
    ["sampling", "alerts", "physics", "feature_schema"],
)
def test_all_configs_load(name: str) -> None:
    from src.config import load_config

    assert isinstance(load_config(name), dict)
