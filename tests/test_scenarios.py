""" acceptance tests — all 12 scenarios generatable with correct labels."""

from __future__ import annotations

import numpy as np

from src.simulator.scenarios import (
    ANOMALY_FLAG_LOCAL,
    ANOMALY_FLAG_NON_SUBSIDENCE,
    COMM_FAILURE_FLAG,
    DATA_QUALITY_FLAG,
    PROGRESSION_ACCELERATING,
    PROGRESSION_RAPID,
    PROGRESSION_SLOW,
    RISK_CRITICAL,
    RISK_NORMAL,
    RISK_WARNING,
    TAXONOMY_ROWS,
    Scenario,
    scenario_risk_mapping,
)


def _gen(scenario: Scenario, seed: int = 0, **kwargs):
    from src.simulator.scenarios import generate_scenario

    rng = np.random.default_rng(seed)
    return generate_scenario(
        scenario, node_x=0.0, node_y=0.0, node_id="N01", rng=rng, **kwargs
    )


def test_all_16_generator_scenarios_run() -> None:
    for s in Scenario:
        res = _gen(s, seed=1)
        assert res.n_steps > 0, f"{s} must generate steps"
        assert res.tilt_x_deg.size == res.n_steps == res.displacement_mm.size


def test_twelve_taxonomy_rows_documented() -> None:
    assert len(TAXONOMY_ROWS) == 12, " defines exactly 12 scenario rows"
    assert "stable ground" in TAXONOMY_ROWS
    assert "communication failure" in TAXONOMY_ROWS


def test_deformation_scenarios_risk_mapping() -> None:
    mapping = scenario_risk_mapping()
    assert mapping["slow_subsidence"] == RISK_WARNING
    assert mapping["accelerating_subsidence"] == RISK_WARNING
    assert mapping["rapid_subsidence"] == RISK_CRITICAL
    assert mapping["irregular_subsidence"] == RISK_WARNING
    assert mapping["stable_ground"] == RISK_NORMAL
    assert mapping["vibration_only"] == RISK_NORMAL


def test_progression_labels() -> None:
    assert _gen(Scenario.SLOW_SUBSIDENCE).progression_label == PROGRESSION_SLOW
    assert _gen(Scenario.ACCELERATING_SUBSIDENCE).progression_label == PROGRESSION_ACCELERATING
    assert _gen(Scenario.RAPID_SUBSIDENCE).progression_label == PROGRESSION_RAPID
    assert _gen(Scenario.STABLE_GROUND).progression_label == "STABLE"


def test_vibration_only_is_never_subsidence() -> None:
    """ hard rule: vibration alone ≠ subsidence."""
    res = _gen(Scenario.VIBRATION_ONLY)
    assert res.risk_label == RISK_NORMAL, "vibration-only must NEVER be SUBSIDENCE/WARNING"
    assert res.anomaly_flag == ANOMALY_FLAG_NON_SUBSIDENCE
    assert res.progression_label == "STABLE"
    # and physically: no ground subsidence
    assert float(np.nanmax(res.subsidence_mm)) == 0.0


def test_single_node_disturbance_is_local_flag_not_subsidence() -> None:
    res = _gen(Scenario.SINGLE_NODE_DISTURBANCE)
    assert res.risk_label == RISK_NORMAL
    assert res.anomaly_flag == ANOMALY_FLAG_LOCAL
    assert res.anomaly_label.sum() > 0, "local disturbance must raise anomaly_label in its window"


def test_fault_scenarios_tag_sensor_fault() -> None:
    for s in (Scenario.SENSOR_BIAS, Scenario.SENSOR_STUCK, Scenario.SENSOR_DROPOUT, Scenario.SENSOR_SPIKE, Scenario.SENSOR_DRIFT):
        res = _gen(s, seed=2)
        tagged = (res.fault_label != "NONE").sum()
        assert tagged > 0, f"{s} must tag faulted samples"
        assert res.risk_label == RISK_NORMAL, "sensor faults are NOT ground risk"
        assert res.meta.get("fault_type"), f"{s} must record its fault type"


def test_packet_loss_flags_data_quality() -> None:
    res = _gen(Scenario.PACKET_LOSS, seed=3)
    flagged = (res.data_quality_label == DATA_QUALITY_FLAG).sum()
    assert flagged > 0, "packet loss must flag DATA_QUALITY"
    assert np.isnan(res.displacement_mm[res.data_quality_label == DATA_QUALITY_FLAG]).all(), (
        "lost packets never arrive as readings"
    )
    assert (res.fault_label == "NONE").all(), "comms loss is not a sensor fault"


def test_communication_failure_flag() -> None:
    res = _gen(Scenario.COMMUNICATION_FAILURE, seed=4)
    flagged = (res.data_quality_label == COMM_FAILURE_FLAG).sum()
    assert flagged > 0, "outage must flag COMMUNICATION_FAILURE"
    assert res.data_quality_label[res.data_quality_label != ""].size == flagged


def test_rapid_subsidence_produces_largest_signal() -> None:
    r = _gen(Scenario.RAPID_SUBSIDENCE, seed=5)
    s = _gen(Scenario.SLOW_SUBSIDENCE, seed=5)
    assert np.nanmax(r.subsidence_mm) > np.nanmax(s.subsidence_mm)


def test_accelerating_profile_steepens() -> None:
    res = _gen(Scenario.ACCELERATING_SUBSIDENCE, seed=6)
    w = res.subsidence_mm
    first_half_slope = (w[len(w) // 2] - w[0]) / (len(w) // 2)
    second_half_slope = (w[-1] - w[len(w) // 2]) / (len(w) // 2)
    assert second_half_slope > first_half_slope, "accelerating profile must steepen"


def test_multiple_zones_zone_dependent_risk() -> None:
    from src.simulator.scenarios import generate_scenario

    cfg_zone_offset = 130.0
    rng = np.random.default_rng(7)
    main = generate_scenario(Scenario.MULTIPLE_ZONES, 0.0, 0.0, "N01", rng)
    rng = np.random.default_rng(7)
    secondary = generate_scenario(Scenario.MULTIPLE_ZONES, cfg_zone_offset, 0.0, "N02", rng)
    assert main.risk_label == RISK_CRITICAL, "node under the main zone must be CRITICAL"
    assert secondary.risk_label == RISK_WARNING, "node under the secondary zone is WARNING"


def test_stable_ground_zero_subsidence() -> None:
    res = _gen(Scenario.STABLE_GROUND, seed=8)
    assert float(np.nanmax(res.subsidence_mm)) == 0.0
    assert res.risk_label == RISK_NORMAL
    assert (res.anomaly_label == 0).all(), "stable ground has no anomalies"


def test_tilt_present_and_finite_outside_outage() -> None:
    res = _gen(Scenario.RAPID_SUBSIDENCE, seed=9)
    assert np.isfinite(res.tilt_x_deg).all()
    assert np.abs(res.tilt_x_deg).max() > 0.0


def test_channels_shapes_consistent() -> None:
    res = _gen(Scenario.SLOW_SUBSIDENCE, seed=10)
    n = res.n_steps
    for arr in (
        res.subsidence_mm, res.tilt_x_deg, res.tilt_y_deg, res.displacement_mm,
        res.strain, res.vibration_rms, res.vibration_peak, res.vibration_crest,
        res.battery_v, res.rssi_dbm, res.snr_db, res.anomaly_label,
        res.fault_label, res.data_quality_label,
    ):
        assert arr.shape == (n,)
