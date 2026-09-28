"""Scenario taxonomy engine — PRD §10 (T-018).

Generates the full §10 event/scenario taxonomy. Each scenario produces, per
virtual node, a raw channel table plus per-row ``progression_label``,
``fault_label``, ``risk_label``, ``anomaly_label`` and data-quality flags.
All numeric behaviour is config-driven (configs/physics.yaml → ``scenarios``).

Label vocabulary mapping (Gap G-1, resolved as the scenario→schema mapping;
see PROGRESS_LOG 2026-09-26):

  Scenario                progression_label    risk_label
  ----------------------  -------------------  -------------------
  stable_ground           STABLE               NORMAL (GREEN slot)
  slow_drift_temperature  STABLE               NORMAL
  sensor_* fault modes     STABLE               NORMAL (fault_label set)
  packet_loss             STABLE               NORMAL (DATA_QUALITY flag)
  single_node_disturbance STABLE               NORMAL (LOCAL_ANOMALY flag)
  vibration_only          STABLE               NORMAL (NON_SUBSIDENCE flag)
  slow_subsidence         SLOW                 WARNING
  accelerating_subsidence ACCELERATING         WARNING
  rapid_subsidence        RAPID                CRITICAL
  irregular_subsidence    ACCELERATING         WARNING
  multiple_zones          SLOW                 WARNING / CRITICAL by zone
  communication_failure   STABLE               NORMAL (COMM_FAILURE flag)

``risk_label`` uses the §12 3-class MVP vocabulary (NORMAL / WARNING /
CRITICAL; NORMAL fills the GREEN slot — Gaps G-2/G-3 resolved at Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.config import physics_config
from src.simulator.channels_strain import edge_strain_series
from src.simulator.channels_vibration import VibrationComponents, summarized_vibration_batch
from src.simulator.comms_degradation import link_outage, random_packet_loss
from src.simulator.deformation_field import DeformationField, FieldParams
from src.simulator.faults import FaultType, inject_fault
from src.simulator.temporal_model import subsidence_growth

PROGRESSION_STABLE = "STABLE"
PROGRESSION_SLOW = "SLOW"
PROGRESSION_ACCELERATING = "ACCELERATING"
PROGRESSION_RAPID = "RAPID"

RISK_NORMAL = "NORMAL"      # MVP 3-class (GREEN slot; G-2/G-3 resolved at Phase 3)
RISK_WARNING = "WARNING"
RISK_CRITICAL = "CRITICAL"

FAULT_NONE = "NONE"

ANOMALY_FLAG_LOCAL = "LOCAL_ANOMALY"
ANOMALY_FLAG_NON_SUBSIDENCE = "NON_SUBSIDENCE"

DATA_QUALITY_FLAG = "DATA_QUALITY"
COMM_FAILURE_FLAG = "COMMUNICATION_FAILURE"


class Scenario(str, Enum):
    """Generator scenarios. The five sensor_* modes together form the single
    §10 taxonomy row 'sensor bias / stuck / dropout / random spike / slow drift'."""

    STABLE_GROUND = "stable_ground"
    SLOW_DRIFT_TEMPERATURE = "slow_drift_temperature"
    SENSOR_BIAS = "sensor_bias"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_DROPOUT = "sensor_dropout"
    SENSOR_SPIKE = "sensor_spike"
    SENSOR_DRIFT = "sensor_drift"
    PACKET_LOSS = "packet_loss"
    SINGLE_NODE_DISTURBANCE = "single_node_disturbance"
    VIBRATION_ONLY = "vibration_only"
    SLOW_SUBSIDENCE = "slow_subsidence"
    ACCELERATING_SUBSIDENCE = "accelerating_subsidence"
    RAPID_SUBSIDENCE = "rapid_subsidence"
    IRREGULAR_SUBSIDENCE = "irregular_subsidence"
    MULTIPLE_ZONES = "multiple_zones"
    COMMUNICATION_FAILURE = "communication_failure"


# §10 taxonomy table — the 12 scenario rows, verbatim
TAXONOMY_ROWS: list[str] = [
    "stable ground",
    "slow drift / temperature drift",
    "sensor bias / stuck / dropout / random spike / slow drift (sensor)",
    "packet loss",
    "single-node disturbance",
    "vibration-only event",
    "slow subsidence",
    "accelerating subsidence",
    "rapid subsidence",
    "irregular subsidence",
    "multiple deformation zones",
    "communication failure",
]

_SCENARIO_TO_FAULT: dict[Scenario, FaultType] = {
    Scenario.SENSOR_BIAS: FaultType.BIAS,
    Scenario.SENSOR_STUCK: FaultType.STUCK,
    Scenario.SENSOR_DROPOUT: FaultType.DROPOUT,
    Scenario.SENSOR_SPIKE: FaultType.SPIKE,
    Scenario.SENSOR_DRIFT: FaultType.DRIFT,
}

_SUBSIDENCE_SCENARIOS: dict[Scenario, tuple[str, str]] = {
    Scenario.SLOW_SUBSIDENCE: (PROGRESSION_SLOW, RISK_WARNING),
    Scenario.ACCELERATING_SUBSIDENCE: (PROGRESSION_ACCELERATING, RISK_WARNING),
    Scenario.RAPID_SUBSIDENCE: (PROGRESSION_RAPID, RISK_CRITICAL),
    Scenario.IRREGULAR_SUBSIDENCE: (PROGRESSION_ACCELERATING, RISK_WARNING),
    Scenario.MULTIPLE_ZONES: (PROGRESSION_SLOW, RISK_WARNING),  # per-zone override below
}

FAULT_SCENARIOS = frozenset(_SCENARIO_TO_FAULT)


def scenario_risk_mapping() -> dict[str, str]:
    """Scenario name → default risk_label, per the §10 taxonomy mapping."""
    mapping: dict[str, str] = {}
    for s in Scenario:
        mapping[s.value] = _SUBSIDENCE_SCENARIOS.get(s, (PROGRESSION_STABLE, RISK_NORMAL))[1]
    return mapping


def _scn_cfg() -> dict:
    return physics_config()["scenarios"]


@dataclass
class ScenarioResult:
    """One node's raw channels and labels under one scenario."""

    scenario: Scenario
    node_id: str
    n_steps: int
    timestamps_hours: np.ndarray
    subsidence_mm: np.ndarray        # true ground subsidence (physical truth)
    tilt_x_deg: np.ndarray
    tilt_y_deg: np.ndarray
    displacement_mm: np.ndarray      # sensor-observed vertical displacement
    strain: np.ndarray               # inter-node strain to a reference node
    vibration_rms: np.ndarray
    vibration_peak: np.ndarray
    vibration_crest: np.ndarray
    battery_v: np.ndarray
    rssi_dbm: np.ndarray
    snr_db: np.ndarray
    progression_label: str
    risk_label: str
    anomaly_label: np.ndarray        # 0 normal / 1 abnormal (§12)
    fault_label: np.ndarray          # NONE / BIAS / STUCK / DROPOUT / SPIKE / DRIFT
    anomaly_flag: str                # '' | LOCAL_ANOMALY | NON_SUBSIDENCE
    data_quality_label: np.ndarray   # '' | DATA_QUALITY | COMMUNICATION_FAILURE
    meta: dict = field(default_factory=dict)


def _temporal_profile(scenario: Scenario, t_days: np.ndarray, base_c: float) -> np.ndarray:
    """W(t) growth curve (mm) driving the scenario's physical deformation."""
    cfg = _scn_cfg()
    amp = DeformationField(FieldParams.from_config()).w_max
    t_max = float(t_days.max()) if t_days.size else 0.0

    if scenario is Scenario.SLOW_SUBSIDENCE:
        return np.asarray(subsidence_growth(t_days, amp, base_c * float(cfg["slow_c_multiplier"])))
    if scenario is Scenario.RAPID_SUBSIDENCE:
        return np.asarray(subsidence_growth(t_days, amp, base_c * float(cfg["rapid_c_multiplier"])))
    if scenario is Scenario.ACCELERATING_SUBSIDENCE:
        beta = float(cfg["accelerating_beta"])
        if t_max <= 0:
            return np.zeros_like(t_days, dtype=float)
        tau = t_days / t_max
        return amp * tau**beta
    if scenario is Scenario.IRREGULAR_SUBSIDENCE:
        base = np.asarray(subsidence_growth(t_days, amp, base_c * float(cfg["slow_c_multiplier"])))
        if t_max <= 0:
            return base
        osc = float(cfg["irregular_osc_amplitude_fraction"]) * amp * np.sin(
            2.0 * np.pi * float(cfg["irregular_osc_periods"]) * t_days / t_max
        )
        return base + osc
    if scenario is Scenario.MULTIPLE_ZONES:
        return np.asarray(subsidence_growth(t_days, amp, base_c))
    return np.zeros_like(t_days, dtype=float)


def _spatial_weight(scenario: Scenario, x: float, y: float, fld: DeformationField) -> float:
    """Per-node spatial amplitude weight in [0, ~1] applied to the profile."""
    cfg = _scn_cfg()
    p = fld.params
    if scenario is Scenario.MULTIPLE_ZONES:
        zx = p.panel_center_x + float(cfg["second_zone_offset_m"])
        zone_a = float(np.exp(-((x - p.panel_center_x) ** 2 + (y - p.panel_center_y) ** 2) / (2 * p.sigma**2)))
        zone_b = float(cfg["second_zone_amplitude_fraction"]) * float(
            np.exp(-((x - zx) ** 2 + (y - p.panel_center_y) ** 2) / (2 * p.sigma**2))
        )
        return zone_a + zone_b
    return float(
        np.exp(-((x - p.panel_center_x) ** 2 + (y - p.panel_center_y) ** 2) / (2 * p.sigma**2))
    )


def generate_scenario(
    scenario: Scenario,
    node_x: float,
    node_y: float,
    node_id: str,
    rng: np.random.Generator,
    reference_node: tuple[float, float] = (0.0, 0.0),
    fault_type: FaultType | None = None,
) -> ScenarioResult:
    """Generate one node's raw channels and labels under one scenario.

    A pure function of (scenario, node position, rng state) — reproducibility
    (T-020) is guaranteed by the caller seeding ``rng``.
    """
    cfg = _scn_cfg()
    fld = DeformationField(FieldParams.from_config())

    # Strain reference must not be co-located with the node (degenerate edge).
    meta: dict = {}
    if np.hypot(node_x - reference_node[0], node_y - reference_node[1]) < 1e-9:
        reference_node = (reference_node[0] + 1.0, reference_node[1])
        meta["strain_reference_shifted"] = True

    steps = int(cfg["steps_per_day"] * cfg["duration_days"])
    t_hours = np.arange(steps, dtype=float) * (24.0 / float(cfg["steps_per_day"]))
    t_days = t_hours / 24.0

    profile = _temporal_profile(scenario, t_days, fld.params.time_coefficient)
    weight = _spatial_weight(scenario, node_x, node_y, fld)
    subsidence = profile * weight

    # tilt: analytic gradient of the same field (T-013 coupling)
    gx = np.asarray(fld.dW_dx(np.full(steps, node_x), np.full(steps, node_y), t_days))
    gy = np.asarray(fld.dW_dy(np.full(steps, node_x), np.full(steps, node_y), t_days))
    tilt_std = float(physics_config()["noise"]["tilt_noise_std_deg"])
    tilt_x = np.degrees(gx / 1000.0) + rng.normal(0.0, tilt_std, steps)
    tilt_y = np.degrees(gy / 1000.0) + rng.normal(0.0, tilt_std, steps)

    disp_noise = float(physics_config()["noise"]["displacement_noise_std_mm"])
    displacement = subsidence + rng.normal(0.0, disp_noise, steps)

    strain = edge_strain_series(fld, node_x, node_y, reference_node[0], reference_node[1], t_days)

    # vibration: supporting-only channel (T-015)
    vib_comps = VibrationComponents(normal_noise=True)
    vib = summarized_vibration_batch(vib_comps, rng, steps)
    vibration_rms = np.asarray(vib.rms)
    vibration_peak = np.asarray(vib.peak)
    vibration_crest = np.asarray(vib.crest_factor)

    tcfg = physics_config()["telemetry"]
    battery = np.linspace(float(tcfg["battery_start_v"]), float(tcfg["battery_end_v"]), steps) + rng.normal(
        0.0, 0.01, steps
    )
    rssi = rng.normal(float(tcfg["rssi_mean_dbm"]), float(tcfg["rssi_std_dbm"]), steps)
    snr = rng.normal(float(tcfg["snr_mean_db"]), float(tcfg["snr_std_db"]), steps)

    progression, risk = _SUBSIDENCE_SCENARIOS.get(scenario, (PROGRESSION_STABLE, RISK_NORMAL))
    anomaly_flag = ""

    fault_arr = np.full(steps, FAULT_NONE, dtype=object)
    anomaly = np.zeros(steps, dtype=int)
    dq = np.empty(steps, dtype=object)
    dq[:] = ""

    # per-scenario behaviour
    if scenario is Scenario.MULTIPLE_ZONES:
        # zone-dependent risk: nodes closer to the secondary zone are weaker (WARNING);
        # nodes under the main zone are CRITICAL (max roll-up at panel level)
        p = fld.params
        zx = p.panel_center_x + float(cfg["second_zone_offset_m"])
        dist_a = float(np.hypot(node_x - p.panel_center_x, node_y - p.panel_center_y))
        dist_b = float(np.hypot(node_x - zx, node_y - p.panel_center_y))
        risk = RISK_WARNING if dist_b < dist_a else RISK_CRITICAL
        meta["zone"] = "secondary" if dist_b < dist_a else "primary"

    if scenario in FAULT_SCENARIOS:
        ft = fault_type or _SCENARIO_TO_FAULT[scenario]
        onset = int(steps * float(cfg["fault_injection_rate"]))
        fr = inject_fault(displacement, ft, rng, onset)
        displacement = fr.values
        fault_arr[fr.fault_mask] = ft.value
        anomaly[fr.fault_mask] = 1
        meta["fault_onset_step"] = onset
        meta["fault_type"] = ft.value

    if scenario is Scenario.PACKET_LOSS:
        _keep, lost = random_packet_loss(steps, float(cfg["packet_loss_rate"]), rng)
        dq[lost] = DATA_QUALITY_FLAG
        displacement[lost] = np.nan
        meta["packets_lost"] = int(lost.sum())

    if scenario is Scenario.COMMUNICATION_FAILURE:
        start = int(steps * (1.0 - float(cfg["outage_fraction"])))
        _keep, lost = link_outage(steps, start, steps - start)
        dq[lost] = COMM_FAILURE_FLAG
        displacement[lost] = np.nan
        tilt_x[lost] = np.nan
        tilt_y[lost] = np.nan
        meta["outage_start"] = start

    if scenario is Scenario.SINGLE_NODE_DISTURBANCE:
        anomaly_flag = ANOMALY_FLAG_LOCAL
        window = int(steps * float(cfg["local_anomaly_window_fraction"]))
        anomaly[10 : 10 + window] = 1

    if scenario is Scenario.VIBRATION_ONLY:
        anomaly_flag = ANOMALY_FLAG_NON_SUBSIDENCE
        anomaly[0] = 1  # the event step
        meta["vibration_event"] = "high_frequency_burst"

    if scenario is Scenario.SLOW_DRIFT_TEMPERATURE:
        # instrument drift from diurnal temperature — ground itself STABLE
        amp_mm = float(cfg["temperature_drift_mm"])
        period = float(cfg["temperature_period_days"])
        displacement += amp_mm * np.sin(2.0 * np.pi * t_days / period)
        meta["temperature_drift_amplitude_deg"] = float(cfg["temperature_drift_amplitude_deg"])

    threshold_mm = float(cfg["anomaly_subsidence_threshold_mm"])
    deformation_mask = subsidence > threshold_mm
    anomaly[deformation_mask] = 1

    return ScenarioResult(
        scenario=scenario,
        node_id=node_id,
        n_steps=steps,
        timestamps_hours=t_hours,
        subsidence_mm=subsidence,
        tilt_x_deg=tilt_x,
        tilt_y_deg=tilt_y,
        displacement_mm=displacement,
        strain=strain,
        vibration_rms=vibration_rms,
        vibration_peak=vibration_peak,
        vibration_crest=vibration_crest,
        battery_v=battery,
        rssi_dbm=rssi,
        snr_db=snr,
        progression_label=progression,
        risk_label=risk,
        anomaly_label=anomaly,
        fault_label=fault_arr,
        anomaly_flag=anomaly_flag,
        data_quality_label=dq,
        meta=meta,
    )
