"""T-015 acceptance tests — vibration components toggleable; vibration-only ≠ subsidence."""

from __future__ import annotations

import numpy as np

from src.simulator.channels_vibration import (
    VibrationComponents,
    default_components,
    generate_vibration,
    vibration_only_is_non_subsidence,
)


def test_all_components_default_off() -> None:
    c = VibrationComponents()
    assert not c.normal_noise
    assert not c.vehicle_disturbance
    assert not c.localized_transient
    assert not c.high_frequency_burst


def test_noise_component_rms_tracks_config() -> None:
    comps = VibrationComponents(normal_noise=True, noise_std=0.02)
    rng = np.random.default_rng(42)
    results = [generate_vibration(float(t), comps, rng) for t in range(200)]
    rms = np.array([r.rms for r in results])
    assert abs(rms.mean() - 0.02) < 0.005, f"RMS ≈ configured noise std, got {rms.mean():.4f}"


def test_each_component_independently_toggleable() -> None:
    rng = np.random.default_rng(7)
    base = VibrationComponents(normal_noise=True, noise_std=0.01)
    only_burst = VibrationComponents(high_frequency_burst=True, burst_rate_per_day=10_000.0)
    r_base = generate_vibration(1.0, base, rng)
    r_burst = generate_vibration(1.0, only_burst, rng)
    assert r_burst.peak > 10 * r_base.peak, "burst component must dominate when enabled alone"
    assert "high_frequency_burst" in r_burst.components_active


def test_vehicle_and_transient_flags_surface_in_result() -> None:
    rng = np.random.default_rng(11)
    comps = VibrationComponents(
        vehicle_disturbance=True, vehicle_rate_per_day=10_000.0,
        localized_transient=True, transient_rate_per_day=10_000.0,
    )
    r = generate_vibration(2.0, comps, rng)
    assert "vehicle_disturbance" in r.components_active
    assert "localized_transient" in r.components_active


def test_crest_factor_positive() -> None:
    comps = VibrationComponents(normal_noise=True)
    r = generate_vibration(0.0, comps, np.random.default_rng(0))
    assert r.crest_factor >= 1.0


def test_summary_never_raw_samples() -> None:
    """§8.3: the returned object carries summary statistics only — no raw array."""
    comps = VibrationComponents(normal_noise=True)
    r = generate_vibration(0.0, comps, np.random.default_rng(0))
    assert not hasattr(r, "samples"), "raw high-rate samples must never be emitted"
    assert r.rms >= 0 and r.peak >= 0


def test_defaults_from_config() -> None:
    comps = default_components().resolved()
    from src.config import physics_config

    assert comps.noise_std == float(physics_config()["noise"]["vibration_noise_std"])
    cfg = physics_config()["vibration"]
    assert comps.vehicle_amplitude == float(cfg["vehicle_amplitude"])
    assert comps.burst_amplitude == float(cfg["burst_amplitude"])


def test_vibration_only_label_rule() -> None:
    """§10: vibration alone ≠ subsidence — the encoding the scenario engine uses."""
    assert vibration_only_is_non_subsidence()
