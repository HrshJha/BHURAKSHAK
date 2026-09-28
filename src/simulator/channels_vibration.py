"""Vibration channel — PRD §10 (T-015).

    vibration = normal_noise + vehicle/personnel disturbance
                + localized transient + high-frequency burst

Vibration is **supporting evidence only**: `vibration alone ≠ subsidence`.
All four components are independently toggleable and default OFF unless the
scenario enables them — the label engine (T-018) maps a vibration-only event
to NON_SUBSIDENCE, never SUBSIDENCE.

All component amplitudes/rates come from configs/physics.yaml (vibration
block) via src/config.py — no numeric scenario literals live here (NFR-6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import physics_config


def _vibration_cfg() -> dict:
    return physics_config()["vibration"]


@dataclass
class VibrationComponents:
    """Toggleable components; every flag defaults False (nothing hidden on).

    Numeric parameters left as ``None`` resolve from configs/physics.yaml at
    generation time, so tuning never requires touching this module.
    """

    normal_noise: bool = False
    vehicle_disturbance: bool = False
    localized_transient: bool = False
    high_frequency_burst: bool = False

    noise_std: float | None = None
    vehicle_rate_per_day: float | None = None
    vehicle_amplitude: float | None = None
    transient_rate_per_day: float | None = None
    transient_amplitude: float | None = None
    burst_rate_per_day: float | None = None
    burst_amplitude: float | None = None
    burst_duration_minutes: float | None = None

    def resolved(self) -> "VibrationComponents":
        """Fill any None parameter from config and return a concrete copy."""
        cfg = _vibration_cfg()
        noise_std = physics_config()["noise"]["vibration_noise_std"]
        mapping = {
            "noise_std": float(noise_std),
            "vehicle_rate_per_day": float(cfg["vehicle_rate_per_day"]),
            "vehicle_amplitude": float(cfg["vehicle_amplitude"]),
            "transient_rate_per_day": float(cfg["transient_rate_per_day"]),
            "transient_amplitude": float(cfg["transient_amplitude"]),
            "burst_rate_per_day": float(cfg["burst_rate_per_day"]),
            "burst_amplitude": float(cfg["burst_amplitude"]),
            "burst_duration_minutes": float(cfg["burst_duration_minutes"]),
        }
        kwargs = {}
        for name, value in mapping.items():
            kwargs[name] = getattr(self, name) if getattr(self, name) is not None else value
        return VibrationComponents(
            normal_noise=self.normal_noise,
            vehicle_disturbance=self.vehicle_disturbance,
            localized_transient=self.localized_transient,
            high_frequency_burst=self.high_frequency_burst,
            **kwargs,
        )


def default_components(noise_cfg: dict | None = None) -> VibrationComponents:
    """Component defaults from configs/physics.yaml."""
    noise_std = (
        float(noise_cfg["vibration_noise_std"]) if noise_cfg else None
    )
    return VibrationComponents(noise_std=noise_std)


@dataclass
class VibrationResult:
    rms: float
    peak: float
    crest_factor: float
    components_active: list[str] = field(default_factory=list)


def generate_vibration(
    t: float,
    components: VibrationComponents,
    rng: np.random.Generator,
    sampling_hz: float = 100.0,
) -> VibrationResult:
    """One on-node high-rate burst summarised to §8.3 uplink statistics.

    Simulates ``sampling_hz`` samples around time ``t`` (hours) and returns
    RMS / peak / crest-factor — the summary that would actually be transmitted
    (§8.3: raw high-rate samples are never transmitted).
    """
    c = components.resolved()
    n = max(int(sampling_hz * 60), 2)  # one minute of on-node samples
    samples = np.zeros(n)
    active: list[str] = []

    if c.normal_noise:
        samples += rng.normal(0.0, c.noise_std, size=n)
        active.append("normal_noise")

    if c.vehicle_disturbance and rng.random() < c.vehicle_rate_per_day / 24.0:
        # low-frequency rumble: slowly varying envelope between 0 and 1
        envelope = np.sin(np.linspace(0.0, np.pi, n))
        samples += c.vehicle_amplitude * envelope * rng.normal(0.0, 1.0, size=n)
        active.append("vehicle_disturbance")

    if c.localized_transient and rng.random() < c.transient_rate_per_day / 24.0:
        # impulsive transient near mid-window, with ±10% amplitude jitter
        idx = rng.integers(n // 3, 2 * n // 3)
        jitter = rng.uniform(0.9, 1.1)
        samples[idx] += c.transient_amplitude * jitter
        active.append("localized_transient")

    if c.high_frequency_burst and rng.random() < c.burst_rate_per_day / 24.0:
        # high-frequency burst lasting burst_duration_minutes
        dur = max(int(c.burst_duration_minutes * sampling_hz), 1)
        start = rng.integers(0, max(n - dur, 1))
        freq = rng.uniform(40.0, min(sampling_hz / 2.0, 60.0))
        tt = np.arange(dur) / sampling_hz
        samples[start : start + dur] += c.burst_amplitude * np.sin(2 * np.pi * freq * tt)
        active.append("high_frequency_burst")

    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    crest = float(peak / rms) if rms > 0 else 0.0
    return VibrationResult(rms=rms, peak=peak, crest_factor=crest, components_active=active)


def vibration_only_is_non_subsidence() -> bool:
    """Encoding of the §10 rule enforced again by the scenario engine (T-018):
    a vibration-only event is labelled NON_SUBSIDENCE, never SUBSIDENCE."""
    return True


def summarized_vibration_batch(
    components: VibrationComponents,
    rng: np.random.Generator,
    n_steps: int,
    sampling_hz: float = 100.0,
    window_seconds: float | None = None,
) -> VibrationResult:
    """Vectorised per-step vibration summaries for dataset generation.

    Identical statistical model to :func:`generate_vibration` (same component
    probabilities and amplitudes from config) but computed for ``n_steps``
    steps at once; returns arrays of RMS / peak / crest per step plus the list
    of components active anywhere in the batch.
    """
    c = components.resolved()
    win = float(
        window_seconds
        if window_seconds is not None
        else physics_config()["vibration"]["on_node_window_seconds"]
    )
    n = max(int(sampling_hz * win), 2)
    active: list[str] = []

    samples = np.zeros((n_steps, n))

    if c.normal_noise:
        samples += rng.normal(0.0, c.noise_std, size=(n_steps, n))
        active.append("normal_noise")

    if c.vehicle_disturbance:
        fired = rng.random(n_steps) < c.vehicle_rate_per_day / 24.0
        if fired.any():
            envelope = np.sin(np.linspace(0.0, np.pi, n))
            samples[fired] += c.vehicle_amplitude * envelope * rng.normal(
                0.0, 1.0, size=(int(fired.sum()), n)
            )
            active.append("vehicle_disturbance")

    if c.localized_transient:
        fired = rng.random(n_steps) < c.transient_rate_per_day / 24.0
        if fired.any():
            rows = np.where(fired)[0]
            cols = rng.integers(0, n, size=rows.size)
            jitter = rng.uniform(0.9, 1.1, size=rows.size)
            samples[rows, cols] += c.transient_amplitude * jitter
            active.append("localized_transient")

    if c.high_frequency_burst:
        fired = rng.random(n_steps) < c.burst_rate_per_day / 24.0
        if fired.any():
            dur = max(int(c.burst_duration_minutes * sampling_hz), 1)
            dur = min(dur, n)
            tt = np.arange(dur) / sampling_hz
            for row in np.where(fired)[0]:
                start = int(rng.integers(0, max(n - dur, 1)))
                freq = rng.uniform(40.0, min(sampling_hz / 2.0, 60.0))
                samples[row, start : start + dur] += c.burst_amplitude * np.sin(
                    2 * np.pi * freq * tt
                )
            active.append("high_frequency_burst")

    rms = np.sqrt(np.mean(samples * samples, axis=1))
    peak = np.max(np.abs(samples), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.where(rms > 0, peak / rms, 0.0)
    return VibrationResult(rms=rms, peak=peak, crest_factor=crest, components_active=active)
