"""Tilt channels derived from the deformation field — PRD §10 (T-013).

    tilt_x ≈ ∂W/∂x ,  tilt_y ≈ ∂W/∂y

Tilt is NEVER independently sampled: it is the analytic gradient of the same
W(x, y, t) that drives every other channel, plus sensor noise. Units: the
field is in mm and coordinates in metres, so ∂W/∂x is mm/m; tilt in degrees
is degrees(∂W/∂x / 1000) for the small-angle regime of a subsidence bowl.
Noise std comes from configs/physics.yaml (noise.tilt_noise_std_deg).
"""

from __future__ import annotations

import numpy as np

from src.config import physics_config
from src.simulator.deformation_field import DeformationField


def _noise_std_deg(noise_cfg: dict | None = None) -> float:
    cfg = noise_cfg or physics_config()["noise"]
    return float(cfg["tilt_noise_std_deg"])


def generate_tilt(
    field: DeformationField,
    x: np.ndarray,
    y: np.ndarray,
    t: float | np.ndarray,
    rng: np.random.Generator,
    noise_std_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (tilt_x, tilt_y) in degrees = analytic gradient of W + Gaussian noise."""
    std = noise_std_deg if noise_std_deg is not None else _noise_std_deg()
    gx_mm_per_m = np.asarray(field.dW_dx(x, y, t), dtype=float)
    gy_mm_per_m = np.asarray(field.dW_dy(x, y, t), dtype=float)
    tilt_x = np.degrees(gx_mm_per_m / 1000.0) + rng.normal(0.0, std, size=gx_mm_per_m.shape)
    tilt_y = np.degrees(gy_mm_per_m / 1000.0) + rng.normal(0.0, std, size=gy_mm_per_m.shape)
    return tilt_x, tilt_y


def tilt_magnitude(tilt_x: np.ndarray, tilt_y: np.ndarray) -> np.ndarray:
    """§11 field: tilt_magnitude = sqrt(tilt_x² + tilt_y²)."""
    return np.sqrt(np.asarray(tilt_x) ** 2 + np.asarray(tilt_y) ** 2)


def finite_difference_gradient(
    field: DeformationField,
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    h: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical ∂W/∂x, ∂W/∂y (mm/m) for validation against the analytic form."""
    gx = (field(x + h, y, t) - field(x - h, y, t)) / (2.0 * h)
    gy = (field(x, y + h, t) - field(x, y - h, t)) / (2.0 * h)
    return np.asarray(gx), np.asarray(gy)
