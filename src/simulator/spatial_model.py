"""Spatial deformation kernel — PRD §10 (T-010).

    W(x, y) = exp( -[(x - x0)^2 + (y - y0)^2] / (2 * sigma^2) )

A Gaussian influence-kernel approximation over the virtual mining panel,
consistent with probability-integral / influence-function concepts used in
the mining subsidence literature. sigma is derived from the configured
influence radius (configs/physics.yaml), never hard-coded.
"""

from __future__ import annotations

import math

import numpy as np

from src.config import physics_config


def sigma_from_config(physics_cfg: dict | None = None) -> float:
    """Kernel width sigma = influence_radius / sigma_divisor (§10, config-driven)."""
    cfg = physics_cfg or physics_config()
    influence_radius = float(cfg["physics"]["influence_radius"])
    divisor = float(cfg["kernel"]["sigma_divisor"])
    if divisor <= 0:
        raise ValueError("kernel.sigma_divisor must be > 0")
    return influence_radius / divisor


def spatial_kernel(
    x: float | np.ndarray,
    y: float | np.ndarray,
    center_x: float,
    center_y: float,
    sigma: float,
) -> float | np.ndarray:
    """Gaussian influence kernel; equals 1.0 at the subsidence center."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    dx = np.asarray(x, dtype=float) - center_x
    dy = np.asarray(y, dtype=float) - center_y
    return np.exp(-((dx * dx + dy * dy) / (2.0 * sigma * sigma)))


def analytic_kernel_check(center_x: float, center_y: float, sigma: float, n_points: int = 20) -> bool:
    """Self-check against math.exp at >=20 sampled points (used by tests/docs)."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(center_x - 3 * sigma, center_x + 3 * sigma, size=n_points)
    ys = rng.uniform(center_y - 3 * sigma, center_y + 3 * sigma, size=n_points)
    got = spatial_kernel(xs, ys, center_x, center_y, sigma)
    for xv, yv, gv in zip(xs, ys, got):
        expected = math.exp(-((xv - center_x) ** 2 + (yv - center_y) ** 2) / (2 * sigma**2))
        if abs(gv - expected) > 1e-9:
            return False
    return True
