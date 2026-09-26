"""Knothe-style temporal subsidence growth — PRD §10 (T-011).

    W(t) = W_max * (1 - e^(-c * t))

W(0) == 0, monotonically increasing, asymptotically approaching W_max.
``c`` is the time coefficient (per day) and ``W_max`` the maximum subsidence
(mm) — both from configs/physics.yaml, never hard-coded.
"""

from __future__ import annotations

import numpy as np

from src.config import physics_parameters


def w_max_from_config(params: dict | None = None) -> float:
    """Maximum subsidence W_max in mm (config-driven)."""
    params = params or physics_parameters()
    return float(params["maximum_subsidence"])


def c_from_config(params: dict | None = None) -> float:
    """Knothe time coefficient c in 1/day (config-driven)."""
    params = params or physics_parameters()
    return float(params["time_coefficient"])


def subsidence_growth(
    t: float | np.ndarray,
    w_max: float,
    c: float,
) -> float | np.ndarray:
    """W(t) = W_max * (1 - e^(-c t)); t in days, result in mm."""
    if c <= 0:
        raise ValueError("time coefficient c must be > 0")
    if w_max < 0:
        raise ValueError("w_max must be >= 0")
    return np.asarray(w_max * (1.0 - np.exp(-c * np.asarray(t, dtype=float))))


def growth_velocity(t: float | np.ndarray, w_max: float, c: float) -> float | np.ndarray:
    """dW/dt = W_max * c * e^(-c t) — mm/day (used for rate labels and event metadata)."""
    return np.asarray(w_max * c * np.exp(-c * np.asarray(t, dtype=float)))


def time_to_fraction(fraction: float, c: float) -> float:
    """Days until W(t) reaches `fraction` of W_max (t = -ln(1-f)/c)."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    return -np.log(1.0 - fraction) / c
