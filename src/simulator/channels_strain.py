"""Inter-node displacement and strain —.

 strain ≈ Δd_ij / d_ij (from inter-node distance change d_ij → d'_ij)

Kinematic model: a subsidence bowl pulls surface points horizontally toward
the subsidence centre (convergence). Each node's ground position moves by

 u(p, t) = -factor · W(x, y, t) · (p − p0) / σ [metres]

with `factor` from configs/physics.yaml (strain.horizontal_displacement_factor).
Inter-node distance d_ij is recomputed from the displaced positions and strain
is the fractional change. When W_max == 0 (t = 0, no extraction) the displaced
positions equal the originals, so strain is exactly 0 everywhere.
"""

from __future__ import annotations

import numpy as np

from src.config import physics_config
from src.simulator.deformation_field import DeformationField


def _factor(strain_cfg: dict | None = None) -> float:
    cfg = strain_cfg or physics_config()["strain"]
    return float(cfg["horizontal_displacement_factor"])


def horizontal_displacement(
    field: DeformationField,
    x: np.ndarray,
    y: np.ndarray,
    t: float | np.ndarray,
    factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convergence displacement (u_x, u_y) in metres at node positions."""
    f = _factor() if factor is None else factor
    w_mm = np.asarray(field(x, y, t), dtype=float)
    scale = f * (w_mm / 1000.0) / field.params.sigma  # metres per metre of offset
    u_x = -scale * (np.asarray(x, dtype=float) - field.params.panel_center_x)
    u_y = -scale * (np.asarray(y, dtype=float) - field.params.panel_center_y)
    return u_x, u_y


def displaced_positions(
    field: DeformationField,
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Node ground positions after convergence at time t."""
    u_x, u_y = horizontal_displacement(field, x, y, t, factor)
    return np.asarray(x, dtype=float) + u_x, np.asarray(y, dtype=float) + u_y


def pairwise_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d_ij between all node pairs from original positions."""
    pts = np.stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)], axis=1)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def strain_matrix(
    field: DeformationField,
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    factor: float | None = None,
) -> np.ndarray:
    """strain_ij = (d'_ij − d_ij) / d_ij from the displaced mesh (NaN on the diagonal)."""
    d0 = pairwise_distance_matrix(x, y)
    x1, y1 = displaced_positions(field, x, y, t, factor)
    d1 = pairwise_distance_matrix(x1, y1)
    with np.errstate(divide="ignore", invalid="ignore"):
        strain = (d1 - d0) / d0
    np.fill_diagonal(strain, np.nan)
    return strain


def edge_strain(
    field: DeformationField,
    x_i: float,
    y_i: float,
    x_j: float,
    y_j: float,
    t: float,
    factor: float | None = None,
) -> float:
    """strain for a single inter-node edge (the per-reading form used in the dataset)."""
    x = np.array([x_i, x_j], dtype=float)
    y = np.array([y_i, y_j], dtype=float)
    s = strain_matrix(field, x, y, t, factor)
    return float(s[0, 1])


def edge_strain_series(
    field: DeformationField,
    x_i: float,
    y_i: float,
    x_j: float,
    y_j: float,
    t_days: np.ndarray,
    factor: float | None = None,
) -> np.ndarray:
    """Vectorised edge strain over a time axis — the dataset-builder form.

 Exactly:func:`edge_strain` evaluated per t, computed once for the whole
 series so generation of 1.4M rows stays fast.
 """
    t = np.asarray(t_days, dtype=float)
    ones_i = np.full(t.size, float(x_i))
    onesj = np.full(t.size, float(x_j))
    uxi, uyi = horizontal_displacement(field, ones_i, np.full(t.size, float(y_i)), t, factor)
    uxj, uyj = horizontal_displacement(field, onesj, np.full(t.size, float(y_j)), t, factor)
    d0 = float(np.hypot(x_j - x_i, y_j - y_i))
    if d0 == 0:
        raise ValueError("co-located nodes have no inter-node distance")
    d1 = np.hypot((x_j + uxj) - (x_i + uxi), (y_j + uyj) - (y_i + uyi))
    return (d1 - d0) / d0
