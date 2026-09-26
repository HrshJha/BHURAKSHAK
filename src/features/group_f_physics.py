"""Feature Group F — Physics — PRD §13 Group F, FR-15 (T-041).

§13 Group F, exactly: ``expected_displacement, expected_tilt, physics_residual,
physics_residual_velocity``.

Sources:
- ``expected_displacement`` / ``expected_tilt``: the §10 model evaluated at
  the node's (x, y) at the window timestamp (via T-040's engine — the same
  model that generated the data);
- ``physics_residual``: observed − expected (T-040), computed on the window's
  mean observed displacement;
- ``physics_residual_velocity``: change of the residual between consecutive
  windows of the same (event, node) series, per hour — a growing residual is
  the classic "unexplained acceleration" signature (§21 motivation).

The engine is built once per call from configs/physics.yaml; the tilt feature
carries the expected tilt **magnitude** (single §13 column; the signed
components are recoverable via the physics engine).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.group_c_spatial import SpatialFeatureError
from src.physics.consistency import physics_engine

__all__ = ["GROUP_F_FEATURES", "emit_group_f"]

GROUP_F_FEATURES = (
    "expected_displacement",
    "expected_tilt",
    "physics_residual",
    "physics_residual_velocity",
)


def emit_group_f(
    windowed: pd.DataFrame,
    node_coords: pd.DataFrame,
    *,
    residual_velocity_interval_hours: float | None = None,
) -> pd.DataFrame:
    """Emit Group F features per window row.

    ``windowed`` must carry the window keys, ``x``/``y`` per node (or be
    joinable to ``node_coords``), ``displacement_mean``, ``tilt_x_mean`` and
    ``tilt_y_mean``. The residual velocity is computed per (event, node)
    series in ``window_index`` order.
    """
    keys = ["event_id", "node_id", "window_index", "window_timestamp", "window_start", "window_end"]
    missing_cols = [c for c in ("displacement_mean",) if c not in windowed.columns]
    if missing_cols:
        raise SpatialFeatureError(f"windowed frame missing Group F sources: {missing_cols}")

    coords = node_coords.set_index("node_id")
    engine = physics_engine()
    interval = (
        float(residual_velocity_interval_hours)
        if residual_velocity_interval_hours is not None
        else _grid_interval_hours() * 10.0  # default: one stride (10 steps)
    )

    frames: list[pd.DataFrame] = []
    for _keys, g in windowed.groupby(["event_id", "node_id"], sort=False):
        g = g.sort_values("window_index", kind="stable").reset_index(drop=True)
        x = coords.loc[g["node_id"], "x"].to_numpy(dtype=float)
        y = coords.loc[g["node_id"], "y"].to_numpy(dtype=float)
        t = g["window_timestamp"].to_numpy(dtype=float)

        exp_disp = engine.expected_displacement_mm(x, y, t)
        etx, ety = engine.expected_tilt(x, y, t)
        expected_tilt_mag = np.hypot(etx, ety)
        observed = g["displacement_mean"].to_numpy(dtype=float)
        residual = observed - exp_disp

        # residual velocity between consecutive windows (first window: 0)
        resid_vel = np.zeros(len(g))
        if len(g) > 1:
            resid_vel[1:] = np.diff(residual) / interval

        if "tilt_x_mean" in g.columns and "tilt_y_mean" in g.columns:
            obs_tilt_mag = np.hypot(
                g["tilt_x_mean"].to_numpy(dtype=float), g["tilt_y_mean"].to_numpy(dtype=float)
            )
            tilt_residual_mag = obs_tilt_mag - expected_tilt_mag
        else:
            tilt_residual_mag = np.zeros(len(g))

        block = g[keys].copy()
        block["expected_displacement"] = exp_disp
        block["expected_tilt"] = expected_tilt_mag
        block["physics_residual"] = residual
        block["physics_residual_velocity"] = resid_vel
        block["tilt_residual_unnamed"] = tilt_residual_mag  # auxiliary, not §13-named
        frames.append(block)

    out = pd.concat(frames, ignore_index=True)
    return out[["event_id", "node_id", "window_index", "window_timestamp", *GROUP_F_FEATURES]]


def _grid_interval_hours() -> float:
    from src.config import load_config

    return float(load_config("preprocessing")["resampling"]["grid_interval_minutes"]) / 60.0
