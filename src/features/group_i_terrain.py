"""Feature Group I — Terrain / mine geometry — PRD §13 Group I (T-066).

§13 Group I, exactly: ``elevation, slope, aspect, curvature, mine_depth,
panel_distance, panel_geometry, overburden``.

Terrain is STATIC at node level — the ground doesn't move between windows —
so the emitter takes the mesh node table and broadcasts each node's row
unchanged across the caller's window frame (same keys, 1:1 order). Nothing
here is measured: **G-10 honesty** — no real DEM, LiDAR or mine survey exists
in MVP scope. Elevation is a smooth synthetic surface (regional dip plus
seeded gentle undulations) parameterised from configs/physics.yaml
``terrain``; the provenance is stamped on every frame via ``terrain_source``
and the surface parameters echo into the frame ``attrs``. ``mine_depth``,
``panel_distance`` and ``panel_geometry`` come from the §10 physics config —
they describe the SYNTHETIC working panel, the same one the physics engine
subsides, so the features are consistent with the data by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import load_config

__all__ = [
    "GROUP_I_FEATURES",
    "TERRAIN_COLUMNS",
    "GroupITerrainError",
    "terrain_surface",
    "emit_group_i",
]

GROUP_I_FEATURES = (
    "elevation",
    "slope",
    "aspect",
    "curvature",
    "mine_depth",
    "panel_distance",
    "panel_geometry",
    "overburden",
)

#: The full Group I frame: window keys + the eight features + provenance.
TERRAIN_COLUMNS = ("event_id", "node_id", "window_index", *GROUP_I_FEATURES, "terrain_source")

_TERRAIN_SOURCE = "synthetic (G-10: no real DEM in MVP scope)"


class GroupITerrainError(ValueError):
    """Raised on invalid Group I inputs or unknown terrain config."""


def _terrain_params() -> dict:
    terrain = load_config("physics").get("terrain")
    if not isinstance(terrain, dict):
        raise GroupITerrainError("configs/physics.yaml missing the `terrain` block (Group I is config-driven)")
    return terrain


def terrain_surface(node_coords: pd.DataFrame) -> pd.DataFrame:
    """Per-node static terrain and mine geometry (the Group I payload).

    ``node_coords``: ``node_id, x, y`` (mesh-local metres, §9.2). Returns one
    row per node with ``elevation, slope, aspect, curvature, mine_depth,
    panel_distance, panel_geometry, overburden`` plus ``terrain_source`` and
    a ``terrain_params`` attrs echo of the config used.
    """
    if node_coords["node_id"].duplicated().any():
        raise GroupITerrainError("node_coords carries duplicate node_id values")
    for col in ("node_id", "x", "y"):
        if col not in node_coords.columns:
            raise GroupITerrainError(f"node_coords missing {col!r}")

    tp = _terrain_params()
    physics = load_config("physics")["physics"]
    x = node_coords["x"].to_numpy(dtype=float)
    y = node_coords["y"].to_numpy(dtype=float)

    # Synthetic elevation surface (G-10): regional dip + seeded undulations.
    base = float(tp["base_elevation_m"])
    slope_frac = float(tp["regional_slope_fraction"])
    az = math.radians(float(tp["regional_slope_azimuth_deg"]))
    amp = float(tp["undulation_amplitude_m"])
    wl = float(tp["undulation_wavelength_m"])
    seed = int(tp["seed"])

    k = (2.0 * math.pi) / wl
    rng = np.random.default_rng(seed)
    h = np.zeros(x.shape, dtype=float)
    dh_dx = np.zeros(x.shape, dtype=float)
    dh_dy = np.zeros(x.shape, dtype=float)
    curv = np.zeros(x.shape, dtype=float)
    # Seeded gentle undulations, with ANALYTIC partials — slope/aspect/curvature
    # are computed from the full gradient of the actual surface, never eyeballed.
    for amp_k, phase_k in zip(
        rng.uniform(0.35, 1.0, size=3),
        rng.uniform(0.0, 2.0 * math.pi, size=3),
        strict=True,
    ):
        theta = k * (math.cos(phase_k) * x + math.sin(phase_k) * y)
        h += amp * amp_k * np.sin(theta)
        c = amp * amp_k * k
        dh_dx += c * np.cos(theta) * math.cos(phase_k)
        dh_dy += c * np.cos(theta) * math.sin(phase_k)
        # Second derivative along the propagation direction: sag minima
        # (concave-up) come out positive.
        curv += -amp * amp_k * k**2 * np.sin(theta)

    # Regional dip added last, configured azimuth = DOWNSLOPE bearing:
    # h -= slope_frac * (x·sin az + y·cos az), so the downhill unit vector is
    # (sin az, cos az) — bearing az exactly on an undulation-free surface.
    dh_dx += -slope_frac * math.sin(az)
    dh_dy += -slope_frac * math.cos(az)
    h -= slope_frac * (x * math.sin(az) + y * math.cos(az))

    slope = np.hypot(dh_dx, dh_dy)  # rise per run (fraction, not degrees)
    aspect = (np.degrees(np.arctan2(-dh_dx, -dh_dy))) % 360.0  # downslope bearing
    curvature = curv

    # --- Mine geometry from the §10 physics config (the synthetic panel).
    cx = float(physics["panel_center_x"])
    cy = float(physics["panel_center_y"])
    width = float(physics["panel_width"])
    length = float(physics["panel_length"])
    depth = float(physics["mine_depth"])

    dx = np.maximum(np.abs(x - cx) - width / 2.0, 0.0)
    dy = np.maximum(np.abs(y - cy) - length / 2.0, 0.0)
    panel_distance = np.hypot(dx, dy)  # metres; 0 inside the panel footprint
    panel_geometry = (width * length) / 1.0e6  # km², constant per deployment
    overburden = depth + h  # ground → seam column, metres

    out = node_coords[["node_id"]].copy()
    out["x"] = x
    out["y"] = y
    out["elevation"] = h
    out["slope"] = slope
    out["aspect"] = aspect
    out["curvature"] = curvature
    out["mine_depth"] = depth
    out["panel_distance"] = panel_distance
    out["panel_geometry"] = panel_geometry
    out["overburden"] = overburden
    out["terrain_source"] = _TERRAIN_SOURCE
    out.attrs["terrain_params"] = dict(tp)
    return out


def emit_group_i(
    node_coords: pd.DataFrame,
    windowed: pd.DataFrame,
    *,
    terrain: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Broadcast static Group I terrain features onto the window frame.

    ``node_coords``: ``node_id, x, y`` (mesh-local metres). ``windowed``: the
    caller's window frame (must carry the window keys; ``window_timestamp``
    optional). The Group I row for a node is IDENTICAL at every window —
    terrain doesn't change between windows — so rows align 1:1 with
    ``windowed`` (same keys, same order). A precomputed ``terrain`` frame
    (``terrain_surface`` output) may be passed to avoid recomputation.
    """
    for col in ("event_id", "node_id", "window_index"):
        if col not in windowed.columns:
            raise GroupITerrainError(f"windowed frame missing {col!r}")

    if terrain is None:
        terrain = terrain_surface(node_coords)
    else:
        missing = {
            "node_id",
            "elevation",
            "slope",
            "aspect",
            "curvature",
            "mine_depth",
            "panel_distance",
            "panel_geometry",
            "overburden",
        } - set(terrain.columns)
        if missing:
            raise GroupITerrainError(f"precomputed terrain frame missing columns: {sorted(missing)}")
    terr_by_node = terrain.set_index("node_id")

    rows: list[dict[str, object]] = []
    for _idx, wrow in windowed.iterrows():
        node = str(wrow["node_id"])
        if node not in terr_by_node.index:
            raise GroupITerrainError(f"windowed row references unknown node {node!r}")
        t = terr_by_node.loc[node]
        rows.append(
            {
                "event_id": wrow["event_id"],
                "node_id": wrow["node_id"],
                "window_index": wrow["window_index"],
                "elevation": float(t["elevation"]),
                "slope": float(t["slope"]),
                "aspect": float(t["aspect"]),
                "curvature": float(t["curvature"]),
                "mine_depth": float(t["mine_depth"]),
                "panel_distance": float(t["panel_distance"]),
                "panel_geometry": float(t["panel_geometry"]),
                "overburden": float(t["overburden"]),
                "terrain_source": _TERRAIN_SOURCE,
            }
        )
    out = pd.DataFrame(rows)
    if "terrain_params" in terrain.attrs:
        out.attrs["terrain_params"] = terrain.attrs["terrain_params"]
    return out
