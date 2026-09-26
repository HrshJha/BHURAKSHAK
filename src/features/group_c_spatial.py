"""Feature Group C — Spatial — PRD §13 Group C, FR-4 (T-037).

§13 Group C, exactly: ``neighbor_mean, neighbor_std, neighbor_anomaly_fraction,
spatial_coherence, local_gradient, local_strain, hotspot_density,
distance_to_subsidence_center``.

All spatial features are computed over the node's **neighbourhood** (nodes
within ``radius_m`` of it in the mesh-local frame, excluding itself; the
hotspot radius is twice that and includes the node). The default radius is
the configured multiplier × grid spacing (configs/physics.yaml ``grid``), i.e.
the 8 immediate neighbours on the §10 square grid.

Two acceptance-critical behaviours:

- **Spatial coherence distinguishes a single-node disturbance from coherent
  multi-node movement**: coherence is the fraction of neighbours whose
  displacement window-trend points the same way as the node's. A lone
  disturbed node has near-zero coherence; a moving panel has near-one.

- **Gap G-5 leakage rule for ``distance_to_subsidence_center``** — on real
  data the subsidence centre is the thing being detected, so using the true
  centre would leak the target. The explicit rule: ``center_mode="detected"``
  (default, inference-safe) computes the centre as the centroid of nodes the
  *pipeline itself* has flagged anomalous in that event/window (fallback:
  the max-displacement node); ``center_mode="oracle"`` uses the configured
  panel centre and is legal only for synthetic/training data where §10 defines
  the centre. The mode is recorded in the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config

__all__ = [
    "GROUP_C_FEATURES",
    "emit_group_c",
    "neighbour_radius_m",
]

GROUP_C_FEATURES = (
    "neighbor_mean",
    "neighbor_std",
    "neighbor_anomaly_fraction",
    "spatial_coherence",
    "local_gradient",
    "local_strain",
    "hotspot_density",
    "distance_to_subsidence_center",
)

_WINDOW_KEYS = ("event_id", "node_id", "window_index", "window_timestamp")


class SpatialFeatureError(ValueError):
    """Raised on invalid spatial-feature inputs."""


def neighbour_radius_m() -> float:
    """Default neighbourhood radius: config multiplier × grid spacing."""
    grid = load_config("physics")["grid"]
    return float(grid["neighbour_radius_multiplier"]) * float(grid["spacing_m"])


def _oracle_center() -> tuple[float, float]:
    p = load_config("physics")["physics"]
    return float(p["panel_center_x"]), float(p["panel_center_y"])


def emit_group_c(
    windowed: pd.DataFrame,
    node_coords: pd.DataFrame,
    *,
    radius_m: float | None = None,
    hotspot_radius_m: float | None = None,
    center_mode: str = "detected",
    channel: str = "displacement",
) -> pd.DataFrame:
    """Emit Group C features for one window level across the mesh.

    ``windowed`` must carry the window keys plus ``{channel}_mean`` and
    ``{channel}_slope``; ``node_coords`` must carry ``node_id, x, y``.
    ``center_mode``: ``"detected"`` (inference-safe, G-5 rule) or ``"oracle"``
    (synthetic/training only).
    """
    if center_mode not in ("detected", "oracle"):
        raise SpatialFeatureError(f"center_mode must be 'detected' or 'oracle', got {center_mode!r}")
    radius = float(radius_m) if radius_m is not None else neighbour_radius_m()
    hot_radius = float(hotspot_radius_m) if hotspot_radius_m is not None else 2.0 * radius
    for col in (f"{channel}_mean", f"{channel}_slope"):
        if col not in windowed.columns:
            raise SpatialFeatureError(f"windowed frame missing {col!r}")

    coords = node_coords.set_index("node_id")
    rows: list[dict[str, object]] = []

    for (event, w_idx), snap in windowed.groupby(["event_id", "window_index"], sort=False):
        snap = snap.sort_values("node_id")
        ids = snap["node_id"].to_numpy()
        xy = coords.loc[ids, ["x", "y"]].to_numpy(dtype=float)
        disp = snap[f"{channel}_mean"].to_numpy(dtype=float)
        slope = snap[f"{channel}_slope"].to_numpy(dtype=float)
        anom = (
            snap["anomaly_label"].to_numpy(dtype=float)
            if "anomaly_label" in snap.columns
            else np.zeros(len(snap))
        )

        dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)  # exclude self from the neighbour set

        # G-5: the centre the FEATURE may see
        if center_mode == "oracle":
            cx, cy = _oracle_center()
        else:
            flagged = anom > 0
            if flagged.any():
                cx, cy = xy[flagged].mean(axis=0)
            else:
                cx, cy = xy[int(np.argmax(disp))]

        for pos, (_idx, row_base) in enumerate(snap.iterrows()):
            nbr = dist[pos] <= radius
            hot = (dist[pos] <= hot_radius) | (ids == row_base["node_id"])
            n_nbr = int(nbr.sum())
            own = disp[pos]
            own_sign = np.sign(slope[pos])

            if n_nbr:
                nbr_disp = disp[nbr]
                neighbor_mean = float(nbr_disp.mean())
                neighbor_std = float(nbr_disp.std())
                anomaly_fraction = float(anom[nbr].mean())
                coherence = float((np.sign(slope[nbr]) == own_sign).mean())
                signed = (own - nbr_disp) / dist[pos][nbr]
                local_gradient = float(signed[np.argmax(np.abs(signed))])
                local_strain = float(np.abs(signed).max())
            else:  # isolated node: self-only, vacuously coherent
                neighbor_mean = float(own)
                neighbor_std = 0.0
                anomaly_fraction = 0.0
                coherence = 1.0
                local_gradient = 0.0
                local_strain = 0.0

            hotspot_density = float(anom[hot].mean())
            d_center = float(np.hypot(xy[pos, 0] - cx, xy[pos, 1] - cy))

            row = {k: row_base[k] for k in _WINDOW_KEYS if k in row_base.index}
            row.update(
                neighbor_mean=neighbor_mean,
                neighbor_std=neighbor_std,
                neighbor_anomaly_fraction=anomaly_fraction,
                spatial_coherence=coherence,
                local_gradient=local_gradient,
                local_strain=local_strain,
                hotspot_density=hotspot_density,
                distance_to_subsidence_center=d_center,
                center_mode=center_mode,
            )
            rows.append(row)

    return pd.DataFrame(rows)
