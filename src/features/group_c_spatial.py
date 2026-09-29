"""Feature Group C — observable spatial features.

Spatial context is emitted only when a snapshot contains co-temporal nodes.
The current one-node-per-event production corpus therefore receives gated
NaNs; nodes from different events are never treated as neighbours. Proxy
anomaly statistics use only same-snapshot observable deformation slopes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config

__all__ = ["GROUP_C_FEATURES", "emit_group_c", "neighbour_radius_m"]

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
    """Configured neighborhood radius: multiplier × grid spacing."""
    grid = load_config("physics")["grid"]
    return float(grid["neighbour_radius_multiplier"]) * float(grid["spacing_m"])


def emit_group_c(
    windowed: pd.DataFrame,
    node_coords: pd.DataFrame,
    *,
    radius_m: float | None = None,
    hotspot_radius_m: float | None = None,
    center_mode: str = "detected",
    channel: str = "displacement",
) -> pd.DataFrame:
    """Emit causal, label-blind spatial features for each event snapshot.

 ``center_mode='oracle'`` is rejected because model feature builds cannot
 consume simulator geometry. ``detected`` estimates the center from the
 current snapshot's observable displacement magnitudes only.
 """
    if center_mode == "oracle":
        raise SpatialFeatureError("center_mode='oracle' is forbidden for model features")
    if center_mode != "detected":
        raise SpatialFeatureError(f"center_mode must be 'detected', got {center_mode!r}")
    required = (f"{channel}_mean", f"{channel}_slope")
    missing = [col for col in required if col not in windowed.columns]
    if missing:
        raise SpatialFeatureError(f"windowed frame missing {missing}")
    for col in ("node_id", "x", "y"):
        if col not in node_coords.columns:
            raise SpatialFeatureError(f"node_coords missing {col!r}")

    cfg = load_config("features")["group_c"]
    radius = float(radius_m) if radius_m is not None else neighbour_radius_m()
    hot_radius = float(hotspot_radius_m) if hotspot_radius_m is not None else float(cfg["hotspot_radius_multiplier"]) * radius
    z_threshold = float(cfg["robust_z_threshold"])
    mad_epsilon = float(cfg["mad_epsilon"])
    coords = node_coords.set_index("node_id")
    rows: list[dict[str, object]] = []

    for (_event, _w_idx), snap in windowed.groupby(["event_id", "window_index"], sort=False):
        snap = snap.sort_values("node_id", kind="stable")
        ids = snap["node_id"].to_numpy()
        xy = coords.loc[ids, ["x", "y"]].to_numpy(dtype=float)
        disp = snap[f"{channel}_mean"].to_numpy(dtype=float)
        slopes = snap[f"{channel}_slope"].to_numpy(dtype=float)
        n = len(snap)
        dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)

        # Robust-z deformation-rate proxy is computed only among co-temporal
        # nodes. It uses no target, model score, or future window.
        finite = slopes[np.isfinite(slopes)]
        if finite.size:
            median = float(np.median(finite))
            mad = float(np.median(np.abs(finite - median)))
            robust_z = np.abs(slopes - median) / max(1.4826 * mad, mad_epsilon)
            proxy_flag = np.isfinite(robust_z) & (robust_z >= z_threshold)
        else:
            proxy_flag = np.zeros(n, dtype=bool)

        # Weighted centroid uses only the current observable snapshot.
        center_weights = np.abs(disp)
        if np.isfinite(center_weights).any() and float(np.nansum(center_weights)) > 0:
            weights = np.where(np.isfinite(center_weights), center_weights, 0.0)
            center = np.average(xy, axis=0, weights=weights)
        else:
            center = np.array([np.nan, np.nan])

        for pos, (_idx, row_base) in enumerate(snap.iterrows()):
            nbr = dist[pos] <= radius
            hot = (dist[pos] <= hot_radius) | (ids == row_base["node_id"])
            n_nbr = int(nbr.sum())
            own = disp[pos]
            own_sign = np.sign(slopes[pos])

            if n < int(cfg["minimum_cotemporal_nodes"]) or n_nbr == 0:
                values = {name: np.nan for name in GROUP_C_FEATURES}
                gate = "single-node events, no co-temporal neighbours"
                mode = "gated_no_cotemporal_neighbors"
            else:
                nbr_disp = disp[nbr]
                signed = (own - nbr_disp) / dist[pos][nbr]
                values = {
                    "neighbor_mean": float(np.nanmean(nbr_disp)),
                    "neighbor_std": float(np.nanstd(nbr_disp)),
                    "neighbor_anomaly_fraction": float(proxy_flag[nbr].mean()),
                    "spatial_coherence": float((np.sign(slopes[nbr]) == own_sign).mean()),
                    "local_gradient": float(signed[np.nanargmax(np.abs(signed))]),
                    "local_strain": float(np.nanmax(np.abs(signed))),
                    "hotspot_density": float(proxy_flag[hot].mean()),
                    "distance_to_subsidence_center": float(np.hypot(*(xy[pos] - center))),
                }
                gate = ""
                mode = "detected"

            row = {k: row_base[k] for k in _WINDOW_KEYS if k in row_base.index}
            row.update(values, center_mode=mode, spatial_gate_reason=gate)
            rows.append(row)

    return pd.DataFrame(rows)
