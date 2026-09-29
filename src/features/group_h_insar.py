"""Feature Group H — InSAR — Group H,.

 Group H, exactly: ``LOS_displacement, LOS_velocity, LOS_acceleration,
cumulative_displacement, coherence, spatial_gradient, local_hotspot_density``.

Sources: the mesh-joined table (per node × acquisition date, nearest
valid PS, staleness-explicit). All seven names are produced as of the
window timestamp:

- ``LOS_displacement`` — the node's most recent LOS observation at or before
 the window; NaN when the observation is staler than the configured carry
 budget (configs/insar.yaml ``mesh_feature.max_staleness_hours``, default
 two× the 12-day cadence) —: staleness visible, never implied, and no
 interpolation across acquisitions.
- ``LOS_velocity`` / ``LOS_acceleration`` — per-PS trend terms carried
 through the join.
- ``cumulative_displacement`` — LOS displacement relative to the earliest
 observation in the node's joined series (the stack's own referencing);
 NaN-preserving.
- ``coherence`` — the joined observation's per-date coherence.
- ``spatial_gradient`` — per metre, the node's LOS minus the nearest
 co-timed neighbour LOS, over mesh-local distance (Group C convention).
- ``local_hotspot_density`` — fraction of mesh nodes within the hotspot
 radius whose most-recent |cumulative LOS| reaches the configured hotspot
 threshold.

Every row keeps ``observation_timestamp`` and ``staleness_hours`` so the
window can see how fresh its satellite evidence is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.geospatial.insar_to_mesh import InsarToMeshError

__all__ = ["GROUP_H_FEATURES", "InSARFeatureError", "emit_group_h"]

GROUP_H_FEATURES = (
    "LOS_displacement",
    "LOS_velocity",
    "LOS_acceleration",
    "cumulative_displacement",
    "coherence",
    "spatial_gradient",
    "local_hotspot_density",
)


class InSARFeatureError(ValueError):
    """Raised on invalid Group H inputs."""


def _mesh_feature_cfg() -> dict:
    """The fusion block of configs/insar.yaml (: no literals here)."""
    from src.geospatial.insar_processing import load_insar_config

    block = load_insar_config().get("mesh_feature")
    if not isinstance(block, dict):
        raise InSARFeatureError("configs/insar.yaml missing the mesh_feature block")
    return block


def emit_group_h(
    joined: pd.DataFrame,
    windowed: pd.DataFrame,
    node_coords: pd.DataFrame,
    *,
    max_staleness_hours: float | None = None,
    hotspot_radius_m: float | None = None,
) -> pd.DataFrame:
    """Emit Group H features per window row.

 ``joined``: the mesh-joined table (node × date with staleness).
 ``windowed``: window-key rows needing satellite evidence — must carry
 ``event_id, node_id, window_index, window_timestamp`` (hours axis).
 ``node_coords``: ``node_id, x, y`` (the shared representation).
 """
    cfg = _mesh_feature_cfg()
    budget = float(max_staleness_hours) if max_staleness_hours is not None else float(cfg["max_staleness_hours"])
    hot_threshold = float(cfg["hotspot_threshold_mm"])
    epoch = pd.Timestamp(cfg["time_epoch"])

    for col in ("event_id", "node_id", "window_index", "window_timestamp"):
        if col not in windowed.columns:
            raise InSARFeatureError(f"windowed frame missing {col!r}")
    need = {"node_id", "date", "observation_timestamp", "insar_los_displacement_mm", "staleness_hours"}
    missing = need - set(joined.columns)
    if missing:
        raise InSARFeatureError(f"joined frame missing columns: {sorted(missing)}")

    coords = node_coords.set_index("node_id")
    hotspot_radius = (
        float(hotspot_radius_m)
        if hotspot_radius_m is not None
        else 2.0 * float(load_config("physics")["grid"]["neighbour_radius_multiplier"])
        * float(load_config("physics")["grid"]["spacing_m"])
    )

    # Most-recent observation per node at or before each window, with
    # cumulative referencing per node (first joined observation = 0 mm).
    j = joined.sort_values(["node_id", "date"], kind="stable").copy()
    j["obs_ts"] = pd.to_datetime(j["observation_timestamp"])
    j["cumulative"] = j.groupby("node_id")["insar_los_displacement_mm"].transform(
        lambda s: s - s.iloc[0] if s.notna().any() else s
    )
    obs = j[["node_id", "obs_ts", "date", "insar_los_displacement_mm", "insar_los_velocity_mm_yr",
             "insar_los_acceleration_mm_yr2", "insar_coherence", "cumulative", "staleness_hours"]]
    obs_by_node = {k: v.sort_values("obs_ts", kind="stable") for k, v in obs.groupby("node_id", sort=False)}

    # Per-window snapshot state for the spatial terms.
    wt_hours = pd.to_numeric(windowed["window_timestamp"], errors="coerce").to_numpy(dtype=float)
    rows_out: list[dict[str, object]] = []
    for wt, (_idx, row) in zip(wt_hours, windowed.iterrows(), strict=True):
        rows_out.append(
            _row_features(row, float(wt), obs_by_node, coords, budget, hot_threshold, hotspot_radius, epoch)
        )

    out = pd.DataFrame(rows_out)
    keys = [c for c in ("event_id", "node_id", "window_index", "window_timestamp") if c in windowed.columns]
    for k in keys:
        out[k] = windowed[k].to_numpy()
    return out[[*keys, *GROUP_H_FEATURES, "observation_timestamp", "staleness_hours"]]


def _row_features(
    row,
    wt: float,
    obs_by_node: dict[str, pd.DataFrame],
    coords: pd.DataFrame,
    budget: float,
    hot_threshold: float,
    hotspot_radius: float,
    epoch: pd.Timestamp,
) -> dict[str, object]:
    """Assemble one window row's Group H values (as-of semantics)."""
    node = row["node_id"]
    empty = {**dict.fromkeys(GROUP_H_FEATURES, np.nan), "observation_timestamp": pd.NaT, "staleness_hours": np.nan}
    series = obs_by_node.get(node)
    if series is None or series.empty or not np.isfinite(wt):
        return empty

    hours = (series["obs_ts"] - epoch).dt.total_seconds() / 3600.0
    eligible = series[(hours <= wt) & (wt - hours <= budget)]
    if eligible.empty:
        return empty

    latest = eligible.iloc[-1]
    vals = {
        "LOS_displacement": latest["insar_los_displacement_mm"],
        "LOS_velocity": latest["insar_los_velocity_mm_yr"],
        "LOS_acceleration": latest["insar_los_acceleration_mm_yr2"],
        "cumulative_displacement": latest["cumulative"],
        "coherence": latest["insar_coherence"],
    }

    # Spatial terms need a co-timed mesh snapshot: the latest eligible
    # observation for EVERY node at this window (as-of join).
    snapshot = _snapshot(wt, obs_by_node, budget, epoch)
    x, y = float(coords.loc[node, "x"]), float(coords.loc[node, "y"])
    snap_xy = snapshot.join(coords[["x", "y"]], how="inner")
    d = np.hypot(snap_xy["x"].to_numpy(float) - x, snap_xy["y"].to_numpy(float) - y)

    finite = np.isfinite(snap_xy["LOS_displacement"].to_numpy(float))
    others = finite & (snap_xy.index != node)
    if others.any():
        k = int(np.argmin(np.where(others, d, np.inf)))
        own = float(vals["LOS_displacement"])
        neighbour = float(snap_xy["LOS_displacement"].to_numpy(float)[k])
        vals["spatial_gradient"] = (own - neighbour) / float(d[k])
    else:
        vals["spatial_gradient"] = np.nan

    hot = d <= hotspot_radius
    if hot.any():
        cum = snap_xy["cumulative"].to_numpy(float)[hot]
        mag = np.abs(cum[np.isfinite(cum)])
        vals["local_hotspot_density"] = float((mag >= hot_threshold).mean()) if mag.size else np.nan
    else:
        vals["local_hotspot_density"] = np.nan

    latest_hours = float((latest["obs_ts"] - epoch).total_seconds()) / 3600.0
    return {
        **vals,
        "observation_timestamp": latest["obs_ts"],
        "staleness_hours": wt - latest_hours,
    }


def _snapshot(
    wt: float,
    obs_by_node: dict[str, pd.DataFrame],
    budget: float,
    epoch: pd.Timestamp,
) -> pd.DataFrame:
    """As-of state per node at window hours ``wt``: latest LOS/cumulative
 within the staleness budget — the co-timed mesh snapshot."""
    rows = []
    for node, series in obs_by_node.items():
        hours = (series["obs_ts"] - epoch).dt.total_seconds() / 3600.0
        eligible = series[(hours <= wt) & (wt - hours <= budget)]
        if eligible.empty:
            continue
        latest = eligible.iloc[-1]
        rows.append(
            {
                "node_id": node,
                "LOS_displacement": latest["insar_los_displacement_mm"],
                "cumulative": latest["cumulative"],
            }
        )
    return pd.DataFrame(rows).set_index("node_id")
