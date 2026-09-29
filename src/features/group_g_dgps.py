"""Compute DGPS comparison features for validation data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.geospatial.dgps import associate_to_nodes, ingest_dgps, mesh_vs_dgps_residual
from src.preprocessing.align_modalities import tolerance_for

__all__ = ["GROUP_G_FEATURES", "GroupGError", "emit_group_g"]

GROUP_G_FEATURES = (
    "vertical_displacement",
    "horizontal_displacement",
    "velocity",
    "acceleration",
    "mesh_vs_dgps_residual",
)

_DAYS_PER_YEAR = 365.25
_MIN_OBS_FOR_FITS = 3  # linear fit; acceleration needs 5


class GroupGError(ValueError):
    """Raised on invalid Group G inputs."""


def _point_series(dgps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per control point: timestamp-ordered series with velocity/acceleration."""
    out = {}
    for pid, g in dgps.groupby("point_id", sort=False):
        g = g.sort_values("observation_timestamp", kind="stable").reset_index(drop=True)
        # pandas 3 preserves datetime64[us] for parsed timestamps; raw int64
        # ticks therefore no longer have a stable unit. Normalize explicitly
        # to nanoseconds before converting elapsed time to days.
        ticks_ns = g["observation_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
        t = ticks_ns.astype(float) / 86400e9  # ns → days
        v = g["vertical_displacement_mm"].to_numpy(dtype=float)
        vel = np.full(len(g), np.nan)
        acc = np.full(len(g), np.nan)
        if len(g) >= _MIN_OBS_FOR_FITS and np.isfinite(v).sum() >= _MIN_OBS_FOR_FITS:
            m = np.isfinite(v)
            coef = np.polyfit(t[m], v[m], 1)  # mm/day
            vel = np.full(len(g), coef[0] * _DAYS_PER_YEAR)
            if len(g) >= 5 and m.sum() >= 5:
                quad = np.polyfit(t[m], v[m], 2)
                acc = np.full(len(g), 2.0 * quad[0] * _DAYS_PER_YEAR**2)
        g = g.assign(velocity_mm_yr=vel, acceleration_mm_yr2=acc)
        out[str(pid)] = g
    return out


def emit_group_g(
    dgps_raw: pd.DataFrame,
    windowed: pd.DataFrame,
    node_coords: pd.DataFrame,
    *,
    epoch: pd.Timestamp | str,
) -> pd.DataFrame:
    """Emit Group G features per window row.

 ``dgps_raw``: raw control-point observations ( ingest schema; the
 synthetic survey fixture also carries ``mesh_displacement_mm``).
 ``windowed``: window-key rows (``event_id, node_id, window_index,
 window_timestamp`` hours on ``epoch``) — the same windows the other
 groups use. ``epoch``: REQUIRED hours-axis origin — the caller states
 the axis, never guessed ( discipline).

 The mesh side of the residual uses the window frame itself as the mesh
 estimate (``displacement_mean`` per node/window), so Group G can be
 computed directly on any windowed feature frame.
 """
    for col in ("event_id", "node_id", "window_index", "window_timestamp"):
        if col not in windowed.columns:
            raise GroupGError(f"windowed frame missing {col!r}")
    if "displacement_mean" not in windowed.columns:
        raise GroupGError("windowed frame must carry displacement_mean (the mesh estimate)")

    dgps = ingest_dgps(dgps_raw)
    wt = pd.to_numeric(windowed["window_timestamp"], errors="coerce")
    epoch_ts = pd.Timestamp(epoch)
    if epoch_ts.tzinfo is not None:
        epoch_ts = epoch_ts.tz_convert("UTC").tz_localize(None)

    windows = windowed.assign(_wt=wt, _ts=epoch_ts + pd.to_timedelta(wt, unit="h"))

    # Mesh side for the residual: the window frame's own displacement estimate.
    mesh_side = windows[["node_id", "window_index", "window_timestamp", "_ts", "displacement_mean"]].rename(
        columns={"_ts": "window_ts"}
    )

    residual = mesh_vs_dgps_residual(mesh_side, dgps, node_coords, epoch=epoch_ts)
    res_by_point: dict[str, pd.DataFrame] = {}
    if len(residual):
        for pid, g in residual.groupby("point_id", sort=False):
            res_by_point[str(pid)] = g.set_index("observation_timestamp")

    series = _point_series(dgps)

    # Node → the point serving that node (sparse: usually zero or one). The
    # survey sits at CONTROL locations, so a point serves its nearest node.
    assignment = associate_to_nodes(dgps, node_coords).set_index("point_id")
    node_to_point: dict[str, str] = {}
    for pid, row in assignment.iterrows():
        node_to_point.setdefault(str(row["node_id"]), str(pid))

    tol = tolerance_for("dgps")  #: DGPS aligns within ±1 h (config-driven)
    rows_out: list[dict[str, object]] = []
    for (_idx, row), w_ts in zip(windowed.iterrows(), windows["_ts"], strict=True):
        node = row["node_id"]
        base = {
            "event_id": row["event_id"],
            "node_id": node,
            "window_index": row["window_index"],
        }
        vals = dict.fromkeys(GROUP_G_FEATURES, np.nan)
        pid = node_to_point.get(str(node))
        if pid is not None and pd.notna(w_ts):
            g = series[pid]
            deltas = (g["observation_timestamp"] - w_ts).abs()
            j = int(deltas.argmin())
            if float(deltas.iloc[j].total_seconds()) / 3600.0 <= tol:
                obs = g.iloc[j]
                vals["vertical_displacement"] = obs["vertical_displacement_mm"]
                vals["horizontal_displacement"] = obs["horizontal_displacement_mm"]
                vals["velocity"] = obs["velocity_mm_yr"]
                vals["acceleration"] = obs["acceleration_mm_yr2"]
                r = res_by_point.get(pid)
                if r is not None:
                    key = obs["observation_timestamp"]
                    if key in r.index:
                        vals["mesh_vs_dgps_residual"] = r.loc[key, "residual_vertical_mm"]
        rows_out.append({**base, **vals})

    return pd.DataFrame(rows_out)
