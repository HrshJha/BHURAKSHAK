"""DGPS ingestion and mesh-vs-DGPS residual pipeline —,.: DGPS receivers occupy **sparse validation/control locations** — a handful
of high-accuracy points, not an at-scale observation layer. Their -mandated
role is **evaluation targets, not primary at-scale training labels**: the mesh
provides coverage, DGPS independently audits it. This module therefore marks
every emitted frame ``usage_class = "evaluation_target_only"`` and provides
``assert_evaluation_only`` for downstream guards (the ablation reads
DGPS only through evaluation-side metrics, never as model input columns at
scale — Group G's five features summarise the *residual*, itself an
evaluation construct).

 honesty note: names no receiver, vendor or survey partner. The
``synthesize_dgps_survey`` fixture stands in for a real campaign (mm-level
noise on the field at control nodes, optionally a known mesh-side bias to
exercise bias detection); it is labelled synthetic everywhere it surfaces.

Schema (ingested): ``point_id, x, y`` (mesh-local metres; ``lat, lon``
optional — filled from the transform when absent),
``observation_timestamp`` plus ``vertical_displacement_mm`` and
``horizontal_displacement_mm`` (optional ``accuracy_mm``). Ingestion
validates required columns, rejects duplicate (point, timestamp) rows and
NaN displacements — a control point that fails QA is DROPPED and counted,
never silently averaged away.

Residuals: each point associates to its nearest mesh node; a DGPS observation
aligns to the mesh window within the DGPS tolerance (±1 h, from the
 config — staleness explicit per ). ``mesh_vs_dgps_residual`` emits
per matched (point, node, window): both displacements, the vertical and
horizontal residuals (mesh − DGPS), the join distance and staleness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.geospatial.crs import CRSDefinition, crs_from_config, local_to_wgs84, wgs84_to_local
from src.preprocessing.align_modalities import tolerance_for

__all__ = [
    "DGPSError",
    "EVALUATION_TARGET_ONLY",
    "ingest_dgps",
    "associate_to_nodes",
    "mesh_vs_dgps_residual",
    "synthesize_dgps_survey",
    "assert_evaluation_only",
]

#: Marker attached to every emitted frame's ``attrs`` ( role discipline).
EVALUATION_TARGET_ONLY = "evaluation_target_only"

_REQUIRED = ("point_id", "x", "y", "observation_timestamp", "vertical_displacement_mm")
_RESIDUAL_COLUMNS = (
    "point_id",
    "node_id",
    "window_timestamp",
    "observation_timestamp",
    "staleness_hours",
    "point_distance_m",
    "dgps_vertical_mm",
    "dgps_horizontal_mm",
    "mesh_displacement_mm",
    "residual_vertical_mm",
    "residual_horizontal_mm",
)


class DGPSError(ValueError):
    """Raised on invalid DGPS inputs, schema violations or QA failures."""


def _normalise_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def ingest_dgps(
    observations: pd.DataFrame,
    *,
    crs: CRSDefinition | None = None,
) -> pd.DataFrame:
    """Validate and normalise raw DGPS control-point observations.

 Fills missing WGS84 (or mesh-local) coordinates from the transform,
 drops QA-failing rows (NaN displacement, unparseable timestamps) with an
 explicit dropped-row count, and returns the frame sorted by
 (point_id, observation_timestamp) with ``observation_timestamp`` as
 naive UTC Timestamps.
 """
    crs_def = crs if crs is not None else crs_from_config()
    missing = [c for c in _REQUIRED if c not in observations.columns]
    if missing:
        raise DGPSError(f"DGPS observations missing required columns: {missing}")

    df = observations.copy()
    df["observation_timestamp"] = pd.to_datetime(df["observation_timestamp"], errors="coerce")
    qa = df["observation_timestamp"].notna() & df["vertical_displacement_mm"].notna()
    n_dropped = int((~qa).sum())
    if not qa.any():
        raise DGPSError("no DGPS observation passes QA — nothing to ingest")
    df = df[qa].reset_index(drop=True)

    if df.duplicated(subset=["point_id", "observation_timestamp"]).any():
        raise DGPSError("duplicate (point_id, observation_timestamp) rows — one observation per point per pass")
    if "horizontal_displacement_mm" not in df.columns:
        df["horizontal_displacement_mm"] = np.nan

    need_xy = df["x"].isna() | df["y"].isna()
    need_ll = df["lat"].isna() | df["lon"].isna() if {"lat", "lon"} <= set(df.columns) else pd.Series(True, index=df.index)
    if need_xy.any() and need_ll.any():
        raise DGPSError("each point needs either mesh-local (x, y) or WGS84 (lat, lon) coordinates")
    if need_xy.any():
        xs, ys = wgs84_to_local(crs_def, df.loc[need_xy, "lat"], df.loc[need_xy, "lon"])
        df.loc[need_xy, "x"], df.loc[need_xy, "y"] = np.asarray(xs), np.asarray(ys)
    if need_ll.any():
        lats, lons = local_to_wgs84(crs_def, df.loc[need_ll, "x"], df.loc[need_ll, "y"])
        df.loc[need_ll, "lat"], df.loc[need_ll, "lon"] = np.asarray(lats), np.asarray(lons)

    df.attrs["usage_class"] = EVALUATION_TARGET_ONLY
    df.attrs["n_rows_dropped_qa"] = n_dropped
    return df.sort_values(["point_id", "observation_timestamp"], kind="stable").reset_index(drop=True)


def associate_to_nodes(dgps: pd.DataFrame, node_coords: pd.DataFrame) -> pd.DataFrame:
    """Attach each control point's nearest mesh node (sparse → one node each).

 Returns one row per point: ``point_id, node_id, point_distance_m, x, y``.
 """
    if node_coords["node_id"].duplicated().any():
        raise DGPSError("node_coords carries duplicate node_id values")
    points = dgps.drop_duplicates("point_id")
    nx = node_coords["x"].to_numpy(dtype=float)
    ny = node_coords["y"].to_numpy(dtype=float)
    rows = []
    for row in points.itertuples(index=False):
        d = np.hypot(nx - float(row.x), ny - float(row.y))
        k = int(d.argmin())
        rows.append(
            {
                "point_id": row.point_id,
                "node_id": node_coords["node_id"].iloc[k],
                "point_distance_m": float(d[k]),
                "x": float(row.x),
                "y": float(row.y),
            }
        )
    return pd.DataFrame(rows)


def mesh_vs_dgps_residual(
    mesh_df: pd.DataFrame,
    dgps: pd.DataFrame,
    node_coords: pd.DataFrame,
    *,
    epoch: pd.Timestamp | str | None = None,
    tolerance_hours: float | None = None,
) -> pd.DataFrame:
    """Matched mesh-vs-DGPS displacement residuals (, evaluation side).

 ``mesh_df`` must carry ``node_id, window_timestamp`` (hours on the same
 epoch axis as ``epoch``) and ``displacement_mean`` (mesh vertical
 displacement in mm; optional ``horizontal_displacement_mm``).
 ``dgps``: ingested control-point observations ( schema).
 ``epoch`` converts DGPS timestamps onto the mesh hours axis; required
 unless the mesh already carries a ``window_ts`` datetime column.
 Observations outside the DGPS tolerance are reported as unmatched
 (in ``attrs``), never force-joined.
 """
    tol = float(tolerance_hours) if tolerance_hours is not None else tolerance_for("dgps")
    need = {"displacement_mean"}
    missing = need - set(mesh_df.columns)
    if missing:
        raise DGPSError(f"mesh frame missing columns: {sorted(missing)}")

    assoc = associate_to_nodes(dgps, node_coords)
    mesh = mesh_df.copy()
    if "window_ts" not in mesh.columns:
        if epoch is None:
            raise DGPSError("provide `epoch` (the window hours origin) or a `window_ts` column on the mesh frame")
        epoch_ts = _normalise_timestamp(epoch)
        mesh["window_ts"] = epoch_ts + pd.to_timedelta(
            pd.to_numeric(mesh["window_timestamp"], errors="coerce"), unit="h"
        )
    else:
        mesh["window_ts"] = pd.to_datetime(mesh["window_ts"])

    mesh_by_node = {k: v.sort_values("window_ts", kind="stable") for k, v in mesh.groupby("node_id", sort=False)}
    rows: list[dict[str, object]] = []
    n_matched = 0
    n_out_of_tolerance = 0
    for obs in dgps.itertuples(index=False):
        node = assoc.loc[assoc.point_id == obs.point_id, "node_id"]
        if node.empty:
            continue
        node = str(node.iloc[0])
        series = mesh_by_node.get(node)
        if series is None or series.empty:
            continue
        t = _normalise_timestamp(obs.observation_timestamp)
        delta = (series["window_ts"] - t).abs()
        j = int(delta.argmin())
        if float(delta.iloc[j].total_seconds()) / 3600.0 > tol:
            n_out_of_tolerance += 1
            continue
        mesh_row = series.iloc[j]
        mesh_v = float(mesh_row["displacement_mean"])
        mesh_h = float(mesh_row["horizontal_displacement_mm"]) if "horizontal_displacement_mm" in mesh_row else np.nan
        rows.append(
            {
                "point_id": obs.point_id,
                "node_id": node,
                "window_timestamp": float(mesh_row["window_timestamp"]) if "window_timestamp" in mesh_row else np.nan,
                "observation_timestamp": t,
                "staleness_hours": float(delta.iloc[j].total_seconds()) / 3600.0,
                "point_distance_m": float(assoc.loc[assoc.point_id == obs.point_id, "point_distance_m"].iloc[0]),
                "dgps_vertical_mm": float(obs.vertical_displacement_mm),
                "dgps_horizontal_mm": float(obs.horizontal_displacement_mm),
                "mesh_displacement_mm": mesh_v,
                "residual_vertical_mm": mesh_v - float(obs.vertical_displacement_mm),
                "residual_horizontal_mm": mesh_h - float(obs.horizontal_displacement_mm)
                if np.isfinite(mesh_h) and np.isfinite(obs.horizontal_displacement_mm)
                else np.nan,
            }
        )
        n_matched += 1

    out = pd.DataFrame(rows, columns=list(_RESIDUAL_COLUMNS))
    out.attrs["usage_class"] = EVALUATION_TARGET_ONLY
    out.attrs["join_summary"] = {
        "n_dgps_observations": len(dgps),
        "n_matched": n_matched,
        "n_out_of_tolerance": n_out_of_tolerance,
        "tolerance_hours": tol,
        "n_control_points": int(dgps.point_id.nunique()),
    }
    return out.sort_values(["point_id", "window_timestamp"], kind="stable").reset_index(drop=True)


def assert_evaluation_only(df: pd.DataFrame) -> None:
    """ discipline guard: raise unless the frame is marked evaluation-only."""
    if df.attrs.get("usage_class") != EVALUATION_TARGET_ONLY:
        raise DGPSError("frame is not marked as an evaluation target — refusing to use it as a training label")


def synthesize_dgps_survey(
    node_coords: pd.DataFrame,
    displacement_at: "np.ndarray | None" = None,
    *,
    point_ids: list[str] | None = None,
    timestamps: list[str] | None = None,
    vertical_mm_at: "np.ndarray | None" = None,
    mesh_bias_mm: float = 0.0,
    noise_std_mm: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic control-point survey ( stand-in, labelled synthetic).

 ``vertical_mm_at``: (n_points, n_times) TRUE vertical displacement at the
 control points (the caller's ground truth — e.g. the field). The
 survey adds GNSS-level noise; ``mesh_bias_mm`` is added to the returned
 ``mesh_displacement_mm`` reference channel so downstream bias detection
 (notebook 09) has something to find.
 """
    if displacement_at is not None and vertical_mm_at is None:
        vertical_mm_at = displacement_at
    if vertical_mm_at is None:
        raise DGPSError("provide vertical_mm_at (the true displacement at the control points)")
    vertical_mm_at = np.asarray(vertical_mm_at, dtype=float)
    if vertical_mm_at.ndim != 2:
        raise DGPSError("vertical_mm_at must be (n_points, n_times)")
    n_points, n_times = vertical_mm_at.shape

    coords = node_coords.reset_index(drop=True)
    if point_ids is None:
        point_ids = [f"CP{i:02d}" for i in range(n_points)]
    if len(point_ids) != n_points:
        raise DGPSError(f"need {n_points} point_ids for the control points")
    if timestamps is None:
        raise DGPSError("provide the survey timestamps (ISO dates)")
    if len(timestamps) != n_times:
        raise DGPSError(f"need {n_times} timestamps for the survey")

    rng = np.random.default_rng(seed)
    rows = []
    for i, pid in enumerate(point_ids):
        for k, t in enumerate(timestamps):
            true_v = vertical_mm_at[i, k]
            rows.append(
                {
                    "point_id": pid,
                    "x": float(coords.loc[i, "x"]),
                    "y": float(coords.loc[i, "y"]),
                    "observation_timestamp": t,
                    "vertical_displacement_mm": float(true_v + rng.normal(0.0, noise_std_mm)),
                    "horizontal_displacement_mm": float(rng.normal(0.0, noise_std_mm)),
                    "accuracy_mm": noise_std_mm,
                    "mesh_displacement_mm": float(true_v + mesh_bias_mm),
                    "source": "synthetic ( stand-in)",
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["usage_class"] = EVALUATION_TARGET_ONLY
    return df
