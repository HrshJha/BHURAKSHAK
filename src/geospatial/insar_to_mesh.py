"""Map satellite features onto the mesh node/grid representation — PRD §18
step 5 + FR-13 (T-060).

§18 step 5: satellite-derived deformation joins the sensor-mesh feature
schema **on the same spatial representation** — the mesh nodes, via the §9.2
WGS84 ↔ mesh-local transform (T-032) — and FR-13 requires every fused record
to carry its own observation timestamp so its staleness is explicit, never
implied (§9.1; the T-031 alignment vocabulary: InSAR ±12 h to a window).

Inputs: the T-059 per-PS × per-date deformation table
(``data/processed/insar/deformation_timeseries.parquet``), whose persistent
scatterers live in multi-look index space, and the mesh node table
(``node_id, x, y`` mesh-local metres). Three steps:

1. **georeference** — map PS indices to WGS84 across the configs/insar.yaml
   study bbox. Sentinel-1 geometry (T121 DESC): the AZIMUTH axis is
   along-track — the satellite flies southward, so azimuth index increases
   north→south with column index; the RANGE axis is cross-track — a
   descending, right-looking pass looks west, so range index increases
   east→west with row index (an ascending pass mirrors the longitude axis).
   For the synthetic MVP scene this linear index→bbox mapping IS the
   georeferencing (documented honestly); real geocoded PS products carry
   lat/lon directly and enter the identical join unchanged.
2. **project** — WGS84 → mesh-local metres via ``wgs84_to_local``, so PS
   points and nodes share one coordinate frame.
3. **join** — per node and acquisition date, the NEAREST persistent
   scatterer carrying a VALID (finite, unmasked) observation that date wins.
   Masked low-coherence observations are never force-joined (§18 step 4
   discipline): if the nearest PS is masked that date, the next-nearest valid
   one is used, and a node with no valid observation gets no row. Every
   output row carries ``observation_timestamp`` (its own) and
   ``staleness_hours`` (``as_of`` − observation, in hours; ``as_of`` defaults
   to the latest acquisition in the frame — deterministic for batch joins).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.geospatial.crs import CRSDefinition, crs_from_config, wgs84_to_local
from src.geospatial.insar_processing import load_insar_config

__all__ = ["InsarToMeshError", "georeference_ps", "map_ps_to_mesh"]

_PS_COLUMNS = ("ps_id", "range_idx", "azimuth_idx", "date", "los_displacement_mm")
_NODE_COLUMNS = ("node_id", "x", "y")
#: PS chunk width for the distance argmin (keeps the node × PS matrix bounded).
_CHUNK = 4096

#: Output schema of the mesh-joined table (one row per node × joined date).
MESH_JOINED_COLUMNS = (
    "node_id",
    "x",
    "y",
    "date",
    "observation_timestamp",
    "staleness_hours",
    "ps_id",
    "ps_distance_m",
    "insar_los_displacement_mm",
    "insar_los_velocity_mm_yr",
    "insar_los_acceleration_mm_yr2",
    "insar_coherence",
    "insar_n_observations",
)


class InsarToMeshError(ValueError):
    """Raised on invalid InSAR→mesh join inputs or degenerate geometry."""


def _normalise_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def georeference_ps(
    timeseries: pd.DataFrame,
    *,
    grid_shape: tuple[int, int] | None = None,
    cfg: dict | None = None,
    crs: CRSDefinition | None = None,
) -> pd.DataFrame:
    """Geo-reference persistent scatterers: indices → WGS84 → mesh-local.

    ``grid_shape`` is the multi-look grid extent ``(n_rows, n_cols)``; when
    omitted it is inferred from the data's maximum indices (the synthetic
    MVP scene fills its grid, so this is exact for the T-059 parquet).
    Returns one geometry row per PS: ``ps_id, range_idx, azimuth_idx, lat,
    lon, x, y``.
    """
    if cfg is None:
        cfg = load_insar_config()
    missing = [c for c in _PS_COLUMNS if c not in timeseries.columns]
    if missing:
        raise InsarToMeshError(f"InSAR timeseries missing columns: {missing}")
    if timeseries.duplicated(subset=["ps_id", "date"]).any():
        raise InsarToMeshError("duplicate (ps_id, date) rows — one observation per point per date expected")
    per_ps = timeseries.groupby("ps_id")[["range_idx", "azimuth_idx"]].nunique()
    if (per_ps > 1).any().any():
        raise InsarToMeshError("a ps_id maps to several index positions — PS geometry must be static across dates")

    bbox = cfg["study_region"]["bbox"]
    lat_min, lat_max = float(bbox["lat_min"]), float(bbox["lat_max"])
    lon_min, lon_max = float(bbox["lon_min"]), float(bbox["lon_max"])

    if grid_shape is None:
        n_rows = int(timeseries["range_idx"].max()) + 1
        n_cols = int(timeseries["azimuth_idx"].max()) + 1
    else:
        n_rows, n_cols = int(grid_shape[0]), int(grid_shape[1])
    if n_rows < 1 or n_cols < 1:
        raise InsarToMeshError(f"degenerate PS grid shape {(n_rows, n_cols)}")

    r = timeseries["range_idx"].to_numpy(dtype=float)      # cross-track → E–W
    a = timeseries["azimuth_idx"].to_numpy(dtype=float)    # along-track → N–S
    # Descending, right-looking (looks WEST): range index increases eastward
    # index 0 at the EAST edge (lon_max), rows increase westward; azimuth
    # index increases southward (flight direction), column 0 at lat_max.
    lon = lon_max - (r / max(n_rows - 1, 1)) * (lon_max - lon_min)
    lat = lat_max - (a / max(n_cols - 1, 1)) * (lat_max - lat_min)

    crs_def = crs if crs is not None else crs_from_config()
    x, y = wgs84_to_local(crs_def, lat, lon)

    geo = pd.DataFrame(
        {
            "ps_id": timeseries["ps_id"].to_numpy(),
            "range_idx": timeseries["range_idx"].to_numpy(),
            "azimuth_idx": timeseries["azimuth_idx"].to_numpy(),
            "lat": lat,
            "lon": lon,
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
        }
    )
    return geo.drop_duplicates(subset=["ps_id"], keep="first").reset_index(drop=True)


def map_ps_to_mesh(
    timeseries: pd.DataFrame,
    node_coords: pd.DataFrame,
    *,
    grid_shape: tuple[int, int] | None = None,
    max_distance_m: float | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join per-date InSAR observations onto mesh nodes (nearest valid PS).

    ``timeseries``: T-059 deformation table (PS × date). ``node_coords``:
    ``node_id, x, y`` mesh-local metres (the §9.2 shared representation).
    ``max_distance_m`` caps the join — nodes beyond it are excluded rather
    than assigned a distant PS. ``as_of`` sets the staleness reference;
    default is the latest acquisition timestamp in the frame.

    Every row carries the observation's OWN ``observation_timestamp`` and its
    ``staleness_hours`` (FR-13: staleness explicit, never implied). A join
    summary (counts of masked/uncovered/capped node-dates) is attached as
    ``df.attrs["join_summary"]``.
    """
    missing = [c for c in _NODE_COLUMNS if c not in node_coords.columns]
    if missing:
        raise InsarToMeshError(f"node_coords missing columns: {missing}")
    if node_coords["node_id"].duplicated().any():
        raise InsarToMeshError("node_coords carries duplicate node_id values")

    geo = georeference_ps(timeseries, grid_shape=grid_shape)  # validates PS schema + uniqueness
    ts = timeseries.assign(date=timeseries["date"].astype(str))
    dates = sorted(ts["date"].unique())

    value = ts.pivot(index="ps_id", columns="date", values="los_displacement_mm").reindex(
        index=geo["ps_id"].to_numpy(), columns=dates
    )
    val_arr = value.to_numpy(dtype=float)
    valid = np.isfinite(val_arr)
    if "masked_low_coherence" in ts.columns:
        masked = (
            ts.pivot(index="ps_id", columns="date", values="masked_low_coherence")
            .reindex(index=geo["ps_id"].to_numpy(), columns=dates)
            .to_numpy(dtype=float)
        )
        valid &= ~(masked > 0)

    def _pivot(col: str) -> np.ndarray:
        if col not in ts.columns:
            return np.full_like(val_arr, np.nan, dtype=float)
        return (
            ts.pivot(index="ps_id", columns="date", values=col)
            .reindex(index=geo["ps_id"].to_numpy(), columns=dates)
            .to_numpy(dtype=float)
        )

    vel_arr = _pivot("los_velocity_mm_yr")
    acc_arr = _pivot("los_acceleration_mm_yr2")
    coh_arr = _pivot("coherence")
    nobs_arr = _pivot("n_observations")

    nodes = node_coords.sort_values("node_id").reset_index(drop=True)
    node_ids = nodes["node_id"].to_numpy()
    nx = nodes["x"].to_numpy(dtype=float)
    ny = nodes["y"].to_numpy(dtype=float)
    ps_x = geo["x"].to_numpy(dtype=float)
    ps_y = geo["y"].to_numpy(dtype=float)
    ps_ids = geo["ps_id"].to_numpy()
    n_nodes, n_ps = nx.size, ps_x.size

    obs_ts = [_normalise_timestamp(d) for d in dates]
    ref = _normalise_timestamp(as_of) if as_of is not None else max(obs_ts)

    rows: list[dict[str, object]] = []
    n_covered = 0
    n_beyond_cap = 0
    for k, d in enumerate(dates):
        col_valid = valid[:, k]
        best_d = np.full(n_nodes, np.inf)
        best_i = np.full(n_nodes, -1)
        for start in range(0, n_ps, _CHUNK):
            stop = min(start + _CHUNK, n_ps)
            idx = np.where(col_valid[start:stop])[0]
            if idx.size == 0:
                continue
            vx = ps_x[start + idx]
            vy = ps_y[start + idx]
            dist = np.hypot(nx[:, None] - vx[None, :], ny[:, None] - vy[None, :])
            loc = dist.argmin(axis=1)
            dv = dist[np.arange(n_nodes), loc]
            upd = dv < best_d
            best_d[upd] = dv[upd]
            best_i[upd] = start + idx[loc[upd]]

        covered = best_i >= 0
        n_covered += int(covered.sum())
        if max_distance_m is not None:
            beyond = covered & (best_d > float(max_distance_m))
            n_beyond_cap += int(beyond.sum())
            covered &= ~beyond

        for i in np.where(covered)[0]:
            p = int(best_i[i])
            t = obs_ts[k]
            rows.append(
                {
                    "node_id": node_ids[i],
                    "x": float(nx[i]),
                    "y": float(ny[i]),
                    "date": d,
                    "observation_timestamp": t,
                    "staleness_hours": (ref - t).total_seconds() / 3600.0,
                    "ps_id": ps_ids[p],
                    "ps_distance_m": float(best_d[i]),
                    "insar_los_displacement_mm": float(val_arr[p, k]),
                    "insar_los_velocity_mm_yr": float(vel_arr[p, k]),
                    "insar_los_acceleration_mm_yr2": float(acc_arr[p, k]),
                    "insar_coherence": float(coh_arr[p, k]),
                    "insar_n_observations": int(nobs_arr[p, k]),
                }
            )

    out = pd.DataFrame(rows, columns=list(MESH_JOINED_COLUMNS))
    out = out.sort_values(["node_id", "date"], kind="stable").reset_index(drop=True)
    out.attrs["join_summary"] = {
        "n_ps": n_ps,
        "n_nodes": n_nodes,
        "n_dates": len(dates),
        "n_rows_emitted": len(out),
        "n_node_dates_with_valid_observation": n_covered,
        "n_node_dates_without_valid_observation": n_nodes * len(dates) - n_covered,
        "n_node_dates_beyond_max_distance": n_beyond_cap,
        "n_masked_or_nan_observations": int((~valid).sum()),
        "max_join_distance_m": float(out["ps_distance_m"].max()) if len(out) else None,
        "as_of": str(ref),
        "staleness_reference": "explicit as_of" if as_of is not None else "latest acquisition in frame",
    }
    return out
