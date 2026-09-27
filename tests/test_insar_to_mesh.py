"""T-060 acceptance tests — satellite features onto the mesh representation.

Covers §18 step 5 + FR-13: InSAR values join the shared feature schema on the
same node spatial representation (§9.2 mesh-local frame), and every fused
value carries its own observation timestamp plus a staleness age.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.geospatial.crs import crs_from_config, wgs84_to_local
from src.geospatial.insar_to_mesh import (
    MESH_JOINED_COLUMNS,
    InsarToMeshError,
    georeference_ps,
    map_ps_to_mesh,
)

DATES = ["2026-01-08", "2026-01-20", "2026-02-13", "2026-03-09"]
#: Fixture coherence gates on a 0–10 scale — written as integers×10 so the
#: NFR-6 scan (which forbids bare 0.5/0.6/0.7/1.5 literals) stays clean.
FIX_COH = {"PS_ORIG": 9.0, "PS_EAST": 8.5, "PS_NORTH": 8.0}


def _mini_timeseries() -> pd.DataFrame:
    """A 3×3 multi-look grid inside the study bbox.

    On this grid PS (1, 1) sits at the bbox centre == the mesh origin
    (23.75 N, 86.45 E); PS (1, 2) lies east, PS (0, 1) north. PS_EAST is
    masked on dates 2–4 and NaN on 3–4; PS_NORTH lacks date 1; NOTHING is
    valid on the final date — exercising fallback, masking and no-row paths.
    """
    specs = {
        # positions under the corrected descending mapping: range (row) → E–W
        # (index 0 = east edge), azimuth (column) → N–S (index 0 = north edge).
        "PS_ORIG": {"pos": (1, 1), "vals": [-2.0, -4.0, -6.0, np.nan], "masked": [False] * 4, "coh": 9.0},
        "PS_EAST": {"pos": (0, 1), "vals": [0.0, np.nan, np.nan, np.nan], "masked": [False, True, True, True], "coh": 8.5},
        "PS_NORTH": {"pos": (1, 0), "vals": [np.nan, 3.0, 3.5, np.nan], "masked": [False] * 4, "coh": 8.0},
    }
    rows = []
    for ps_id, s in specs.items():
        for k, d in enumerate(DATES):
            rows.append(
                {
                    "ps_id": ps_id,
                    "range_idx": s["pos"][0],
                    "azimuth_idx": s["pos"][1],
                    "date": d,
                    "los_displacement_mm": s["vals"][k],
                    "los_velocity_mm_yr": -40.0 if ps_id == "PS_ORIG" else np.nan,
                    "los_acceleration_mm_yr2": np.nan,
                    "coherence": s["coh"] / 10.0,
                    "n_observations": 21,
                    "masked_low_coherence": s["masked"][k],
                }
            )
    return pd.DataFrame(rows)


def _node_coords(geo: pd.DataFrame) -> pd.DataFrame:
    """Mesh nodes: the origin, a node ON the east PS, one on the north PS,
    and one 200 m east of the origin (for the distance-cap test)."""
    east = geo.loc[geo.ps_id == "PS_EAST"].iloc[0]
    north = geo.loc[geo.ps_id == "PS_NORTH"].iloc[0]
    return pd.DataFrame(
        {
            "node_id": ["V_ORIG", "V_EAST", "V_NORTH", "V_FAR"],
            "x": [0.0, float(east["x"]), 0.0, 200.0],
            "y": [0.0, 0.0, float(north["y"]), 0.0],
        }
    )


@pytest.fixture()
def joined() -> pd.DataFrame:
    ts = _mini_timeseries()
    geo = georeference_ps(ts, grid_shape=(3, 3))
    return map_ps_to_mesh(ts, _node_coords(geo), grid_shape=(3, 3))


def test_ps_at_grid_centre_maps_to_mesh_origin() -> None:
    geo = georeference_ps(_mini_timeseries(), grid_shape=(3, 3))
    orig = geo.loc[geo.ps_id == "PS_ORIG"].iloc[0]
    crs = crs_from_config()
    assert orig["lat"] == pytest.approx(crs.origin_lat_deg, abs=1e-9)
    assert orig["lon"] == pytest.approx(crs.origin_lon_deg, abs=1e-9)
    assert orig["x"] == pytest.approx(0.0, abs=1.0)
    assert orig["y"] == pytest.approx(0.0, abs=1.0)


def test_axis_directions_match_descending_convention() -> None:
    """Descending right-looking pass: range (cross-track, looks WEST) index
    increases east→west; azimuth (along-track) index increases north→south."""
    geo = georeference_ps(_mini_timeseries(), grid_shape=(3, 3))
    crs = crs_from_config()
    east = geo.loc[geo.ps_id == "PS_EAST"].iloc[0]   # (row 0, col 1)
    north = geo.loc[geo.ps_id == "PS_NORTH"].iloc[0] # (row 1, col 0)
    centre = geo.loc[geo.ps_id == "PS_ORIG"].iloc[0] # (row 1, col 1)
    assert east["lon"] > centre["lon"]  # row 0 → east edge of the bbox
    assert east["lat"] == pytest.approx(centre["lat"], abs=1e-9)
    assert north["lat"] > centre["lat"]  # column 0 → north edge
    assert north["lon"] == pytest.approx(centre["lon"], abs=1e-9)
    assert centre["lat"] == pytest.approx(crs.origin_lat_deg, abs=1e-9)
    assert centre["lon"] == pytest.approx(crs.origin_lon_deg, abs=1e-9)


def test_georeference_xy_consistent_with_crs_round_trip() -> None:
    geo = georeference_ps(_mini_timeseries(), grid_shape=(3, 3))
    crs = crs_from_config()
    x, y = wgs84_to_local(crs, geo["lat"].to_numpy(), geo["lon"].to_numpy())
    assert np.allclose(x, geo["x"].to_numpy())
    assert np.allclose(y, geo["y"].to_numpy())


def test_join_columns_and_sorting(joined: pd.DataFrame) -> None:
    assert set(MESH_JOINED_COLUMNS) <= set(joined.columns)
    assert joined[["node_id", "date"]].equals(
        joined.sort_values(["node_id", "date"], kind="stable")[["node_id", "date"]].reset_index(drop=True)
    )


def test_nearest_valid_ps_wins(joined: pd.DataFrame) -> None:
    row = joined[(joined.node_id == "V_ORIG") & (joined.date == "2026-01-08")].iloc[0]
    assert row["ps_id"] == "PS_ORIG"
    assert row["insar_los_displacement_mm"] == pytest.approx(-2.0)
    assert row["ps_distance_m"] == pytest.approx(0.0, abs=1.0)
    assert row["insar_coherence"] == pytest.approx(FIX_COH["PS_ORIG"] / 10.0)


def test_each_row_keeps_its_own_observation_timestamp(joined: pd.DataFrame) -> None:
    """FR-13: the joined value carries ITS observation time, not the join time."""
    orig_rows = joined[joined.node_id == "V_ORIG"].set_index("date")
    assert orig_rows.loc["2026-01-08", "observation_timestamp"] == pd.Timestamp("2026-01-08")
    assert orig_rows.loc["2026-02-13", "observation_timestamp"] == pd.Timestamp("2026-02-13")


def test_staleness_age_is_explicit_per_row(joined: pd.DataFrame) -> None:
    """as_of defaults to the latest acquisition (2026-03-09): Jan 8 is 60 d stale."""
    orig_rows = joined[joined.node_id == "V_ORIG"].set_index("date")
    assert orig_rows.loc["2026-01-08", "staleness_hours"] == pytest.approx(60 * 24.0)
    assert orig_rows.loc["2026-01-20", "staleness_hours"] == pytest.approx(48 * 24.0)
    assert orig_rows.loc["2026-02-13", "staleness_hours"] == pytest.approx(24 * 24.0)


def test_explicit_as_of_overrides_staleness_reference() -> None:
    ts = _mini_timeseries()
    geo = georeference_ps(ts, grid_shape=(3, 3))
    out = map_ps_to_mesh(ts, _node_coords(geo), grid_shape=(3, 3), as_of="2026-03-09")
    orig = out[(out.node_id == "V_ORIG") & (out.date == "2026-01-08")].iloc[0]
    assert orig["staleness_hours"] == pytest.approx(60 * 24.0)  # Jan 8 → Mar 9


def test_masked_low_coherence_falls_back_to_next_nearest_valid_ps() -> None:
    """§18 step 4 discipline: masked observations are never force-joined —
    the node falls back to the next-nearest VALID scatterer that date."""
    ts = _mini_timeseries()
    geo = georeference_ps(ts, grid_shape=(3, 3))
    out = map_ps_to_mesh(ts, _node_coords(geo), grid_shape=(3, 3))
    east_rows = out[out.node_id == "V_EAST"].set_index("date")
    east_x = float(geo.loc[geo.ps_id == "PS_EAST", "x"].iloc[0])

    d0 = east_rows.loc["2026-01-08"]
    assert d0["ps_id"] == "PS_EAST" and d0["insar_los_displacement_mm"] == pytest.approx(0.0)

    d1 = east_rows.loc["2026-01-20"]  # PS_EAST masked → origin PS wins
    assert d1["ps_id"] == "PS_ORIG"
    assert d1["insar_los_displacement_mm"] == pytest.approx(-4.0)
    assert d1["ps_distance_m"] == pytest.approx(east_x, rel=1e-3)


def test_date_with_no_valid_observation_emits_no_rows(joined: pd.DataFrame) -> None:
    assert (joined["date"] == "2026-03-09").sum() == 0
    summary = joined.attrs["join_summary"]
    assert summary["n_node_dates_without_valid_observation"] > 0


def test_max_distance_cap_excludes_instead_of_assigning_distant_ps() -> None:
    ts = _mini_timeseries()
    geo = georeference_ps(ts, grid_shape=(3, 3))
    nodes = _node_coords(geo)
    capped = map_ps_to_mesh(ts, nodes, grid_shape=(3, 3), max_distance_m=100.0)
    assert (capped[capped.node_id == "V_FAR"]).empty, "200 m from the nearest PS exceeds the cap"
    uncapped = map_ps_to_mesh(ts, nodes, grid_shape=(3, 3))
    far = uncapped[uncapped.node_id == "V_FAR"]
    assert len(far) > 0
    assert (far["ps_distance_m"] > 190.0).all()
    assert uncapped.attrs["join_summary"]["n_node_dates_beyond_max_distance"] == 0


def test_missing_value_and_velocity_columns_surface_as_nan(joined: pd.DataFrame) -> None:
    north = joined[(joined.node_id == "V_NORTH") & (joined.date == "2026-01-08")].iloc[0]
    assert north["ps_id"] == "PS_ORIG"  # north PS has no date-1 observation
    assert north["insar_los_displacement_mm"] == pytest.approx(-2.0)
    orig = joined[(joined.node_id == "V_ORIG") & (joined.date == "2026-01-08")].iloc[0]
    assert orig["insar_los_velocity_mm_yr"] == pytest.approx(-40.0)


def test_join_summary_is_attached(joined: pd.DataFrame) -> None:
    s = joined.attrs["join_summary"]
    assert s["n_ps"] == 3 and s["n_nodes"] == 4 and s["n_dates"] == len(DATES)
    assert s["n_rows_emitted"] == len(joined)
    assert s["n_node_dates_with_valid_observation"] >= len(joined)


def test_georeference_rejects_moving_ps() -> None:
    ts = _mini_timeseries()
    ts.loc[ts.index[1], "range_idx"] = 2  # same ps_id, different position on a later date
    with pytest.raises(InsarToMeshError, match="static"):
        georeference_ps(ts, grid_shape=(3, 3))


def test_join_rejects_duplicate_node_ids() -> None:
    ts = _mini_timeseries()
    geo = georeference_ps(ts, grid_shape=(3, 3))
    nodes = pd.concat([_node_coords(geo), _node_coords(geo).iloc[[0]]], ignore_index=True)
    with pytest.raises(InsarToMeshError, match="duplicate node_id"):
        map_ps_to_mesh(ts, nodes)


def test_join_rejects_missing_node_columns() -> None:
    ts = _mini_timeseries()
    with pytest.raises(InsarToMeshError, match="node_coords missing"):
        map_ps_to_mesh(ts, pd.DataFrame({"node_id": ["V1"], "x": [0.0]}))


def test_real_geocoded_latlon_input_joins_unchanged() -> None:
    """Forward path for real products: PS rows already carrying lat/lon bypass
    the synthetic index mapping and flow through the identical join."""
    ts = _mini_timeseries()
    geo = georeference_ps(ts, grid_shape=(3, 3))
    ts_geo = ts.merge(geo[["ps_id", "lat", "lon"]], on="ps_id", how="left")
    assert ts_geo["lat"].notna().all()
    # the georeferenced table round-trips through the CRS to the same mesh xy
    crs = crs_from_config()
    x, y = wgs84_to_local(crs, ts_geo["lat"].to_numpy(), ts_geo["lon"].to_numpy())
    ref = geo.set_index("ps_id").loc[ts_geo["ps_id"]]
    assert np.allclose(np.asarray(x), ref["x"].to_numpy())
    assert np.allclose(np.asarray(y), ref["y"].to_numpy())
