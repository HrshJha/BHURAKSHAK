"""T-066 — Feature Group I (terrain / mine geometry) tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features.group_i_terrain import (
    GROUP_I_FEATURES,
    TERRAIN_COLUMNS,
    GroupITerrainError,
    emit_group_i,
    terrain_surface,
)
from src.simulator.grid import build_grid


@pytest.fixture()
def grid_nodes() -> pd.DataFrame:
    g = build_grid()
    return pd.DataFrame({"node_id": g.node_ids, "x": g.x, "y": g.y})


@pytest.fixture()
def windowed() -> pd.DataFrame:
    # two events × two nodes × two windows, deliberately unsorted node order
    rows = []
    for event in ("EV_A", "EV_B"):
        for k, node in enumerate(["V0018", "V0000"]):
            for w in range(2):
                rows.append(
                    {
                        "event_id": event,
                        "node_id": node,
                        "window_index": w,
                        "window_timestamp": 0.6 * w,
                        "displacement_mean": float(k),
                    }
                )
    return pd.DataFrame(rows)


def test_group_i_feature_names_exact() -> None:
    schema = load_config("feature_schema")["feature_groups"]["I_terrain"]
    assert tuple(schema) == GROUP_I_FEATURES
    assert GROUP_I_FEATURES == (
        "elevation",
        "slope",
        "aspect",
        "curvature",
        "mine_depth",
        "panel_distance",
        "panel_geometry",
        "overburden",
    )


def test_terrain_surface_shape_and_columns(grid_nodes: pd.DataFrame) -> None:
    terr = terrain_surface(grid_nodes)
    assert len(terr) == len(grid_nodes) == 400
    assert {"elevation", "slope", "aspect", "curvature", "mine_depth",
            "panel_distance", "panel_geometry", "overburden", "terrain_source"} <= set(terr.columns)
    assert terr["terrain_source"].str.startswith("synthetic").all()


def test_terrain_surface_reproducible(grid_nodes: pd.DataFrame) -> None:
    a, b = terrain_surface(grid_nodes), terrain_surface(grid_nodes)
    for col in GROUP_I_FEATURES:
        np.testing.assert_allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float))


def test_elevation_finite_and_bounded_by_config(grid_nodes: pd.DataFrame) -> None:
    tp = load_config("physics")["terrain"]
    terr = terrain_surface(grid_nodes)
    h = terr["elevation"].to_numpy(dtype=float)
    assert np.isfinite(h).all()
    # worst case: dip span across the mesh diagonal + peak-to-trough of
    # 3 superposed undulations (2·Σamp ≤ 2·3·amp)
    dip_span = float(tp["regional_slope_fraction"]) * 450.0 * np.sqrt(2.0)
    bound = dip_span + 6.0 * float(tp["undulation_amplitude_m"])
    assert float(h.max() - h.min()) < bound


def test_slope_aspect_from_analytic_gradient(grid_nodes: pd.DataFrame) -> None:
    """slope/aspect must match the analytic gradient of the emitted surface."""
    tp = load_config("physics")["terrain"]
    sf = float(tp["regional_slope_fraction"])
    az = np.radians(float(tp["regional_slope_azimuth_deg"]))
    amp = float(tp["undulation_amplitude_m"])
    wl = float(tp["undulation_wavelength_m"])
    seed = int(tp["seed"])
    k = 2 * np.pi / wl
    x = grid_nodes["x"].to_numpy(dtype=float)
    y = grid_nodes["y"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    dhdx = np.zeros_like(x)
    dhdy = np.zeros_like(x)
    for amp_k, phase_k in zip(rng.uniform(0.35, 1.0, size=3), rng.uniform(0.0, 2 * np.pi, size=3), strict=True):
        theta = k * (np.cos(phase_k) * x + np.sin(phase_k) * y)
        c = amp * amp_k * k
        dhdx += c * np.cos(theta) * np.cos(phase_k)
        dhdy += c * np.cos(theta) * np.sin(phase_k)
    dhdx += -sf * np.sin(az)
    dhdy += -sf * np.cos(az)

    terr = terrain_surface(grid_nodes)
    np.testing.assert_allclose(terr["slope"].to_numpy(dtype=float), np.hypot(dhdx, dhdy), rtol=1e-9)
    np.testing.assert_allclose(
        terr["aspect"].to_numpy(dtype=float), np.degrees(np.arctan2(-dhdx, -dhdy)) % 360.0, rtol=1e-9
    )


def test_aspect_is_downslope_bearing(grid_nodes: pd.DataFrame) -> None:
    """On a pure dip surface, aspect must equal the configured downslope azimuth."""
    from unittest.mock import patch

    pure = {"base_elevation_m": 0.0, "regional_slope_fraction": 0.01,
            "regional_slope_azimuth_deg": 250.0, "undulation_amplitude_m": 0.0,
            "undulation_wavelength_m": 180.0, "seed": 42, "source": "synthetic"}
    with patch("src.features.group_i_terrain._terrain_params", return_value=pure):
        terr = terrain_surface(grid_nodes)
    np.testing.assert_allclose(terr["aspect"].to_numpy(dtype=float), 250.0, atol=1e-9)


def test_curvature_positive_in_sag_minima(grid_nodes: pd.DataFrame) -> None:
    terr = terrain_surface(grid_nodes)
    h = terr["elevation"].to_numpy(dtype=float)
    c = terr["curvature"].to_numpy(dtype=float)
    assert np.isfinite(c).all()
    lows = h < np.quantile(h, 0.15)
    highs = h > np.quantile(h, 0.85)
    assert c[lows].mean() > 0.0, "sag minima are concave-up (positive Laplacian)"
    assert c[highs].mean() < 0.0, "crests are concave-down"


def test_mine_geometry_from_physics_config(grid_nodes: pd.DataFrame) -> None:
    p = load_config("physics")["physics"]
    terr = terrain_surface(grid_nodes)
    assert (terr["mine_depth"] == float(p["mine_depth"])).all()
    assert (terr["panel_geometry"] == float(p["panel_width"]) * float(p["panel_length"]) / 1.0e6).all()

    cx, cy = float(p["panel_center_x"]), float(p["panel_center_y"])
    w, ln = float(p["panel_width"]), float(p["panel_length"])
    x = grid_nodes["x"].to_numpy(dtype=float)
    y = grid_nodes["y"].to_numpy(dtype=float)
    expected = np.hypot(np.maximum(np.abs(x - cx) - w / 2.0, 0.0), np.maximum(np.abs(y - cy) - ln / 2.0, 0.0))
    np.testing.assert_allclose(terr["panel_distance"].to_numpy(dtype=float), expected, rtol=1e-12)
    # centre node sits inside the footprint
    centre = int(np.argmin(x**2 + y**2))
    assert terr["panel_distance"].iloc[centre] == 0.0


def test_overburden_is_depth_plus_elevation(grid_nodes: pd.DataFrame) -> None:
    terr = terrain_surface(grid_nodes)
    np.testing.assert_allclose(
        terr["overburden"].to_numpy(dtype=float),
        terr["mine_depth"].to_numpy(dtype=float) + terr["elevation"].to_numpy(dtype=float),
        rtol=1e-12,
    )


def test_emit_group_i_broadcasts_static_rows(grid_nodes: pd.DataFrame, windowed: pd.DataFrame) -> None:
    out = emit_group_i(grid_nodes, windowed)
    assert len(out) == len(windowed)
    assert list(out["event_id"]) == list(windowed["event_id"])
    assert list(out["node_id"]) == list(windowed["node_id"])
    assert set(GROUP_I_FEATURES) <= set(out.columns)
    # static: same node, two windows → identical feature values
    for node in windowed["node_id"].unique():
        sub = out[out.node_id == node]
        for col in GROUP_I_FEATURES:
            assert sub[col].nunique(dropna=False) == 1
    # and they agree with terrain_surface
    terr = terrain_surface(grid_nodes).set_index("node_id")
    for _, row in out.iterrows():
        t = terr.loc[row["node_id"]]
        for col in GROUP_I_FEATURES:
            assert row[col] == pytest.approx(float(t[col]))


def test_emit_group_i_matches_window_index_count(grid_nodes: pd.DataFrame) -> None:
    win = pd.DataFrame(
        {
            "event_id": ["EV_A"] * 6,
            "node_id": ["V0000"] * 6,
            "window_index": list(range(6)),
            "window_timestamp": [0.6 * i for i in range(6)],
        }
    )
    out = emit_group_i(grid_nodes, win)
    assert TERRAIN_COLUMNS == ("event_id", "node_id", "window_index", *GROUP_I_FEATURES, "terrain_source")
    assert len(out) == 6 and out["window_index"].tolist() == list(range(6))


def test_emit_group_i_rejects_unknown_node(grid_nodes: pd.DataFrame) -> None:
    win = pd.DataFrame({"event_id": ["EV_A"], "node_id": ["NOPE"], "window_index": [0]})
    with pytest.raises(GroupITerrainError, match="unknown node"):
        emit_group_i(grid_nodes, win)


def test_emit_group_i_rejects_missing_window_keys(grid_nodes: pd.DataFrame) -> None:
    with pytest.raises(GroupITerrainError, match="missing"):
        emit_group_i(grid_nodes, pd.DataFrame({"event_id": ["EV_A"]}))


def test_terrain_surface_rejects_duplicate_nodes() -> None:
    dup = pd.DataFrame({"node_id": ["V0000", "V0000"], "x": [0.0, 1.0], "y": [0.0, 1.0]})
    with pytest.raises(GroupITerrainError, match="duplicate"):
        terrain_surface(dup)


def test_emit_group_i_precomputed_terrain_round_trip(grid_nodes: pd.DataFrame, windowed: pd.DataFrame) -> None:
    terr = terrain_surface(grid_nodes)
    a = emit_group_i(grid_nodes, windowed)
    b = emit_group_i(grid_nodes, windowed, terrain=terr)
    for col in GROUP_I_FEATURES:
        np.testing.assert_allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float))


def test_emit_group_i_window_timestamp_optional(grid_nodes: pd.DataFrame) -> None:
    win = pd.DataFrame({"event_id": ["EV_A"], "node_id": ["V0000"], "window_index": [0]})
    out = emit_group_i(grid_nodes, win)
    assert len(out) == 1
