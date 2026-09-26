"""T-032 acceptance tests — §9.2 dual coordinate reference system."""

from __future__ import annotations

import numpy as np
import pytest

from src.geospatial.crs import (
    CRSError,
    crs_from_config,
    local_to_wgs84,
    wgs84_to_local,
)


def test_config_crs_loads_jharia_deployment_parameters() -> None:
    crs = crs_from_config()
    # §18 study region: Jharia coalfield (Dhanbad, Jharkhand), UTM zone 45N
    assert 23.0 < crs.origin_lat_deg < 24.5
    assert 86.0 < crs.origin_lon_deg < 87.5
    assert crs.utm_zone == 45
    assert crs.projection_model == "equirectangular"


def test_origin_maps_to_itself() -> None:
    crs = crs_from_config()
    lat, lon = local_to_wgs84(crs, 0.0, 0.0)
    assert lat == pytest.approx(crs.origin_lat_deg)
    assert lon == pytest.approx(crs.origin_lon_deg)
    x, y = wgs84_to_local(crs, lat, lon)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_every_node_carries_both_representations() -> None:
    """§9.2: mesh-local (x, y) AND WGS84 (lat, lon) for every node."""
    from src.simulator.grid import build_grid

    crs = crs_from_config()
    grid = build_grid()
    lat, lon = local_to_wgs84(crs, grid.x, grid.y)
    assert lat.shape == grid.x.shape and lon.shape == grid.x.shape
    assert grid.n_nodes == 400
    assert np.all((lat >= 23.0) & (lat <= 24.5)), "nodes stay inside the study region"


def test_round_trip_returns_to_within_0_1_m() -> None:
    crs = crs_from_config()
    rng = np.random.default_rng(42)
    x = rng.uniform(-250.0, 250.0, size=500)
    y = rng.uniform(-250.0, 250.0, size=500)
    lat, lon = local_to_wgs84(crs, x, y)
    xr, yr = wgs84_to_local(crs, lat, lon)
    err = np.hypot(xr - x, yr - y)
    assert err.max() < 0.1, f"worst round-trip error {err.max():.2e} m exceeds 0.1 m"


def test_round_trip_survives_serialisation_to_6_decimal_places() -> None:
    """GIS interchange rounds lat/lon to micro-degrees; the pair must still be
    recoverable to better than 0.1 m for a ~500 m mesh."""
    crs = crs_from_config()
    x, y = 137.5, -212.5
    lat, lon = local_to_wgs84(crs, x, y)
    lat_r, lon_r = round(float(lat), 6), round(float(lon), 6)
    xr, yr = wgs84_to_local(crs, lat_r, lon_r)
    assert np.hypot(xr - x, yr - y) < 0.1


def test_east_and_north_are_positive_in_the_right_direction() -> None:
    crs = crs_from_config()
    lat, lon = local_to_wgs84(crs, 100.0, 50.0)
    assert lon > crs.origin_lon_deg, "+x (east) must increase longitude"
    assert lat > crs.origin_lat_deg, "+y (north) must increase latitude"


def test_transform_parameters_are_stored_per_deployment() -> None:
    """§9.2: origin lat/lon and UTM zone live in the deployment config, not code."""
    from src.config import load_config

    cfg = load_config("physics")["crs"]
    assert {"origin_lat_deg", "origin_lon_deg", "utm_zone"} <= set(cfg)


def test_crs_from_config_rejects_bad_latitude() -> None:
    import src.geospatial.crs as crs_mod

    real = crs_mod.load_config

    def fake(name: str):
        data = real(name)
        if name == "physics":
            data = {**data, "crs": {**data["crs"], "origin_lat_deg": 95.0}}
        return data

    crs_mod.load_config = fake
    try:
        with pytest.raises(CRSError):
            crs_mod.crs_from_config()
    finally:
        crs_mod.load_config = real


def test_crs_from_config_rejects_bad_zone() -> None:
    import src.geospatial.crs as crs_mod

    real = crs_mod.load_config

    def fake(name: str):
        data = real(name)
        if name == "physics":
            data = {**data, "crs": {**data["crs"], "utm_zone": 99}}
        return data

    crs_mod.load_config = fake
    try:
        with pytest.raises(CRSError):
            crs_mod.crs_from_config()
    finally:
        crs_mod.load_config = real
