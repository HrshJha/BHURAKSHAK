"""§9.2 dual coordinate reference system (T-032).

PRD §9.2:
- **Mesh-local coordinates** — metres, origin at a fixed reference — used for
  all inter-node distance/strain/gradient calculations.
- **Global coordinates** — every node also stores a WGS84 (lat, lon) pair for
  GIS mapping and alignment with Sentinel-1/DGPS/NISAR products.
- **Transformation** — a fixed local tangent-plane converts between the two;
  the transformation parameters (origin lat/lon, UTM zone) are stored per
  deployment so synthetic and real datasets remain comparable.

Implementation: pure-NumPy equirectangular local tangent plane (§9.2 explicitly
allows "a simple equirectangular approximation for small panel extents" — a
400-node, 500 m mesh is far below the kilometre scale where distortion matters;
pyproj is deliberately not a dependency). Transformation parameters come from
the ``crs`` block of configs/physics.yaml, stored per deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import load_config

__all__ = [
    "CRSError",
    "CRSDefinition",
    "crs_from_config",
    "local_to_wgs84",
    "wgs84_to_local",
]

#: A mesh-local point: (x, y) metres.
LocalXY = tuple[float, float]
#: A geodetic point: (lat, lon) degrees, WGS84.
WGS84 = tuple[float, float]


class CRSError(ValueError):
    """Raised on invalid CRS definitions or out-of-domain inputs."""


@dataclass(frozen=True)
class CRSDefinition:
    """Per-deployment transformation parameters (§9.2)."""

    origin_lat_deg: float
    origin_lon_deg: float
    utm_zone: int
    earth_radius_m: float
    projection_model: str

    @property
    def metres_per_deg_lat(self) -> float:
        return float(np.pi / 180.0) * self.earth_radius_m


def crs_from_config() -> CRSDefinition:
    """Load the deployment's CRS parameters from configs/physics.yaml (``crs``)."""
    cfg = load_config("physics")["crs"]
    lat, lon = float(cfg["origin_lat_deg"]), float(cfg["origin_lon_deg"])
    if not -90.0 <= lat <= 90.0:
        raise CRSError(f"origin_lat_deg out of range: {lat}")
    if not -180.0 <= lon <= 180.0:
        raise CRSError(f"origin_lon_deg out of range: {lon}")
    zone = int(cfg["utm_zone"])
    if not 1 <= zone <= 60:
        raise CRSError(f"utm_zone out of range: {zone}")
    model = str(cfg.get("projection_model", "equirectangular"))
    if model != "equirectangular":
        raise CRSError(f"unsupported projection_model {model!r} (only 'equirectangular')")
    return CRSDefinition(
        origin_lat_deg=lat,
        origin_lon_deg=lon,
        utm_zone=zone,
        earth_radius_m=float(cfg["earth_radius_m"]),
        projection_model=model,
    )


def local_to_wgs84(crs: CRSDefinition, x: float | np.ndarray, y: float | np.ndarray):
    """Mesh-local metres → WGS84 (lat, lon) degrees (equirectangular tangent plane)."""
    m_per_deg_lat = crs.metres_per_deg_lat
    m_per_deg_lon = m_per_deg_lat * np.cos(np.deg2rad(crs.origin_lat_deg))
    if m_per_deg_lon <= 0:
        raise CRSError("degenerate metres-per-degree-longitude (polar origin?)")
    lat = crs.origin_lat_deg + np.asarray(y) / m_per_deg_lat
    lon = crs.origin_lon_deg + np.asarray(x) / m_per_deg_lon
    return lat, lon


def wgs84_to_local(crs: CRSDefinition, lat: float | np.ndarray, lon: float | np.ndarray):
    """WGS84 (lat, lon) degrees → mesh-local metres (inverse tangent plane)."""
    m_per_deg_lat = crs.metres_per_deg_lat
    m_per_deg_lon = m_per_deg_lat * np.cos(np.deg2rad(crs.origin_lat_deg))
    x = (np.asarray(lon) - crs.origin_lon_deg) * m_per_deg_lon
    y = (np.asarray(lat) - crs.origin_lat_deg) * m_per_deg_lat
    return x, y
