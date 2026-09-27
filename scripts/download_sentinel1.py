#!/usr/bin/env python3
"""T-057 — Acquire the Sentinel-1 SLC scene stack (PRD §18 step 2).

Reads the FIXED geometry from configs/insar.yaml (T-056): track 121,
descending, Jharia bbox, 2026 acquisition window. Two modes:

  inventory (default) — query the Copernicus Data Space Ecosystem OData
  catalogue anonymously and write data/raw/sentinel1/inventory_manifest.json
  listing every matching slice: date, track, geometry, product id, size.

  download — additionally fetch each product's SLC zip. This REQUIRES
  Copernicus credentials (env CDSE_USERNAME / CDSE_PASSWORD or --username /
  --password): product download is an authenticated endpoint. Tokens are
  requested from the CDSE identity service; nothing is hard-coded (NFR-6).

Usage:
  .venv/bin/python scripts/download_sentinel1.py                 # inventory
  CDSE_USERNAME=... CDSE_PASSWORD=... .venv/bin/python scripts/download_sentinel1.py --download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({id})/$value"

MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "sentinel1" / "inventory_manifest.json"
SCENE_DIR = REPO_ROOT / "data" / "raw" / "sentinel1"


def load_geometry() -> dict:
    with (REPO_ROOT / "configs" / "insar.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def bbox_wkt(geom: dict) -> str:
    b = geom["study_region"]["bbox"]
    poly = (
        f"{b['lon_min']} {b['lat_min']},{b['lon_max']} {b['lat_min']},"
        f"{b['lon_max']} {b['lat_max']},{b['lon_min']} {b['lat_max']},"
        f"{b['lon_min']} {b['lat_min']}"
    )
    return f"geography'SRID=4326;POLYGON(({poly}))'"


def search_catalogue(geom: dict, session: requests.Session) -> list[dict]:
    """One catalogue page-set for the configured track/geometry/window."""
    acq = geom["acquisition"]
    wkt = bbox_wkt(geom)
    start, end = acq["acquisition_window"]["start"], acq["acquisition_window"]["end"]
    filt = (
        f"contains(Name,'_SLC') and OData.CSC.Intersects(area={wkt})"
        f" and ContentDate/Start gt {start}T00:00:00.000Z"
        f" and ContentDate/Start lt {end}T00:00:00.000Z"
    )
    # filter to the fixed track client-side (server-side attribute filters are brittle)
    rows: list[dict] = []
    skip = 0
    while True:
        params = {
            "$filter": filt,
            "$expand": "Attributes",
            "$top": 500,
            "$skip": skip,
            "$count": "true",
        }
        r = session.get(CATALOGUE, params=params, timeout=90)
        r.raise_for_status()
        data = r.json()
        for p in data.get("value", []):
            attrs = {a["Name"]: a["Value"] for a in p.get("Attributes", [])}
            if attrs.get("relativeOrbitNumber") != acq["relative_orbit"]:
                continue
            if str(attrs.get("orbitDirection", "")).upper() != acq["orbit_direction"].upper():
                continue
            rows.append(
                {
                    "product_name": p["Name"],
                    "product_id": p["Id"],
                    "acquisition_date": str(attrs.get("beginningDateTime", ""))[:10],
                    "acquisition_start_utc": attrs.get("beginningDateTime"),
                    "relative_orbit": int(attrs["relativeOrbitNumber"]),
                    "orbit_direction": attrs.get("orbitDirection"),
                    "platform": attrs.get("platformSerialIdentifier"),
                    "polarisation": attrs.get("polarisationChannels"),
                    "size_bytes": p.get("ContentLength"),
                    "online": p.get("Online", True),
                    "geometry_wkt": wkt,
                }
            )
        total = int(data.get("@odata.count", 0))
        skip += 500
        if skip >= total or not data.get("value"):
            break
    return rows


def build_inventory(rows: list[dict], geom: dict) -> dict:
    acq = geom["acquisition"]
    by_date: dict[str, list[dict]] = {}
    for r in rows:
        by_date.setdefault(r["acquisition_date"], []).append(r)
    dates = sorted(by_date)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": "configs/insar.yaml",
        "study_region": geom["study_region"]["name"],
        "bbox": geom["study_region"]["bbox"],
        "relative_orbit": acq["relative_orbit"],
        "orbit_direction": acq["orbit_direction"],
        "acquisition_window": acq["acquisition_window"],
        "n_products": len(rows),
        "n_acquisition_dates": len(dates),
        "meets_minimum_stack": len(dates) >= 20,
        "platforms": sorted({r["platform"] for r in rows}),
        "dates": dates,
        "products": sorted(rows, key=lambda r: (r["acquisition_date"], r["product_name"])),
    }


def cdse_token(username: str, password: str, session: requests.Session) -> str:
    r = session.post(
        TOKEN_URL,
        data={
            "grant_type": "password",
            "username": username,
            "password": password,
            "client_id": "cdse-public",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def download_scenes(manifest: dict, username: str, password: str, limit: int | None) -> list[Path]:
    session = requests.Session()
    token = cdse_token(username, password, session)
    got: list[Path] = []
    products = manifest["products"] if limit is None else manifest["products"][:limit]
    for prod in products:
        dest = SCENE_DIR / prod["product_name"]
        if dest.exists():
            got.append(dest)
            continue
        for attempt in (1, 2, 3):
            try:
                with session.get(
                    DOWNLOAD_URL.format(id=prod["product_id"]),
                    headers={"Authorization": f"Bearer {token}"},
                    stream=True,
                    timeout=600,
                ) as r:
                    r.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with tmp.open("wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                tmp.rename(dest)
                got.append(dest)
                print(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
                break
            except Exception as exc:  # noqa: BLE001 — retry with backoff
                if attempt == 3:
                    print(f"  FAILED {prod['product_name']}: {exc}")
                else:
                    time.sleep(10 * attempt)
    return got


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="download scenes (requires credentials)")
    ap.add_argument("--limit", type=int, default=None, help="download only the first N scenes")
    ap.add_argument("--username", default=os.environ.get("CDSE_USERNAME", ""))
    ap.add_argument("--password", default=os.environ.get("CDSE_PASSWORD", ""))
    args = ap.parse_args()

    geom = load_geometry()
    session = requests.Session()
    print("Searching CDSE catalogue for T121 DESC SLC over Jharia …")
    rows = search_catalogue(geom, session)
    manifest = build_inventory(rows, geom)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"inventory: {manifest['n_products']} slices / {manifest['n_acquisition_dates']} dates "
        f"→ {MANIFEST_PATH.relative_to(REPO_ROOT)} (minimum stack met: {manifest['meets_minimum_stack']})"
    )

    if args.download:
        if not (args.username and args.password):
            print("ERROR: download requires CDSE credentials — set CDSE_USERNAME/CDSE_PASSWORD", file=sys.stderr)
            return 2
        paths = download_scenes(manifest, args.username, args.password, args.limit)
        for p in paths:
            entry = next(x for x in manifest["products"] if x["product_name"] == p.name)
            entry["downloaded"] = True
            entry["sha256"] = sha256(p)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"downloaded {len(paths)} scene(s) with checksums recorded in the manifest")
    else:
        print("inventory-only mode (add --download with CDSE credentials to fetch SLC zips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
