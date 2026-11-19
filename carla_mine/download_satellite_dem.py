#!/usr/bin/env python3
"""Download and crop Copernicus GLO-30 DEM tiles around the seven mine runs."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds as window_from_bounds


SCENES = (0, 1, 2, 3, 5, 6, 7)
URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{lat}_{lon}_DEM/"
    "Copernicus_DSM_COG_10_{lat}_{lon}_DEM.tif"
)


def load_gps(root: Path):
    runs, all_valid = {}, []
    for scene in SCENES:
        path = root / f"scene_{scene}" / "gps_fix.npy"
        x = np.load(path)
        valid = (
            np.isfinite(x[:, 0]) & np.isfinite(x[:, 2]) & np.isfinite(x[:, 3])
            & (x[:, 2] > 40) & (x[:, 2] < 50)
            & (x[:, 3] > 110) & (x[:, 3] < 120)
        )
        y = x[valid]
        if not len(y):
            raise RuntimeError(f"No valid fixes in {path}")
        runs[f"M0S{scene}"] = y
        all_valid.append(y)
    return runs, np.concatenate(all_valid)


def tile_name(lat: int, lon: int):
    ns = f"N{lat:02d}_00" if lat >= 0 else f"S{abs(lat):02d}_00"
    ew = f"E{lon:03d}_00" if lon >= 0 else f"W{abs(lon):03d}_00"
    return ns, ew


def download(url: str, path: Path):
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("carla_mine/external_dem/copernicus_glo30"))
    ap.add_argument("--padding-m", type=float, default=0.0)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    runs, gps = load_gps(args.data_root)
    south, north = float(gps[:, 2].min()), float(gps[:, 2].max())
    west, east = float(gps[:, 3].min()), float(gps[:, 3].max())
    mid_lat = (south + north) / 2
    pad_lat = args.padding_m / 111_320.0
    pad_lon = args.padding_m / (111_320.0 * math.cos(math.radians(mid_lat)))
    bounds = (west - pad_lon, south - pad_lat, east + pad_lon, north + pad_lat)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for lat in range(math.floor(bounds[1]), math.floor(bounds[3]) + 1):
        for lon in range(math.floor(bounds[0]), math.floor(bounds[2]) + 1):
            ns, ew = tile_name(lat, lon)
            stem = f"Copernicus_DSM_COG_10_{ns}_{ew}_DEM"
            path = out / f"{stem}.tif"
            if not args.no_download:
                download(URL.format(lat=ns, lon=ew), path)
            if path.exists():
                tiles.append(path)
    if not tiles:
        raise RuntimeError("No DEM tiles available")

    srcs = [rasterio.open(p) for p in tiles]
    try:
        mosaic, transform = merge(srcs, bounds=bounds)
        profile = srcs[0].profile.copy()
        profile.update(width=mosaic.shape[2], height=mosaic.shape[1], transform=transform,
                       compress="deflate", tiled=True)
        crop = out / "mine_dem_glo30_wgs84.tif"
        with rasterio.open(crop, "w", **profile) as dst:
            dst.write(mosaic)

        utm = "EPSG:32650"
        crop_bounds = rasterio.transform.array_bounds(
            mosaic.shape[1], mosaic.shape[2], transform
        )
        dst_transform, width, height = calculate_default_transform(
            profile["crs"], utm, profile["width"], profile["height"], *crop_bounds
        )
        utm_profile = profile.copy()
        utm_profile.update(crs=utm, transform=dst_transform, width=width, height=height,
                           nodata=-9999.0)
        utm_path = out / "mine_dem_glo30_utm50n.tif"
        utm_dem = np.full((height, width), -9999.0, dtype=np.float32)
        reproject(source=mosaic[0], destination=utm_dem,
                  src_transform=transform, src_crs=profile["crs"],
                  dst_transform=dst_transform, dst_crs=utm,
                  dst_nodata=-9999.0, resampling=Resampling.bilinear)
        with rasterio.open(utm_path, "w", **utm_profile) as dst:
            dst.write(utm_dem, 1)
    finally:
        for src in srcs:
            src.close()

    report = {
        "source": "Copernicus DEM GLO-30 (DSM), 2021 release; TanDEM-X acquisition 2011-2015",
        "gps_data_root": str(args.data_root),
        "scenes": [f"M0S{x}" for x in SCENES],
        "gps_bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "crop_bounds_wgs84": {"west": bounds[0], "south": bounds[1], "east": bounds[2], "north": bounds[3]},
        "gps_time_utc": {
            "start": datetime.fromtimestamp(float(gps[:, 0].min()), timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(float(gps[:, 0].max()), timezone.utc).isoformat(),
        },
        "tiles": [p.name for p in tiles],
        "outputs": [crop.name, utm_path.name],
        "vertical_datum": "EGM2008 geoid heights; verify GPS altitude datum before vertical alignment",
    }
    (out / "dem_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
