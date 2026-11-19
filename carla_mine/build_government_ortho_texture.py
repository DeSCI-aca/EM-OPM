#!/usr/bin/env python3
"""Align the government RGB orthophoto with the CARLA hybrid DEM extent."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ortho", type=Path,
                    default=Path("carla_mine/outputs/tif_from_gover/mine.tif"))
    ap.add_argument("--dem", type=Path,
                    default=Path("carla_mine/outputs/carla_hybrid_dem/hybrid_dem.npz"))
    ap.add_argument("--satellite", type=Path,
                    default=Path("carla_mine/outputs/carla_hybrid_dem/carla_preview/MineHybridTexture_Satellite.png"))
    ap.add_argument("--satellite-geotiff", type=Path,
                    default=Path("carla_mine/external_imagery/sentinel2_20260628/crop_1km/visual.tif"))
    ap.add_argument("--routes", type=Path,
                    default=Path("carla_mine/outputs/carla_hybrid_dem/routes_carla.csv"))
    ap.add_argument("--output", type=Path,
                    default=Path("carla_mine/outputs/carla_hybrid_dem/government_ortho"))
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--feather-m", type=float, default=8.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dem = np.load(args.dem)
    z = dem["height"]
    west, south = float(dem["west"]), float(dem["south"])
    resolution = float(dem["resolution_m"])
    east = west + (z.shape[1] - 1) * resolution
    north = south + (z.shape[0] - 1) * resolution
    n = args.size
    transform = from_bounds(west, south, east, north, n, n)

    ortho = np.zeros((n, n, 3), np.uint8)
    coverage = np.zeros((n, n), np.uint8)
    with rasterio.open(args.ortho) as src:
        for band in range(3):
            reproject(rasterio.band(src, band + 1), ortho[:, :, band],
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:32650",
                      resampling=Resampling.lanczos, src_nodata=src.nodata,
                      dst_nodata=0)
        source_valid = np.ones((src.height, src.width), np.uint8)
        # This export declares no nodata but uses pure black outside the image.
        sample = src.read([1, 2, 3], out_shape=(3, src.height, src.width),
                          resampling=Resampling.nearest)
        source_valid[np.max(sample, axis=0) <= 2] = 0
        reproject(source_valid, coverage,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs="EPSG:32650",
                  resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)

    satellite = np.zeros((n, n, 3), np.uint8)
    satellite_valid = np.zeros((n, n), np.uint8)
    with rasterio.open(args.satellite_geotiff) as src:
        for band in range(3):
            reproject(rasterio.band(src, band + 1), satellite[:, :, band],
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:32650",
                      resampling=Resampling.bilinear, src_nodata=src.nodata,
                      dst_nodata=0)
        reproject(np.ones((src.height, src.width), np.uint8), satellite_valid,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs="EPSG:32650",
                  resampling=Resampling.nearest, dst_nodata=0)
    # Earth-tone terrain fallback where neither optical raster has coverage.
    terrain = cv2.resize(z, (n, n), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(terrain, [1, 99])
    hn = np.clip((terrain - lo) / max(hi - lo, 1e-6), 0, 1)
    fallback = np.stack([100 + 70 * hn, 85 + 70 * hn, 58 + 45 * hn], axis=-1)
    satellite[satellite_valid == 0] = np.clip(fallback[satellite_valid == 0], 0, 255)
    valid = coverage > 0
    inside = distance_transform_edt(valid)
    metres_per_pixel = (east - west) / n
    alpha = np.clip(inside * metres_per_pixel / args.feather_m, 0, 1)
    alpha = gaussian_filter(alpha.astype(np.float32), 0.8)
    result = np.clip(satellite * (1 - alpha[..., None]) + ortho * alpha[..., None],
                     0, 255).astype(np.uint8)

    png = args.output / "MineHybridTexture_GovernmentOrtho.png"
    mask_png = args.output / "MineHybridTexture_GovernmentCoverage.png"
    tif = args.output / "MineHybridTexture_GovernmentOrtho_utm50n.tif"
    cv2.imwrite(str(png), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_png), (alpha * 255).astype(np.uint8))
    profile = {"driver": "GTiff", "height": n, "width": n, "count": 3,
               "dtype": "uint8", "crs": "EPSG:32650", "transform": transform,
               "compress": "deflate", "tiled": True, "photometric": "RGB"}
    with rasterio.open(tif, "w", **profile) as dst:
        for band in range(3):
            dst.write(result[:, :, band], band + 1)

    route_preview = result.copy()
    route_groups = {}
    with args.routes.open() as stream:
        for row in csv.DictReader(stream):
            route_groups.setdefault(row["scene"], []).append(
                (float(row["x_east_m"]), float(row["y_north_m"])))
    palette = [(255, 40, 40), (40, 255, 255), (255, 40, 255),
               (40, 255, 40), (255, 160, 40), (80, 80, 255), (255, 255, 255)]
    extent_x, extent_y = east - west, north - south
    for index, (name, points) in enumerate(sorted(route_groups.items())):
        p = np.asarray(points)
        uv = np.column_stack([p[:, 0] / extent_x * (n - 1),
                              (1 - p[:, 1] / extent_y) * (n - 1)]).astype(np.int32)
        cv2.polylines(route_preview, [uv.reshape(-1, 1, 2)], False,
                      palette[index % len(palette)], 5, cv2.LINE_AA)
    route_path = args.output / "MineHybridTexture_GovernmentOrtho_routes.png"
    cv2.imwrite(str(route_path), cv2.cvtColor(route_preview, cv2.COLOR_RGB2BGR))

    yy, xx = np.nonzero(valid)
    comparison = args.output / "government_vs_sentinel_comparison.png"
    if len(xx):
        pad = 50
        x0, x1 = max(0, xx.min() - pad), min(n, xx.max() + pad + 1)
        y0, y1 = max(0, yy.min() - pad), min(n, yy.max() + pad + 1)
        before, after = satellite[y0:y1, x0:x1], result[y0:y1, x0:x1]
        scale = min(1.0, 900 / max(1, before.shape[0]))
        size = (max(1, int(before.shape[1] * scale)), max(1, int(before.shape[0] * scale)))
        before = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
        after = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
        panel = cv2.cvtColor(np.concatenate([before, after], 1), cv2.COLOR_RGB2BGR)
        cv2.putText(panel, "Sentinel-2", (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(panel, "Government 0.4 m ortho", (size[0] + 20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imwrite(str(comparison), panel)

    report = {
        "source": str(args.ortho), "source_crs": "EPSG:4548",
        "output_crs": "EPSG:32650", "texture_size": [n, n],
        "utm_bounds": [west, south, east, north],
        "government_valid_pixels": int(valid.sum()),
        "government_coverage_percent": float(100 * valid.mean()),
        "fallback": str(args.satellite_geotiff), "feather_m": args.feather_m,
        "outputs": [str(png), str(mask_png), str(tif), str(comparison), str(route_path)],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
