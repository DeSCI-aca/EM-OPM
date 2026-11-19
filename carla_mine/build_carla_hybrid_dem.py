#!/usr/bin/env python3
"""Fuse seven-run LiDAR terrain with Copernicus DEM for CARLA/Unreal import."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter


def robust_offset(lidar, satellite, mask):
    delta = (lidar - satellite)[mask]
    delta = delta[np.isfinite(delta)]
    median = float(np.median(delta))
    mad = float(np.median(np.abs(delta - median)))
    keep = np.abs(delta - median) <= max(3.0, 3.0 * 1.4826 * mad)
    return float(np.median(delta[keep])), mad, int(keep.sum()), int(len(delta))


def hillshade(z, resolution):
    gy, gx = np.gradient(z, resolution)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.deg2rad(315), np.deg2rad(45)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip((shade + 1) * 127.5, 0, 255).astype(np.uint8)


def write_preview(path, dem, source, routes, west, south, resolution):
    shade = hillshade(dem, resolution)
    lo, hi = np.percentile(dem, [1, 99])
    color = cv2.applyColorMap((np.clip((dem - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8),
                              cv2.COLORMAP_TURBO)
    color = (0.72 * color + 0.28 * shade[..., None]).astype(np.uint8)
    color[source == 2] = (0.85 * color[source == 2] + 0.15 * np.array([255, 255, 255])).astype(np.uint8)
    color[source == 1] = (0.85 * color[source == 1] + 0.15 * np.array([255, 200, 0])).astype(np.uint8)
    for route in routes:
        u = ((route[:, 0] - west) / resolution).astype(np.int32)
        v = ((route[:, 1] - south) / resolution).astype(np.int32)
        pts = np.column_stack([u, dem.shape[0] - 1 - v])
        cv2.polylines(color, [pts.reshape(-1, 1, 2)], False, (20, 20, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), color)


def write_obj(path, dem, resolution, stride, z_origin):
    z = dem[::stride, ::stride]
    h, w = z.shape
    step = resolution * stride
    with path.open("w") as f:
        f.write("# CARLA-local terrain; x=east, y=north, z relative to z_origin\n")
        for iy in range(h):
            for ix in range(w):
                f.write(f"v {ix*step:.3f} {iy*step:.3f} {z[iy,ix]-z_origin:.3f}\n")
        for iy in range(h - 1):
            row, nxt = iy * w + 1, (iy + 1) * w + 1
            for ix in range(w - 1):
                a, b, c, d = row + ix, row + ix + 1, nxt + ix, nxt + ix + 1
                f.write(f"f {a} {b} {d}\nf {a} {d} {c}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lidar-dem", type=Path,
                    default=Path("carla_mine/outputs/m0_combined_dem/terrain_dem.npz"))
    ap.add_argument("--routes", type=Path,
                    default=Path("carla_mine/outputs/m0_combined_dem/routes.npz"))
    ap.add_argument("--satellite-dem", type=Path,
                    default=Path("carla_mine/external_dem/copernicus_glo30/mine_dem_glo30_utm50n.tif"))
    ap.add_argument("--output", type=Path,
                    default=Path("carla_mine/outputs/carla_hybrid_dem"))
    ap.add_argument("--size", type=int, default=2017)
    ap.add_argument("--resolution", type=float, default=2.0)
    ap.add_argument("--blend-distance", type=float, default=60.0)
    ap.add_argument("--obj-stride", type=int, default=4)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pkg = np.load(args.lidar_dem)
    local_dem = pkg["height"].astype(np.float32)
    local_conf = pkg["confidence"].astype(np.float32)
    local_count = pkg["count"]
    local_lo = pkg["origin_xy"].astype(np.float64)
    local_res = float(pkg["resolution_m"])
    lat0, lon0, alt0 = pkg["geodetic_origin"].astype(np.float64)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    ref_e, ref_n = to_utm.transform(lon0, lat0)

    route_pkg = np.load(args.routes)
    local_routes = [route_pkg[k].astype(np.float64) for k in route_pkg.files]
    routes = []
    for route in local_routes:
        q = route.copy()
        q[:, 0] += ref_e
        q[:, 1] += ref_n
        q[:, 2] += alt0
        routes.append(q)
    route_xy = np.concatenate([q[:, :2] for q in routes])
    center = (route_xy.min(0) + route_xy.max(0)) / 2
    extent = (args.size - 1) * args.resolution
    west, south = center - extent / 2
    north = south + extent
    # Treat height samples as Landscape vertices. GeoTIFF pixel centres therefore
    # coincide with the first/last terrain vertices, rather than pixel edges.
    target_transform = from_origin(west - args.resolution / 2,
                                   north + args.resolution / 2,
                                   args.resolution, args.resolution)
    shape = (args.size, args.size)

    satellite = np.full(shape, np.nan, np.float32)
    with rasterio.open(args.satellite_dem) as src:
        reproject(rasterio.band(src, 1), satellite, src_transform=src.transform,
                  src_crs=src.crs, dst_transform=target_transform, dst_crs="EPSG:32650",
                  src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
    if not np.isfinite(satellite).all():
        valid = np.isfinite(satellite)
        _, ind = distance_transform_edt(~valid, return_indices=True)
        satellite[~valid] = satellite[ind[0][~valid], ind[1][~valid]]

    src_transform = from_origin(ref_e + local_lo[0] - local_res / 2,
                                ref_n + local_lo[1] + (local_dem.shape[0] - 0.5) * local_res,
                                local_res, local_res)
    lidar = np.full(shape, np.nan, np.float32)
    conf = np.zeros(shape, np.float32)
    counts = np.zeros(shape, np.float32)
    lidar_abs = np.flipud(local_dem + alt0)
    reproject(lidar_abs, lidar, src_transform=src_transform, src_crs="EPSG:32650",
              dst_transform=target_transform, dst_crs="EPSG:32650",
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
    reproject(np.flipud(local_conf), conf, src_transform=src_transform, src_crs="EPSG:32650",
              dst_transform=target_transform, dst_crs="EPSG:32650",
              resampling=Resampling.average)
    reproject(np.flipud(local_count.astype(np.float32)), counts, src_transform=src_transform,
              src_crs="EPSG:32650", dst_transform=target_transform, dst_crs="EPSG:32650",
              resampling=Resampling.average)
    measured = np.isfinite(lidar) & (counts >= 1)
    align_mask = measured & (conf >= 0.35)
    vertical_offset, mad, aligned, available = robust_offset(lidar, satellite, align_mask)
    satellite += vertical_offset

    distance_px, nearest = distance_transform_edt(~measured, return_indices=True)
    nearest_lidar = lidar[nearest[0], nearest[1]]
    smooth_lidar = gaussian_filter(nearest_lidar, sigma=1.0)
    smooth_lidar[measured] = lidar[measured]
    distance_m = distance_px * args.resolution
    weight = np.clip(1.0 - distance_m / args.blend_distance, 0, 1) ** 2
    weight[measured] = 1.0
    fused = satellite * (1 - weight) + smooth_lidar * weight
    source = np.zeros(shape, np.uint8)
    source[weight > 0] = 1
    source[measured] = 2

    z_min, z_max = float(np.floor(fused.min() - 2)), float(np.ceil(fused.max() + 2))
    z_center = (z_min + z_max) / 2
    z_scale = (z_max - z_min) / 5.12
    height16 = np.clip(np.rint(32768 + (fused - z_center) * 12800 / z_scale), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(args.output / "carla_heightmap_2017.png"), np.flipud(height16))

    profile = {"driver": "GTiff", "height": args.size, "width": args.size,
               "count": 1, "dtype": "float32", "crs": "EPSG:32650",
               "transform": target_transform, "compress": "deflate", "tiled": True,
               "nodata": -9999.0}
    with rasterio.open(args.output / "hybrid_dem_utm50n.tif", "w", **profile) as dst:
        dst.write(fused.astype(np.float32), 1)
    np.savez_compressed(args.output / "hybrid_dem.npz", height=fused.astype(np.float32),
                        lidar_weight=weight.astype(np.float32), source=source,
                        transform=np.asarray(target_transform)[:6], resolution_m=args.resolution,
                        crs="EPSG:32650", west=west, south=south, z_origin_m=z_center)
    write_preview(args.output / "hybrid_dem_preview.png", fused, source, routes,
                  west, south, args.resolution)
    write_obj(args.output / "terrain_preview_8m.obj", np.flipud(fused), args.resolution,
              args.obj_stride, z_center)

    with (args.output / "routes_carla.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "x_east_m", "y_north_m", "z_relative_m"])
        for name, route in zip(route_pkg.files, routes):
            for p in route:
                writer.writerow([name, p[0] - west, p[1] - south, p[2] - z_center])

    report = {
        "method": "Copernicus GLO-30 base, vertically aligned to seven-run LiDAR; measured LiDAR cells override the base and taper to satellite terrain over the configured blend distance.",
        "size_pixels": [args.size, args.size], "resolution_m": args.resolution,
        "extent_m": [extent, extent], "crs": "EPSG:32650",
        "utm_bounds": {"west": west, "south": south, "east": west + extent, "north": north},
        "height_min_max_m_egm2008_approx": [float(fused.min()), float(fused.max())],
        "lidar_vertical_reference_alt_m": float(alt0),
        "satellite_vertical_alignment_offset_m": vertical_offset,
        "alignment_mad_m": mad, "alignment_cells_kept_total": [aligned, available],
        "measured_lidar_cells": int(measured.sum()),
        "blend_cells": int(((source == 1)).sum()),
        "satellite_only_cells": int((source == 0).sum()),
        "unreal_import": {
            "heightmap": "carla_heightmap_2017.png", "landscape_xy_scale_cm": args.resolution * 100,
            "landscape_z_scale": z_scale, "landscape_actor_z_m": z_center,
            "heightmap_flip": "PNG row 0 is the south edge, matching Unreal Landscape import orientation used here; verify orientation with routes_carla.csv before packaging.",
        },
        "outputs": ["carla_heightmap_2017.png", "hybrid_dem_utm50n.tif", "hybrid_dem.npz",
                    "hybrid_dem_preview.png", "terrain_preview_8m.obj", "routes_carla.csv"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
