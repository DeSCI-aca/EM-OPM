#!/usr/bin/env python3
"""Refine the cropped CARLA terrain with UTM-aligned seven-run LiDAR."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, gaussian_filter, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_mine.build_combined_dem import rigid_utm_alignment


def grouped_quantile(flat, values, shape, quantile=0.25):
    order = np.argsort(flat)
    flat, values = flat[order], values[order]
    cells, starts, counts = np.unique(flat, return_index=True, return_counts=True)
    surface = np.array([np.quantile(values[s:s+n], quantile)
                        for s, n in zip(starts, counts)], np.float32)
    out = np.full(shape[0] * shape[1], np.nan, np.float32)
    num = np.zeros(shape[0] * shape[1], np.uint32)
    out[cells], num[cells] = surface, counts
    return out.reshape(shape), num.reshape(shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path,
                    default=Path("carla_mine/outputs/carla_route_buffer_1km_utm_aligned/hybrid_dem.npz"))
    ap.add_argument("--points", type=Path,
                    default=Path("carla_mine/outputs/m0_combined_dem_utm_aligned/global_points.npy"))
    ap.add_argument("--all-scene-points", action="store_true",
                    help="Use every point in the seven per-scene maps before cross-run voxel fusion")
    ap.add_argument("--scene-maps", type=Path, default=Path("carla_mine/outputs"))
    ap.add_argument("--data-root", type=Path, default=Path("/media/ivan/Elements SE/06-28"))
    ap.add_argument("--scenes", default="0,1,2,3,5,6,7")
    ap.add_argument("--target-resolution", type=float, default=2.0)
    ap.add_argument("--alignment-report", type=Path,
                    default=Path("carla_mine/outputs/m0_combined_dem_utm_aligned/report.json"))
    ap.add_argument("--routes", type=Path,
                    default=Path("carla_mine/outputs/carla_route_buffer_1km_utm_aligned/routes_carla.csv"))
    ap.add_argument("--output", type=Path,
                    default=Path("carla_mine/outputs/carla_route_buffer_1km_utm_aligned/lidar_optimized"))
    ap.add_argument("--min-points", type=int, default=3)
    ap.add_argument("--road-radius-m", type=float, default=14.0)
    ap.add_argument("--blend-distance-m", type=float, default=50.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    package = np.load(args.base)
    base = package["height"].astype(np.float32)
    west, south = float(package["west"]), float(package["south"])
    source_res, z0 = float(package["resolution_m"]), float(package["z_origin_m"])
    source_north = south + (base.shape[0] - 1) * source_res
    source_east = west + (base.shape[1] - 1) * source_res
    res = float(args.target_resolution)
    if abs(res - source_res) > 1e-9:
        width = int(round((source_east - west) / res)) + 1
        height = int(round((source_north - south) / res)) + 1
        base = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
    north = south + (base.shape[0] - 1) * res
    east = west + (base.shape[1] - 1) * res

    report = json.loads(args.alignment_report.read_text())
    lat0, lon0, alt0 = report["geodetic_origin_lat_lon_alt"]
    ref_e, ref_n = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True).transform(lon0, lat0)
    if args.all_scene_points:
        xyz_chunks = []
        for scene_id in [int(v) for v in args.scenes.split(",")]:
            rotation, translation, scene_alt, _ = rigid_utm_alignment(
                args.data_root / f"scene_{scene_id}")
            raw = np.load(args.scene_maps / f"scene_{scene_id}_closed_loop/global_points.npy",
                          mmap_mode="r")
            local_xy = np.asarray(raw[:, :2], np.float64)
            utm_xy = local_xy @ rotation.T + translation
            abs_z = np.asarray(raw[:, 2], np.float32) + scene_alt
            keep = ((utm_xy[:, 0] >= west) & (utm_xy[:, 0] <= east) &
                    (utm_xy[:, 1] >= south) & (utm_xy[:, 1] <= north) &
                    np.isfinite(abs_z))
            xyz_chunks.append(np.column_stack([utm_xy[keep], abs_z[keep]]))
            print(f"M0S{scene_id}: retained {keep.sum():,}/{len(raw):,} points", flush=True)
        all_xyz = np.concatenate(xyz_chunks)
        x, y, z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2].astype(np.float32)
        input_mode = "all seven per-scene map points before cross-run voxel fusion"
    else:
        points = np.load(args.points, mmap_mode="r")
        x = np.asarray(points[:, 0], np.float64) + ref_e
        y = np.asarray(points[:, 1], np.float64) + ref_n
        z = np.asarray(points[:, 2], np.float32) + float(alt0)
        input_mode = "0.5 m cross-run voxel-fused point map"
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    good &= (x >= west) & (x <= east) & (y >= south) & (y <= north)
    x, y, z = x[good], y[good], z[good]
    ix = np.floor((x - west) / res).astype(np.int32)
    # Base array is north-up.
    iy = np.floor((north - y) / res).astype(np.int32)
    inside = (ix >= 0) & (ix < base.shape[1]) & (iy >= 0) & (iy < base.shape[0])
    ix, iy, z = ix[inside], iy[inside], z[inside]
    lidar, counts = grouped_quantile(iy * base.shape[1] + ix, z, base.shape, 0.25)
    measured = np.isfinite(lidar) & (counts >= args.min_points)

    # Robustly remove isolated object/dust spikes, retaining persistent benches.
    nearest_ids = distance_transform_edt(~measured, return_distances=False, return_indices=True)
    filled = lidar[nearest_ids[0], nearest_ids[1]]
    local_median = median_filter(filled, size=3, mode="nearest")
    spike = measured & (np.abs(lidar - local_median) > 3.0) & (counts < 10)
    lidar[spike] = local_median[spike]

    # Rasterize the driven corridor. Repeated route coverage defines where a
    # smoother, vehicle-usable surface is preferred over scan-line roughness.
    route_mask = np.zeros(base.shape, np.uint8)
    groups = {}
    with args.routes.open() as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(row["scene"], []).append(
                [float(row["x_east_m"]), float(row["y_north_m"])])
    for points_xy in groups.values():
        q = np.asarray(points_xy)
        uv = np.column_stack([q[:, 0] / res,
                              (base.shape[0] - 1) - q[:, 1] / res]).astype(np.int32)
        cv2.polylines(route_mask, [uv.reshape(-1, 1, 2)], False, 1,
                      max(1, int(round(2 * args.road_radius_m / res))), cv2.LINE_AA)
    road = route_mask > 0

    smooth = gaussian_filter(filled, sigma=1.0)
    # Keep sharp mine benches; smooth only moderate-slope road cells.
    gy, gx = np.gradient(filled, res)
    moderate = np.hypot(gx, gy) < 0.35
    refined = lidar.copy()
    road_refine = measured & road & moderate
    refined[road_refine] = 0.35 * lidar[road_refine] + 0.65 * smooth[road_refine]
    nonroad = measured & ~road
    refined[nonroad] = 0.8 * lidar[nonroad] + 0.2 * local_median[nonroad]

    # Blend refined LiDAR into the existing satellite terrain, tapering outside
    # measured support so the crop has no hard vertical seam.
    distance_px, nearest = distance_transform_edt(~measured, return_indices=True)
    nearest_refined = refined[nearest[0], nearest[1]]
    weight = np.clip(1 - distance_px * res / args.blend_distance_m, 0, 1) ** 2
    weight[measured] = 1
    optimized = base * (1 - weight) + nearest_refined * weight

    transform = from_origin(west - res / 2, north + res / 2, res, res)
    profile = {"driver": "GTiff", "height": base.shape[0], "width": base.shape[1],
               "count": 1, "dtype": "float32", "crs": "EPSG:32650",
               "transform": transform, "compress": "deflate", "tiled": True}
    tif = args.output / "lidar_optimized_dem_utm50n.tif"
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(optimized.astype(np.float32), 1)
    np.savez_compressed(args.output / "hybrid_dem.npz",
                        height=optimized.astype(np.float32), resolution_m=res,
                        crs="EPSG:32650", west=west, south=south, z_origin_m=z0,
                        lidar_weight=weight.astype(np.float32), count=counts)

    valid_eval = measured & np.isfinite(refined)
    base_err = base[valid_eval] - lidar[valid_eval]
    opt_err = optimized[valid_eval] - lidar[valid_eval]
    correction = optimized - base
    summary = {
        "method": "Seven-run UTM-aligned LiDAR 25th-percentile ground surface; sparse-spike rejection; road-corridor edge-aware smoothing; taper to satellite base.",
        "input_mode": input_mode,
        "input_points_in_crop": int(len(z)),
        "measured_cells": int(measured.sum()),
        "removed_sparse_spikes": int(spike.sum()),
        "road_refined_cells": int(road_refine.sum()),
        "base_to_lidar_rmse_m": float(np.sqrt(np.mean(base_err ** 2))),
        "optimized_to_lidar_rmse_m": float(np.sqrt(np.mean(opt_err ** 2))),
        "correction_median_abs_m": float(np.median(np.abs(correction[weight > 0]))),
        "correction_p95_abs_m": float(np.percentile(np.abs(correction[weight > 0]), 95)),
        "bounds_utm": [west, south, east, north],
        "resolution_m": res,
        "output": str(tif),
    }
    (args.output / "report.json").write_text(json.dumps(summary, indent=2))

    lo, hi = np.percentile(optimized, [1, 99])
    preview = cv2.applyColorMap((np.clip((optimized-lo)/(hi-lo), 0, 1)*255).astype(np.uint8),
                                cv2.COLORMAP_TURBO)
    preview[road] = (0.7 * preview[road] + 0.3 * np.array([255,255,255])).astype(np.uint8)
    cv2.imwrite(str(args.output / "lidar_optimized_preview.png"), preview)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
