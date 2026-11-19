#!/usr/bin/env python3
"""Fuse repeated traversals into one georeferenced mine DEM and terrain mesh."""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_mine.build_stage1 import METERS, make_dem, write_obj


TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)


def rigid_utm_alignment(scene: Path):
    """Fit local tangent XY to direct UTM, including meridian convergence."""
    gps = np.load(scene / "gps_fix.npy")
    good = np.isfinite(gps[:, [0, 2, 3, 4]]).all(1) & (gps[:, 5] >= 0)
    good &= (gps[:, 2] > 40) & (gps[:, 2] < 50)
    good &= (gps[:, 3] > 110) & (gps[:, 3] < 120)
    gps = gps[good]
    order = np.argsort(gps[:, 0]); gps = gps[order]
    _, keep = np.unique(gps[:, 0], return_index=True); gps = gps[np.sort(keep)]
    lat0, lon0, alt0 = gps[0, 2:5]
    local = np.column_stack([
        (gps[:, 3] - lon0) * np.cos(np.deg2rad(lat0)) * METERS,
        (gps[:, 2] - lat0) * METERS,
    ])
    east, north = TO_UTM.transform(gps[:, 3], gps[:, 2])
    utm = np.column_stack([east, north])
    x0, y0 = local.mean(0), utm.mean(0)
    u, _, vt = np.linalg.svd((local - x0).T @ (utm - y0))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = y0 - x0 @ rotation.T
    fitted = local @ rotation.T + translation
    residual = np.linalg.norm(fitted - utm, axis=1)
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return rotation, translation, float(alt0), {
        "rotation_deg": angle,
        "gps_fit_median_m": float(np.median(residual)),
        "gps_fit_p95_m": float(np.percentile(residual, 95)),
        "gps_fit_max_m": float(residual.max()),
        "utm_origin_east_north": TO_UTM.transform(lon0, lat0),
    }


def load_scene(root, data_root, scene_id, reference_utm, reference_alt):
    folder = root / f"scene_{scene_id}_closed_loop"
    report = json.loads((folder / "report.json").read_text())
    rotation, translation, scene_alt, alignment = rigid_utm_alignment(
        data_root / f"scene_{scene_id}")
    points = np.load(folder / "global_points.npy", mmap_mode="r")
    points = np.asarray(points, np.float32).copy()
    points[:, :2] = points[:, :2] @ rotation.T + translation - reference_utm
    points[:, 2] += scene_alt - reference_alt
    route = np.genfromtxt(folder / "road_centerline.csv", delimiter=",",
                          names=True, dtype=np.float64)
    xyz = np.column_stack([route["east_m"], route["north_m"], route["up_m"]])
    xyz[:, :2] = xyz[:, :2] @ rotation.T + translation - reference_utm
    xyz[:, 2] += scene_alt - reference_alt
    return points, xyz, report, alignment


def preview(path, dem, count, lo, resolution, routes):
    valid = np.isfinite(dem)
    image = np.zeros((*dem.shape, 3), np.uint8)
    if valid.any():
        low, high = np.percentile(dem[valid], [2, 98])
        norm = np.clip((dem - low) / max(high - low, 1e-6), 0, 1)
        color = cv2.applyColorMap((np.nan_to_num(norm) * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
        image[valid] = color[valid]
    image[count < 3] //= 3
    palette = [(255, 255, 255), (255, 255, 0), (255, 0, 255),
               (0, 255, 255), (255, 160, 80), (80, 255, 160), (160, 80, 255)]
    for index, route in enumerate(routes):
        uv = np.column_stack(((route[:, 0] - lo[0]) / resolution,
                              (route[:, 1] - lo[1]) / resolution)).astype(np.int32)
        cv2.polylines(image, [uv.reshape(-1, 1, 2)], False,
                      palette[index % len(palette)], max(1, int(2 / resolution)),
                      cv2.LINE_AA)
    image = cv2.flip(image, 0)
    if max(image.shape[:2]) > 3000:
        scale = 3000 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=Path("carla_mine/outputs"))
    parser.add_argument("--data-root", type=Path,
                        default=Path("/media/ivan/Elements SE/06-28"))
    parser.add_argument("--output", type=Path,
                        default=Path("carla_mine/outputs/m0_combined_dem"))
    parser.add_argument("--scenes", default="0,1,2,3,5,6,7")
    parser.add_argument("--reference-scene", type=int, default=0)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--padding", type=float, default=2.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scene_ids = [int(value) for value in args.scenes.split(",")]
    reference_report = json.loads(
        (args.maps / f"scene_{args.reference_scene}_closed_loop/report.json").read_text()
    )
    reference = np.asarray(
        reference_report["local_geodetic_origin_lat_lon_alt"], np.float64
    )
    reference_utm = np.asarray(TO_UTM.transform(reference[1], reference[0]), np.float64)

    clouds, routes, scene_reports = [], [], {}
    for scene_id in scene_ids:
        points, route, report, alignment = load_scene(
            args.maps, args.data_root, scene_id, reference_utm, reference[2])
        clouds.append(points)
        routes.append(route)
        scene_reports[f"M0S{scene_id}"] = {
            "input_points": int(len(points)),
            "utm_alignment": alignment,
            "source_origin_lat_lon_alt": report["local_geodetic_origin_lat_lon_alt"],
        }
        print(f"M0S{scene_id}: {len(points):,} points, "
              f"UTM rotation {alignment['rotation_deg']:.6f} deg, "
              f"P95 residual {alignment['gps_fit_p95_m']:.3f} m",
              flush=True)

    points = np.concatenate(clouds, axis=0)
    dem, count, confidence, lo = make_dem(points, args.resolution, args.padding)

    voxel = np.floor(points / args.voxel_size).astype(np.int32)
    _, keep = np.unique(voxel, axis=0, return_index=True)
    global_points = points[np.sort(keep)].astype(np.float32)
    np.save(args.output / "global_points.npy", global_points)
    np.savez_compressed(
        args.output / "terrain_dem.npz",
        height=dem,
        count=count,
        confidence=confidence,
        origin_xy=lo,
        resolution_m=args.resolution,
        geodetic_origin=reference,
        source_scenes=np.asarray(scene_ids, np.int16),
    )
    np.savez_compressed(args.output / "routes.npz",
                        **{f"M0S{sid}": route.astype(np.float32)
                           for sid, route in zip(scene_ids, routes)})

    with (args.output / "route_origins.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "east_offset_m", "north_offset_m", "up_offset_m"])
        for scene_id in scene_ids:
            info = scene_reports[f"M0S{scene_id}"]["utm_alignment"]
            origin = info["utm_origin_east_north"]
            writer.writerow([f"M0S{scene_id}", origin[0]-reference_utm[0],
                             origin[1]-reference_utm[1],
                             scene_reports[f"M0S{scene_id}"]["source_origin_lat_lon_alt"][2]-reference[2]])

    vertices = write_obj(args.output / "terrain.obj", dem, confidence, lo,
                         args.resolution)
    preview(args.output / "terrain_dem.png", dem, count, lo, args.resolution, routes)

    valid = np.isfinite(dem)
    report = {
        "scenes": [f"M0S{sid}" for sid in scene_ids],
        "reference_scene": f"M0S{args.reference_scene}",
        "geodetic_origin_lat_lon_alt": reference.tolist(),
        "input_points": int(len(points)),
        "fused_point_voxels": int(len(global_points)),
        "voxel_size_m": args.voxel_size,
        "resolution_m": args.resolution,
        "dem_shape": [int(value) for value in dem.shape],
        "observed_cells": int(valid.sum()),
        "height_min_max_m": [float(np.nanmin(dem)), float(np.nanmax(dem))],
        "mesh_vertices": vertices,
        "scene_alignment": scene_reports,
        "outputs": ["terrain_dem.npz", "terrain_dem.png", "terrain.obj",
                    "global_points.npy", "routes.npz", "route_origins.csv"],
        "method": "Each traversal local tangent map is rigidly fitted to direct EPSG:32650 GPS coordinates, compensating UTM meridian convergence before voxel fusion.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
