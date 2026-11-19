#!/usr/bin/env python3
"""Colorize the seven-traversal DEM with synchronized six-camera imagery."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_mine.build_stage1 import Trajectory, select_scans
from em_opm.extract_dynamic_actors import (P_CL, R_CL, R_I0, T_I0,
                                            intrinsics, nearest, stamp)


def cached_files(folder: Path, suffix: str, cache: Path):
    """Index timestamp-named sensor files once; external disks make glob costly."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        names = [line.strip() for line in cache.read_text().splitlines() if line.strip()]
        files = [folder / name for name in names]
        if files and files[0].exists() and files[-1].exists():
            return files
    print(f"Indexing {folder} ...", flush=True)
    entries = []
    with os.scandir(folder) as iterator:
        for entry in iterator:
            if entry.is_file() and entry.name.endswith(suffix):
                try:
                    entries.append((float(Path(entry.name).stem), entry.name))
                except ValueError:
                    pass
    entries.sort(key=lambda item: item[0])
    cache.write_text("\n".join(name for _, name in entries) + "\n")
    print(f"Indexed {len(entries):,} files -> {cache}", flush=True)
    return [folder / name for _, name in entries]


def cached_scans(scene: Path, step_s: float, max_scans: int, cache_root: Path):
    files = cached_files(scene / "LiDAR", ".npy", cache_root / "lidar.txt")
    out, last = [], -np.inf
    for path in files:
        t = float(path.stem)
        if t - last >= step_s:
            out.append(path)
            last = t
            if max_scans and len(out) >= max_scans:
                break
    return out


def project(points_l: np.ndarray, cam: int, k: np.ndarray, dist: np.ndarray):
    """Return original point indices and distorted image coordinates."""
    p0 = (R_CL.T @ (points_l - P_CL).T).T
    pc = (R_I0[cam].T @ (p0 - T_I0[cam]).T).T
    ids = np.flatnonzero(pc[:, 2] > 0.5)
    if not len(ids):
        return ids, np.empty((0, 2), np.float64)
    uv, _ = cv2.projectPoints(pc[ids], np.zeros(3), np.zeros(3), k, dist)
    return ids, uv[:, 0]


def load_surface_points(path: Path, traj: Trajectory, offset: np.ndarray,
                        dem: np.ndarray, lo: np.ndarray, resolution: float,
                        max_points: int, height_tolerance: float):
    raw = np.load(path, mmap_mode="r")
    xyz = np.asarray(raw[:, :3], np.float64)
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    good = np.isfinite(xyz).all(1) & (radius >= 3) & (radius <= 80)
    good &= (xyz[:, 2] > -8) & (xyz[:, 2] < 20)
    xyz = xyz[good]
    if len(xyz) > max_points:
        ids = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
        xyz = xyz[ids]
    pos, rot = traj.poses([float(path.stem)])
    world = rot.apply(xyz) + pos[0] + offset
    ix = np.floor((world[:, 0] - lo[0]) / resolution).astype(np.int32)
    iy = np.floor((world[:, 1] - lo[1]) / resolution).astype(np.int32)
    inside = (ix >= 0) & (iy >= 0) & (ix < dem.shape[1]) & (iy < dem.shape[0])
    ids = np.flatnonzero(inside)
    if not len(ids):
        return xyz[:0], world[:0], ix[:0], iy[:0]
    surface = np.isfinite(dem[iy[ids], ix[ids]])
    surface &= np.abs(world[ids, 2] - dem[iy[ids], ix[ids]]) <= height_tolerance
    ids = ids[surface]
    return xyz[ids], world[ids], ix[ids], iy[ids]


def write_colored_obj(path: Path, dem: np.ndarray, confidence: np.ndarray,
                      rgb: np.ndarray, lo: np.ndarray, resolution: float,
                      min_confidence: float = 0.25):
    valid = np.isfinite(dem) & (confidence >= min_confidence)
    ids = -np.ones(dem.shape, np.int64)
    yy, xx = np.nonzero(valid)
    ids[yy, xx] = np.arange(len(xx)) + 1
    with path.open("w") as stream:
        stream.write("# local ENU terrain; vertex fields are x y z r g b\n")
        for y, x in zip(yy, xx):
            r, g, b = rgb[y, x].astype(np.float64) / 255.0
            stream.write(f"v {lo[0]+x*resolution:.3f} {lo[1]+y*resolution:.3f} "
                         f"{dem[y,x]:.3f} {r:.4f} {g:.4f} {b:.4f}\n")
        for y in range(dem.shape[0] - 1):
            for x in range(dem.shape[1] - 1):
                a, b, c, d = ids[y, x], ids[y, x+1], ids[y+1, x], ids[y+1, x+1]
                if min(a, b, c, d) > 0 and np.ptp(dem[y:y+2, x:x+2]) <= max(2.0, 3*resolution):
                    stream.write(f"f {a} {b} {d}\nf {a} {d} {c}\n")
    return int(valid.sum())


def save_preview(path: Path, rgb: np.ndarray, valid: np.ndarray):
    image = rgb.copy()
    image[~valid] = 0
    image = np.flipud(image)
    if max(image.shape[:2]) > 3000:
        scale = 3000 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("/media/ivan/Elements SE/06-28"))
    ap.add_argument("--dem-dir", type=Path, default=Path("carla_mine/outputs/m0_combined_dem"))
    ap.add_argument("--scenes", default="0,1,2,3,5,6,7")
    ap.add_argument("--step-s", type=float, default=60.0)
    ap.add_argument("--max-scans", type=int, default=0)
    ap.add_argument("--max-points", type=int, default=8000)
    ap.add_argument("--max-camera-dt", type=float, default=0.12)
    ap.add_argument("--height-tolerance", type=float, default=1.25)
    ap.add_argument("--fill-radius", type=float, default=6.0,
                    help="Maximum nearest-color fill distance in metres")
    ap.add_argument("--index-root", type=Path,
                    default=Path("carla_mine/cache/sensor_indices"))
    args = ap.parse_args()

    package = np.load(args.dem_dir / "terrain_dem.npz")
    dem = package["height"]
    confidence = package["confidence"]
    lo = package["origin_xy"].astype(np.float64)
    resolution = float(package["resolution_m"])
    alignment = json.loads((args.dem_dir / "report.json").read_text())["scene_alignment"]
    sums = np.zeros((*dem.shape, 3), np.float64)
    counts = np.zeros(dem.shape, np.uint32)
    scene_reports = {}

    for scene_id in [int(v) for v in args.scenes.split(",")]:
        scene = args.data_root / f"scene_{scene_id}"
        traj = Trajectory(scene)
        offset = np.asarray(alignment[f"M0S{scene_id}"]["offset_to_reference_m"], np.float64)
        cache_root = args.index_root / f"scene_{scene_id}"
        scans = cached_scans(scene, args.step_s, args.max_scans, cache_root)
        print(f"M0S{scene_id}: selected {len(scans)} LiDAR scans at {args.step_s:g}s", flush=True)
        cameras = []
        for cam in range(6):
            folder = scene / f"mav0/cam{cam}/data"
            files = cached_files(folder, ".png", cache_root / f"cam{cam}.txt")
            cameras.append((files, np.asarray([stamp(p) for p in files]), *intrinsics(scene, cam)))
        used_scans = used_images = projected = 0
        for scan_index, scan in enumerate(scans, 1):
            t = float(scan.stem)
            if t < traj.start or t > traj.end:
                continue
            local, _, ix, iy = load_surface_points(
                scan, traj, offset, dem, lo, resolution,
                args.max_points, args.height_tolerance)
            if not len(local):
                continue
            used_scans += 1
            for cam, (files, times, k, distortion) in enumerate(cameras):
                image_path = nearest(files, times, t, args.max_camera_dt)
                if image_path is None:
                    continue
                # Quarter-resolution decode substantially reduces CPU and memory;
                # it is still finer than the 1 m DEM grid.
                image = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_4)
                if image is None:
                    continue
                ids, uv = project(local, cam, k, distortion)
                uv /= 4.0
                finite_uv = np.isfinite(uv).all(1)
                finite_uv &= (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
                finite_uv &= (uv[:, 0] < image.shape[1]) & (uv[:, 1] < image.shape[0])
                ids, uv = ids[finite_uv], uv[finite_uv]
                u = np.rint(uv[:, 0]).astype(np.int32)
                v = np.rint(uv[:, 1]).astype(np.int32)
                visible = (u < image.shape[1]) & (v < image.shape[0])
                if not visible.any():
                    continue
                ids, u, v = ids[visible], u[visible], v[visible]
                colors = image[v, u, ::-1].astype(np.float64)
                # Very dark/saturated pixels usually arise at image borders or glare.
                photometric = (colors.max(1) > 8) & (colors.min(1) < 250)
                ids, colors = ids[photometric], colors[photometric]
                if not len(ids):
                    continue
                np.add.at(sums, (iy[ids], ix[ids]), colors)
                np.add.at(counts, (iy[ids], ix[ids]), 1)
                projected += len(ids)
                used_images += 1
            if scan_index % 50 == 0:
                print(f"M0S{scene_id}: {scan_index}/{len(scans)} scans", flush=True)
        scene_reports[f"M0S{scene_id}"] = {
            "selected_scans": len(scans), "used_scans": used_scans,
            "used_camera_images": used_images, "projected_surface_samples": projected,
        }
        print(f"M0S{scene_id}: {projected:,} color samples from {used_images} images", flush=True)

    measured = counts > 0
    rgb = np.zeros((*dem.shape, 3), np.uint8)
    rgb[measured] = np.clip(sums[measured] / counts[measured, None], 0, 255).astype(np.uint8)
    observed = np.isfinite(dem)
    if measured.any():
        distance, indices = distance_transform_edt(~measured, return_indices=True)
        fill = observed & ~measured & (distance * resolution <= args.fill_radius)
        rgb[fill] = rgb[indices[0][fill], indices[1][fill]]
    else:
        fill = np.zeros_like(observed)
    colored = measured | fill

    np.savez_compressed(args.dem_dir / "terrain_rgb.npz", rgb=rgb,
                        sample_count=counts, measured_mask=measured,
                        filled_mask=fill, resolution_m=resolution, origin_xy=lo)
    save_preview(args.dem_dir / "terrain_rgb.png", rgb, colored)
    vertices = write_colored_obj(args.dem_dir / "terrain_colored.obj", dem,
                                 confidence, rgb, lo, resolution)
    report = {
        "scenes": list(scene_reports), "cameras_per_scene": 6,
        "step_s": args.step_s, "max_camera_dt_s": args.max_camera_dt,
        "height_tolerance_m": args.height_tolerance,
        "surface_color_samples": int(counts.sum()),
        "observed_dem_cells": int(observed.sum()),
        "directly_colored_cells": int((observed & measured).sum()),
        "direct_color_coverage_percent": float(100 * (observed & measured).sum() / max(observed.sum(), 1)),
        "colored_after_local_fill_cells": int((observed & colored).sum()),
        "filled_color_coverage_percent": float(100 * (observed & colored).sum() / max(observed.sum(), 1)),
        "mesh_vertices": vertices, "scene_statistics": scene_reports,
        "outputs": ["terrain_rgb.npz", "terrain_rgb.png", "terrain_colored.obj"],
        "method": "LiDAR/GNSS define DEM geometry. Synchronized pixels from all six calibrated cameras color only LiDAR returns within the DEM height tolerance; nearby holes are filled and explicitly recorded.",
    }
    (args.dem_dir / "camera_dem_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
