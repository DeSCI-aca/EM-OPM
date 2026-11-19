#!/usr/bin/env python3
"""Build a local RL mine region from the spatial footprint of M0S0's first 20%.

The first 20% is defined by the original time-ordered odometry rows.  Later
samples from M0S0 and all samples from the other traversals are retained when
their vehicle position re-enters that spatial corridor.  Only sensors attached
to retained poses contribute to the local terrain/data manifests.
"""
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
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_mine.build_stage1 import Trajectory, world_scan
from carla_mine.build_combined_dem import rigid_utm_alignment


TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
SCENES = [("M0S0", 0), ("M0S1", 1), ("M0S2", 2), ("M0S3", 3),
          ("M0S5", 4), ("M0S6", 5), ("M0S7", 6)]


def utm_trajectory(folder: Path):
    odom = np.load(folder / "gnss_odom.npy")
    gps = np.load(folder / "gps_fix.npy")
    odom = odom[np.isfinite(odom[:, 0])]
    odom = odom[np.argsort(odom[:, 0])]
    _, ids = np.unique(odom[:, 0], return_index=True)
    odom = odom[np.sort(ids)]
    good = np.isfinite(gps[:, [0, 2, 3, 4]]).all(1)
    good &= (gps[:, 2] > 40) & (gps[:, 2] < 50)
    good &= (gps[:, 3] > 110) & (gps[:, 3] < 120)
    gps = gps[good]
    gps = gps[np.argsort(gps[:, 0])]
    _, ids = np.unique(gps[:, 0], return_index=True)
    gps = gps[np.sort(ids)]
    t = odom[:, 0]
    lon = np.interp(t, gps[:, 0], gps[:, 3])
    lat = np.interp(t, gps[:, 0], gps[:, 2])
    alt = np.interp(t, gps[:, 0], gps[:, 4])
    east, north = TO_UTM.transform(lon, lat)
    return t, np.column_stack((east, north, alt)), odom


def thin_path(points, spacing=.5):
    keep = [0]
    anchor = points[0, :2]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i, :2] - anchor) >= spacing:
            keep.append(i)
            anchor = points[i, :2]
    return points[np.asarray(keep)]


def time_inside(tree, times, positions, query_times, radius):
    x = np.interp(query_times, times, positions[:, 0])
    y = np.interp(query_times, times, positions[:, 1])
    return tree.query(np.column_stack((x, y)), workers=-1)[0] <= radius


def write_obj(path, dem, west, south, z0, res, stride, texture_name):
    z = np.flipud(dem)[::stride, ::stride]
    h, w = z.shape
    step = res * stride * 100.0
    x0, y0 = -(w - 1) * step / 2, -(h - 1) * step / 2
    mtl = path.with_suffix(".mtl")
    mtl.write_text("newmtl MineLocalSurface\nKd 1 1 1\nKa 0 0 0\nKs 0 0 0\n"
                   f"map_Kd {texture_name}\n")
    with path.open("w") as f:
        f.write(f"mtllib {mtl.name}\no MineLocalTerrain\n")
        for iy in range(h):
            for ix in range(w):
                f.write(f"v {x0+ix*step:.3f} {y0+iy*step:.3f} "
                        f"{(z[iy,ix]-z0)*100:.3f}\n")
        for iy in range(h):
            for ix in range(w):
                f.write(f"vt {ix/(w-1):.7f} {iy/(h-1):.7f}\n")
        f.write("usemtl MineLocalSurface\ns 1\n")
        for iy in range(h-1):
            a = iy*w+1; c = (iy+1)*w+1
            for ix in range(w-1):
                p, q, r, s = a+ix, a+ix+1, c+ix, c+ix+1
                f.write(f"f {p}/{p} {q}/{q} {s}/{s}\n"
                        f"f {p}/{p} {s}/{s} {r}/{r}\n")
    return [w, h]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path,
                    default=Path("/media/ivan/Elements SE/06-28"))
    ap.add_argument("--base-dem", type=Path, default=Path(
        "carla_mine/outputs/carla_route_buffer_1km_utm_aligned/"
        "lidar_all_points_1m/hybrid_dem.npz"))
    ap.add_argument("--ortho", type=Path, default=Path(
        "carla_mine/outputs/carla_route_buffer_1km_utm_aligned/"
        "government_ortho/MineHybridTexture_GovernmentOrtho_utm50n.tif"))
    ap.add_argument("--output", type=Path, default=Path(
        "carla_mine/outputs/m0s0_first20_rl_region"))
    ap.add_argument("--fraction", type=float, default=.2)
    ap.add_argument("--pose-radius", type=float, default=25.)
    ap.add_argument("--terrain-radius", type=float, default=85.)
    ap.add_argument("--lidar-step", type=float, default=.5)
    ap.add_argument("--max-points-per-scan", type=int, default=12000)
    ap.add_argument("--mesh-stride", type=int, default=2)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    t0, p0, _ = utm_trajectory(args.data_root / "scene_0")
    cut = int(np.ceil(args.fraction * len(p0)))
    core_dense = p0[:cut]
    core = thin_path(core_dense)
    tree = cKDTree(core[:, :2])
    bounds_min = core[:, :2].min(0) - args.terrain_radius
    bounds_max = core[:, :2].max(0) + args.terrain_radius

    base = np.load(args.base_dem)
    base_z = base["height"].astype(np.float32)
    res = float(base["resolution_m"])
    base_west, base_south = float(base["west"]), float(base["south"])
    base_north = base_south + (base_z.shape[0]-1)*res
    c0 = max(0, int(np.floor((bounds_min[0]-base_west)/res)))
    c1 = min(base_z.shape[1]-1, int(np.ceil((bounds_max[0]-base_west)/res)))
    s0 = max(0, int(np.floor((bounds_min[1]-base_south)/res)))
    s1 = min(base_z.shape[0]-1, int(np.ceil((bounds_max[1]-base_south)/res)))
    r0, r1 = base_z.shape[0]-1-s1, base_z.shape[0]-s0
    dem = base_z[r0:r1, c0:c1+1].copy()
    west, south = base_west+c0*res, base_south+s0*res
    north = south+(dem.shape[0]-1)*res
    east = west+(dem.shape[1]-1)*res
    yy, xx = np.indices(dem.shape)
    grid_xy = np.column_stack((west+xx.ravel()*res,
                               north-yy.ravel()*res))
    roi_mask = (tree.query(grid_xy, workers=-1)[0] <= args.terrain_radius).reshape(dem.shape)

    route_rows, lidar_rows, camera_rows = [], [], []
    clouds = []
    scene_stats = {}
    for label, raw_id in SCENES:
        folder = args.data_root / f"scene_{raw_id}"
        times, positions, _ = utm_trajectory(folder)
        inside = tree.query(positions[:, :2], workers=-1)[0] <= args.pose_radius
        change = np.diff(np.r_[False, inside, False].astype(np.int8))
        starts, ends = np.flatnonzero(change == 1), np.flatnonzero(change == -1)
        interval = np.full(len(times), -1, np.int32)
        for iid, (a, b) in enumerate(zip(starts, ends)):
            interval[a:b] = iid
        # Route output is thinned spatially but preserves both outbound/return intervals.
        for iid, (a, b) in enumerate(zip(starts, ends)):
            q = thin_path(positions[a:b], .5)
            for e, n, z in q:
                route_rows.append([label, iid, e, n, z])

        lidar_files = sorted((folder / "LiDAR").glob("*.npy"),
                             key=lambda p: float(p.stem))
        selected = []
        last = -np.inf
        for path in lidar_files:
            stamp = float(path.stem)
            if stamp-last < args.lidar_step:
                continue
            last = stamp
            if stamp < times[0] or stamp > times[-1]:
                continue
            if not time_inside(tree, times, positions, np.array([stamp]),
                               args.pose_radius)[0]:
                continue
            selected.append(path)
            lidar_rows.append([label, stamp, str(path)])

        traj = Trajectory(folder)
        rotation, translation, altitude, _ = rigid_utm_alignment(folder)
        for index, path in enumerate(selected, 1):
            q = world_scan(path, traj, args.max_points_per_scan)
            q[:, :2] = q[:, :2] @ rotation.T + translation
            q[:, 2] += altitude
            box = ((q[:, 0] >= west) & (q[:, 0] <= east) &
                   (q[:, 1] >= south) & (q[:, 1] <= north))
            q = q[box]
            if len(q):
                q = q[tree.query(q[:, :2], workers=-1)[0] <= args.terrain_radius]
            if len(q):
                clouds.append(q.astype(np.float32))
            if index % 250 == 0:
                print(f"{label}: {index}/{len(selected)} LiDAR scans", flush=True)

        camera_count = 0
        for cam in range(6):
            for path in sorted((folder/f"mav0/cam{cam}/data").glob("*.png"),
                               key=lambda p: float(p.stem)):
                stamp = float(path.stem)
                if stamp < times[0] or stamp > times[-1]:
                    continue
                if time_inside(tree, times, positions, np.array([stamp]),
                               args.pose_radius)[0]:
                    camera_rows.append([label, cam, stamp, str(path)])
                    camera_count += 1
        scene_stats[label] = {
            "raw_folder": f"scene_{raw_id}",
            "odom_total": int(len(times)), "odom_inside": int(inside.sum()),
            "entry_intervals": int(len(starts)),
            "lidar_selected": int(len(selected)),
            "camera_selected": int(camera_count),
        }
        print(label, scene_stats[label], flush=True)

    points = np.concatenate(clouds)
    ix = np.floor((points[:, 0]-west)/res).astype(np.int32)
    iy = np.floor((north-points[:, 1])/res).astype(np.int32)
    ok = (ix>=0)&(ix<dem.shape[1])&(iy>=0)&(iy<dem.shape[0])
    ix, iy, pz = ix[ok], iy[ok], points[ok, 2]
    flat = iy*dem.shape[1]+ix
    order = np.argsort(flat)
    flat, pz = flat[order], pz[order]
    cells, starts, counts = np.unique(flat, return_index=True, return_counts=True)
    ground = np.asarray([np.quantile(pz[a:a+n], .25)
                         for a, n in zip(starts, counts)], np.float32)
    observed = np.zeros(dem.size, np.uint32)
    observed[cells] = counts
    observed = observed.reshape(dem.shape)
    lidar_ground = np.full(dem.size, np.nan, np.float32)
    lidar_ground[cells] = ground
    lidar_ground = lidar_ground.reshape(dem.shape)
    trusted = (observed >= 3) & roi_mask
    filtered = ndimage.median_filter(np.where(trusted, lidar_ground, dem), size=3)
    dem[trusted] = filtered[trusted]

    # Routes in local raster coordinates.
    routes_csv = args.output / "routes_roi.csv"
    with routes_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "interval", "x_east_m",
                                       "y_north_m", "z_m"])
        for label, iid, e, n, z in route_rows:
            w.writerow([label, iid, e-west, n-south, z])
    for name, header, rows in [
        ("lidar_manifest.csv", ["scene", "timestamp", "path"], lidar_rows),
        ("camera_manifest.csv", ["scene", "camera", "timestamp", "path"], camera_rows)]:
        with (args.output/name).open("w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    # Drivable area: connected gentle terrain around the seven proven routes,
    # strictly clipped to the local ROI.
    route_seed = np.zeros(dem.shape, np.uint8)
    for label, iid in sorted({(r[0], r[1]) for r in route_rows}):
        p = np.asarray([[r[2], r[3]] for r in route_rows
                        if r[0] == label and r[1] == iid])
        uv = np.column_stack(((p[:, 0]-west)/res,
                              (north-p[:, 1])/res)).astype(np.int32)
        cv2.polylines(route_seed, [uv.reshape(-1,1,2)], False, 1,
                      max(1, int(round(12/res))), cv2.LINE_8)
    smooth = ndimage.gaussian_filter(dem, max(1., 1.5/res))
    gy, gx = np.gradient(smooth, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    trend = ndimage.gaussian_filter(dem, max(1., 3/res))
    residual = dem-trend
    rough = ndimage.maximum_filter(residual, 3)-ndimage.minimum_filter(residual, 3)
    candidate = roi_mask & (slope <= 12) & (rough <= .5)
    candidate |= route_seed.astype(bool)
    labels, _ = ndimage.label(candidate, np.ones((3,3), np.uint8))
    connected_ids = np.unique(labels[route_seed > 0]); connected_ids = connected_ids[connected_ids>0]
    drivable = np.isin(labels, connected_ids) & roi_mask
    np.save(args.output/"drivable.npy", drivable)
    cv2.imwrite(str(args.output/"drivable.png"), drivable.astype(np.uint8)*255)

    z0 = float(np.floor(np.nanmin(dem)-1))
    np.savez_compressed(args.output/"hybrid_dem.npz", height=dem,
                        resolution_m=res, crs="EPSG:32650", west=west,
                        south=south, z_origin_m=z0, count=observed,
                        roi_mask=roi_mask, lidar_trusted=trusted)

    # Georeferenced ortho crop for terrain UVs.
    tex_size = 2048
    transform = from_bounds(west, south, east, north, tex_size, tex_size)
    rgb = np.zeros((tex_size, tex_size, 3), np.uint8)
    with rasterio.open(args.ortho) as src:
        for band in range(3):
            reproject(rasterio.band(src, band+1), rgb[:,:,band],
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:32650",
                      resampling=Resampling.bilinear)
    texture = args.output/"MineLocalTexture.png"
    cv2.imwrite(str(texture), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    mesh_size = write_obj(args.output/"MineLocalTerrain.obj", dem, west, south,
                          z0, res, args.mesh_stride, texture.name)

    # RL starts on the first 60 m; five terminal points span the road width at
    # the exact end of M0S0's first-20% segment.
    q = core_dense
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q[:,:2], axis=0), axis=1))]
    start_s = np.linspace(0, min(60., arc[-1]*.1), 7)
    start_points = np.column_stack([np.interp(start_s, arc, q[:,j]) for j in range(3)])
    tangent = q[-1,:2]-q[max(0,len(q)-50),:2]
    tangent /= max(np.linalg.norm(tangent), 1e-6)
    normal = np.array([-tangent[1], tangent[0]])
    goals = []
    for offset in [-4., -2., 0., 2., 4.]:
        xy = q[-1,:2]+offset*normal
        col = int(round((xy[0]-west)/res)); row = int(round((north-xy[1])/res))
        if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1] and drivable[row,col]:
            goals.append([float(xy[0]-west), float(xy[1]-south),
                          float(dem[row,col]-z0), float(offset)])
    starts_out = []
    for p in start_points:
        col = int(round((p[0]-west)/res)); row = int(round((north-p[1])/res))
        starts_out.append([float(p[0]-west), float(p[1]-south),
                           float(dem[row,col]-z0+1.),
                           float(np.degrees(np.arctan2(tangent[1],tangent[0])))])
    for name, header, rows in [
        ("rl_start_points.csv", ["x_local_m","y_local_m","z_relative_m","yaw_deg"], starts_out),
        ("rl_terminal_points.csv", ["x_local_m","y_local_m","z_relative_m","lateral_offset_m"], goals)]:
        with (args.output/name).open("w", newline="") as f:
            w=csv.writer(f);w.writerow(header);w.writerows(rows)

    qa = cv2.cvtColor(cv2.normalize(dem, None, 0, 180, cv2.NORM_MINMAX).astype(np.uint8),
                      cv2.COLOR_GRAY2BGR)
    qa[~roi_mask] //= 4
    qa[drivable] = (.35*qa[drivable]+.65*np.array([40,200,70])).astype(np.uint8)
    qa[route_seed>0] = (255,255,255)
    for x,y,_,_ in starts_out:
        cv2.circle(qa,(int(x/res),int((dem.shape[0]-1)-y/res)),5,(255,80,20),-1)
    for x,y,_,_ in goals:
        cv2.circle(qa,(int(x/res),int((dem.shape[0]-1)-y/res)),7,(20,20,255),-1)
    cv2.imwrite(str(args.output/"roi_drivable_rl_preview.png"), qa)

    report = {
        "definition": "M0S0 first 20% of original time-ordered odometry rows; later re-entry retained spatially",
        "fraction": args.fraction, "m0s0_cut_row": cut,
        "m0s0_core_duration_s": float(t0[cut-1]-t0[0]),
        "m0s0_core_path_length_m": float(np.linalg.norm(np.diff(core_dense[:,:2],axis=0),axis=1).sum()),
        "pose_radius_m": args.pose_radius, "terrain_radius_m": args.terrain_radius,
        "bounds_utm": [west,south,east,north], "dem_shape": list(dem.shape),
        "lidar_points_used": int(len(points)), "lidar_trusted_cells": int(trusted.sum()),
        "roi_area_m2": float(roi_mask.sum()*res*res),
        "drivable_area_m2": float(drivable.sum()*res*res),
        "start_points": len(starts_out), "terminal_points": len(goals),
        "mesh_vertices_xy": mesh_size, "scenes": scene_stats,
        "raw_to_logical_scene_mapping": {f"scene_{raw}": label for label,raw in SCENES},
    }
    (args.output/"report.json").write_text(json.dumps(report, indent=2))
    (args.output/"rl_task.json").write_text(json.dumps({
        "start_points": "rl_start_points.csv", "terminal_points": "rl_terminal_points.csv",
        "goal_radius_m": 5.0, "terminate_outside_drivable": True,
        "terminate_outside_roi": True, "max_route_length_m": report["m0s0_core_path_length_m"]*1.5,
    }, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
