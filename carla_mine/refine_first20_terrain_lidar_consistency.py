#!/usr/bin/env python3
"""Refine the first-20% RL terrain for odometry/LiDAR consistency.

The full satellite terrain is retained.  Only cells inside the previously
approved first-20% ROI are changed.  Every selected 10 Hz Ouster scan is deskewed
with interpolated odometry, ground-like returns are associated with the current
surface, and a robust per-cell residual median is applied.  The program also
reports point-to-heightfield residuals per traversal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_mine.build_combined_dem import rigid_utm_alignment
from carla_mine.build_m0s0_first20_rl_region import SCENES, utm_trajectory
from carla_mine.build_stage1 import Trajectory


def bilinear(z, west, south, res, x, y):
    """Sample a north-up height array."""
    north = south + (z.shape[0] - 1) * res
    fx = (x - west) / res
    fy = (north - y) / res
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < z.shape[1]) & (y0 + 1 < z.shape[0])
    out = np.full(len(x), np.nan, np.float32)
    if ok.any():
        xx, yy = x0[ok], y0[ok]
        ax, ay = fx[ok] - xx, fy[ok] - yy
        out[ok] = ((1-ax)*(1-ay)*z[yy,xx] + ax*(1-ay)*z[yy,xx+1] +
                   (1-ax)*ay*z[yy+1,xx] + ax*ay*z[yy+1,xx+1])
    return out


def deskew_ground(path, traj, rotation, translation, altitude, bins=16,
                  points_per_scan=8192):
    """Return ground-like points in UTM, preserving the measured ~2 m offset."""
    raw = np.load(path, mmap_mode="r")
    xyz = np.asarray(raw[:, :3], np.float32)
    dt = np.asarray(raw[:, 4], np.float64)
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    # OS1-64 is about 2 m above the road.  Keep downward/near-horizontal rays;
    # final surface association below removes low outliers and structures.
    good = (np.isfinite(xyz).all(1) & np.isfinite(dt) &
            (radius >= 3.0) & (radius <= 70.0) &
            (xyz[:, 2] >= -5.5) & (xyz[:, 2] <= 0.35))
    xyz, dt = xyz[good], dt[good]
    if not len(xyz):
        return np.empty((0, 3), np.float32)
    # Preserve every 10 Hz acquisition time while preventing a stopped/slow
    # vehicle from dominating the surface.  Ouster packets are ordered by scan
    # column/ring, so deterministic index spacing is uniform over the sweep.
    if len(xyz) > points_per_scan:
        take = np.linspace(0, len(xyz)-1, points_per_scan).astype(np.int64)
        xyz, dt = xyz[take], dt[take]
    out = np.empty_like(xyz)
    edges = np.linspace(0, max(1e8, float(dt.max()) + 1), bins + 1)
    stamp = float(path.stem)
    for i in range(bins):
        m = (dt >= edges[i]) & (dt < edges[i+1] if i+1 < bins else dt <= edges[i+1])
        if not m.any():
            continue
        t = stamp + float(np.median(dt[m])) * 1e-9
        pos, rot = traj.poses([t])
        q = rot.apply(xyz[m]) + pos[0]
        q[:, :2] = q[:, :2] @ rotation.T + translation
        q[:, 2] += altitude
        out[m] = q
    return out


def metrics(a):
    a = np.abs(np.asarray(a, np.float64))
    return {"count": int(len(a)), "mae_m": float(a.mean()),
            "rmse_m": float(np.sqrt(np.mean(a*a))),
            "p50_m": float(np.percentile(a, 50)),
            "p95_m": float(np.percentile(a, 95))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("/media/ivan/Elements SE/06-28"))
    ap.add_argument("--input", type=Path, default=Path("carla_mine/outputs/m0s0_first20_rl_full_map"))
    ap.add_argument("--output", type=Path, default=Path("carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined"))
    ap.add_argument("--pose-radius", type=float, default=25.0)
    ap.add_argument("--association-m", type=float, default=1.25)
    ap.add_argument("--residual-bin-m", type=float, default=.05)
    ap.add_argument("--min-returns", type=int, default=8)
    ap.add_argument("--points-per-scan", type=int, default=4096)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pkg = np.load(args.input / "hybrid_dem.npz")
    base = pkg["height"].astype(np.float32)
    roi = pkg["roi_mask"].astype(bool)
    drive = pkg["drivable"].astype(bool)
    west, south, res = float(pkg["west"]), float(pkg["south"]), float(pkg["resolution_m"])
    north = south + (base.shape[0] - 1) * res
    core = np.load(args.input / "roi_definition.npz")["core_utm"]
    tree = cKDTree(core[:, :2])

    half = int(np.ceil(args.association_m / args.residual_bin_m))
    nbins = 2 * half + 1
    # Histogram only for the compact bounding rectangle of the approved ROI.
    ry, rx = np.nonzero(roi)
    y0, y1, x0, x1 = ry.min(), ry.max()+1, rx.min(), rx.max()+1
    rh, rw = y1-y0, x1-x0
    hist = np.zeros((rh*rw, nbins), np.uint32)
    total_scans = total_candidates = 0
    scene_counts = {}

    for label, raw_id in SCENES:
        folder = args.data_root / f"scene_{raw_id}"
        times, positions, _ = utm_trajectory(folder)
        traj = Trajectory(folder)
        rotation, translation, altitude, align = rigid_utm_alignment(folder)
        files = sorted((folder / "LiDAR").glob("*.npy"), key=lambda p: float(p.stem))
        used = accepted = 0
        for index, path in enumerate(files, 1):
            stamp = float(path.stem)
            if stamp < traj.start or stamp + .11 > traj.end or stamp < times[0] or stamp > times[-1]:
                continue
            px = np.interp(stamp, times, positions[:, 0]); py = np.interp(stamp, times, positions[:, 1])
            if tree.query([px, py])[0] > args.pose_radius:
                continue
            q = deskew_ground(path, traj, rotation, translation, altitude,
                              points_per_scan=args.points_per_scan)
            if not len(q):
                continue
            ix = np.floor((q[:, 0]-west)/res).astype(np.int32)
            iy = np.floor((north-q[:, 1])/res).astype(np.int32)
            inside = ((ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1))
            q, ix, iy = q[inside], ix[inside], iy[inside]
            if len(q):
                inside_roi = roi[iy, ix]
                q, ix, iy = q[inside_roi], ix[inside_roi], iy[inside_roi]
            if len(q):
                ref = bilinear(base, west, south, res, q[:,0], q[:,1])
                dz = q[:,2] - ref
                ok = np.isfinite(dz) & (np.abs(dz) <= args.association_m)
                ix, iy, dz = ix[ok], iy[ok], dz[ok]
                bi = np.clip(np.rint(dz/args.residual_bin_m).astype(np.int32)+half, 0, nbins-1)
                cell = (iy-y0)*rw + (ix-x0)
                np.add.at(hist, (cell, bi), 1)
                accepted += len(dz)
            used += 1
            if used % 500 == 0:
                print(f"{label}: {used} scans, {accepted:,} associated ground returns", flush=True)
        total_scans += used; total_candidates += accepted
        scene_counts[label] = {"scans_10hz": used, "associated_ground_returns": accepted,
                               "utm_alignment": align}
        print(label, scene_counts[label], flush=True)

    count = hist.sum(1)
    cdf = np.cumsum(hist, axis=1)
    medbin = (cdf >= ((count+1)//2)[:,None]).argmax(1)
    residual = (medbin.astype(np.float32)-half)*args.residual_bin_m
    residual[count < args.min_returns] = np.nan
    correction = np.full(base.shape, np.nan, np.float32)
    # A north-up row slice is not guaranteed to flatten as a writable view;
    # assign the 2-D block explicitly so corrections reach the full raster.
    correction[y0:y1, x0:x1] = residual.reshape(rh, rw)
    observed = np.isfinite(correction) & roi
    # Edge-preserving cleanup; never extrapolate outside the approved ROI.
    clean = ndimage.median_filter(np.where(observed, correction, 0), size=3)
    refined = base.copy()
    refined[observed] += clean[observed]
    # Light road-only regularization removes triangle-scale bumps while retaining grade.
    road_obs = observed & ndimage.binary_dilation(drive, iterations=2)
    smooth = ndimage.gaussian_filter(refined, .65)
    refined[road_obs] = .8*refined[road_obs] + .2*smooth[road_obs]

    # Evaluate the final heightfield with the histogram distribution itself.
    # Weighted metrics without expanding hundreds of millions of returns.
    abs_hist = np.zeros(nbins * 2 + 1, np.uint64)
    err_sum = err_sq_sum = 0.0
    err_n = 0
    centers = (np.arange(nbins)-half)*args.residual_bin_m
    final_corr = (refined-base)[y0:y1,x0:x1].reshape(-1)
    for b in range(nbins):
        ids = np.flatnonzero(hist[:,b])
        if len(ids):
            e = centers[b]-final_corr[ids]
            w = hist[ids,b].astype(np.uint64)
            ae = np.abs(e)
            err_n += int(w.sum())
            err_sum += float((ae*w).sum())
            err_sq_sum += float((e*e*w).sum())
            eb = np.clip(np.rint(ae/args.residual_bin_m).astype(np.int32),0,len(abs_hist)-1)
            np.add.at(abs_hist,eb,w)
    ecdf=np.cumsum(abs_hist)
    def wpct(p):
        if not err_n: return float('nan')
        return float(np.searchsorted(ecdf,err_n*p/100)*args.residual_bin_m)
    consistency={"count":err_n,"mae_m":err_sum/max(err_n,1),
                 "rmse_m":float(np.sqrt(err_sq_sum/max(err_n,1))),
                 "p50_m":wpct(50),"p95_m":wpct(95)}

    z0 = float(pkg["z_origin_m"])
    np.savez_compressed(args.output/"hybrid_dem.npz", height=refined,
                        resolution_m=res, crs="EPSG:32650", west=west, south=south,
                        z_origin_m=z0, roi_mask=roi, drivable=drive,
                        lidar_count=np.pad(count.reshape(rh,rw), ((y0,base.shape[0]-y1),(x0,base.shape[1]-x1))),
                        lidar_correction_m=np.nan_to_num(correction, nan=0).astype(np.float32))
    np.save(args.output/"drivable.npy", drive)
    for name in ["MineFullTexture.png", "routes_roi.csv", "roi_definition.npz"]:
        (args.output/name).write_bytes((args.input/name).read_bytes())
    for sub in ["navigation"]:
        target=args.output/sub; target.mkdir(exist_ok=True)
        for p in (args.input/sub).iterdir():
            if p.is_file(): (target/p.name).write_bytes(p.read_bytes())

    lim = np.nanpercentile(np.abs(correction[observed]), 99) if observed.any() else 1
    vis = np.zeros((*base.shape,3),np.uint8)
    norm = np.clip(np.nan_to_num(correction)/(max(lim,.05)*2)+.5,0,1)
    vis = cv2.applyColorMap((norm*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    vis[~roi] //= 5
    cv2.imwrite(str(args.output/"lidar_correction_preview.png"),vis)
    report = {"method":"10 Hz OS1-64 scan replay; 16-bin odometry deskew; robust ground residual median; ROI-only update",
              "lidar_mounting_height_observation_m":"approximately 2.0--2.4; preserved through measured point coordinates, not manually subtracted",
              "full_terrain_and_texture_preserved":True,"outside_roi_unchanged":bool(np.array_equal(refined[~roi],base[~roi])),
              "scans_used":total_scans,"uniform_points_per_scan":args.points_per_scan,
              "associated_ground_returns":total_candidates,
              "updated_cells":int(observed.sum()),"minimum_returns_per_cell":args.min_returns,
              "correction_abs_p50_p95_max_m":[float(np.nanpercentile(np.abs(correction[observed]),p)) for p in [50,95,100]],
              "heightfield_consistency":consistency,"per_scene_input":scene_counts}
    (args.output/"lidar_consistency_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "per_scene_input"},indent=2))


if __name__ == "__main__":
    main()
