#!/usr/bin/env python3
"""Build DEM, terrain OBJ and a simple OpenDRIVE road from one mine traversal."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial.transform import Rotation, Slerp

METERS = 111195.0


class Trajectory:
    def __init__(self, scene: Path):
        gps = np.load(scene / "gps_fix.npy")
        odom = np.load(scene / "gnss_odom.npy")
        good = np.isfinite(gps[:, [0, 2, 3, 4]]).all(1) & (gps[:, 5] >= 0)
        gps = gps[good]
        order = np.argsort(gps[:, 0]); gps = gps[order]
        _, keep = np.unique(gps[:, 0], return_index=True); gps = gps[np.sort(keep)]
        good = np.isfinite(odom[:, [0, 5, 6, 7, 8]]).all(1)
        good &= np.abs(np.linalg.norm(odom[:, 5:9], axis=1) - 1.0) < .1
        odom = odom[good]
        order = np.argsort(odom[:, 0]); odom = odom[order]
        _, keep = np.unique(odom[:, 0], return_index=True); odom = odom[np.sort(keep)]
        self.lat0, self.lon0, self.alt0 = gps[0, 2:5]
        self.gps_t = gps[:, 0]
        self.xyz = np.column_stack(((gps[:, 3] - self.lon0) * np.cos(np.deg2rad(self.lat0)) * METERS,
                                    (gps[:, 2] - self.lat0) * METERS,
                                    gps[:, 4] - self.alt0))
        self.odom_t = odom[:, 0]
        self.slerp = Slerp(self.odom_t, Rotation.from_quat(odom[:, 5:9]))
        self.start = max(self.gps_t[0], self.odom_t[0])
        self.end = min(self.gps_t[-1], self.odom_t[-1])

    def poses(self, times):
        t = np.asarray(times, np.float64)
        xyz = np.column_stack([np.interp(t, self.gps_t, self.xyz[:, j]) for j in range(3)])
        return xyz, self.slerp(t)


def select_scans(scene, step_s, max_scans):
    files = sorted((scene / "LiDAR").glob("*.npy"), key=lambda p: float(p.stem))
    out, last = [], -np.inf
    for p in files:
        t = float(p.stem)
        if t - last >= step_s:
            out.append(p); last = t
            if max_scans and len(out) >= max_scans: break
    return out


def world_scan(path, traj, max_points, bins=5):
    raw = np.load(path, mmap_mode="r")
    xyz = np.asarray(raw[:, :3], np.float32)
    offsets = np.asarray(raw[:, 4], np.float64)
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    good = np.isfinite(xyz).all(1) & np.isfinite(offsets)
    good &= (radius >= 3) & (radius <= 80) & (xyz[:, 2] > -8) & (xyz[:, 2] < 20)
    xyz, offsets = xyz[good], offsets[good]
    if len(xyz) > max_points:
        ids = np.linspace(0, len(xyz) - 1, max_points).astype(int)
        xyz, offsets = xyz[ids], offsets[ids]
    out = np.empty_like(xyz)
    edges = np.linspace(0, max(1e8, float(offsets.max()) + 1), bins + 1)
    for i in range(bins):
        mask = (offsets >= edges[i]) & (offsets < edges[i + 1] if i + 1 < bins else offsets <= edges[i + 1])
        if not mask.any(): continue
        t = float(path.stem) + float(np.median(offsets[mask])) * 1e-9
        pos, rot = traj.poses([t])
        out[mask] = rot.apply(xyz[mask]) + pos[0]
    return out


def make_dem(points, resolution, padding):
    lo = np.floor((points[:, :2].min(0) - padding) / resolution) * resolution
    hi = np.ceil((points[:, :2].max(0) + padding) / resolution) * resolution
    width, height = np.ceil((hi - lo) / resolution).astype(int) + 1
    if width * height > 80_000_000:
        raise RuntimeError(f"DEM is too large: {width}x{height}; increase --resolution")
    ix = np.floor((points[:, 0] - lo[0]) / resolution).astype(int)
    iy = np.floor((points[:, 1] - lo[1]) / resolution).astype(int)
    flat = iy * width + ix
    order = np.argsort(flat); flat, z = flat[order], points[order, 2]
    cells, starts, counts = np.unique(flat, return_index=True, return_counts=True)
    # The low quartile suppresses vehicles, vegetation and dust returns while
    # retaining sloped road surfaces better than a per-cell minimum.
    ground = np.array([np.quantile(z[s:s+n], .25) for s, n in zip(starts, counts)], np.float32)
    dem = np.full(height * width, np.nan, np.float32); dem[cells] = ground
    count = np.zeros(height * width, np.uint32); count[cells] = counts
    dem, count = dem.reshape(height, width), count.reshape(height, width)
    observed = count >= 3
    dem[observed] = median_filter(np.where(observed, dem, 0), size=3)[observed]
    confidence = np.clip(np.log1p(count) / np.log(30), 0, 1).astype(np.float32)
    return dem, count, confidence, lo


def write_obj(path, dem, confidence, lo, res, min_conf=.25):
    valid = np.isfinite(dem) & (confidence >= min_conf)
    ids = -np.ones(dem.shape, np.int64); y, x = np.nonzero(valid)
    ids[y, x] = np.arange(len(x)) + 1
    with path.open("w") as f:
        f.write("# observed mine terrain; coordinates are local ENU metres\n")
        for yy, xx in zip(y, x):
            f.write(f"v {lo[0]+xx*res:.3f} {lo[1]+yy*res:.3f} {dem[yy,xx]:.3f}\n")
        h, w = dem.shape
        for yy in range(h - 1):
            for xx in range(w - 1):
                a, b, c, d = ids[yy,xx], ids[yy,xx+1], ids[yy+1,xx], ids[yy+1,xx+1]
                if min(a,b,c,d) > 0:
                    # Reject faces spanning a likely wall or unmodelled object.
                    zz = dem[yy:yy+2, xx:xx+2]
                    if np.ptp(zz) <= max(2.0, 3.0*res):
                        f.write(f"f {a} {b} {d}\nf {a} {d} {c}\n")
    return int(valid.sum())


def resample_centerline(traj, spacing=2.0):
    keep = np.r_[True, np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1) > .15]
    q = traj[keep]
    if len(q) < 2: raise RuntimeError("trajectory has fewer than two distinct positions")
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1))]
    s = np.arange(0, distance[-1] + spacing*.5, spacing)
    out = np.column_stack([np.interp(s, distance, q[:, j]) for j in range(3)])
    if len(out) >= 7:
        for j in range(3): out[:, j] = gaussian_filter(out[:, j], 1.0, mode="nearest")
    return s, out


def write_xodr(path, s, xyz, road_width):
    headings = np.unwrap(np.arctan2(np.gradient(xyz[:,1]), np.gradient(xyz[:,0])))
    with path.open("w") as f:
        f.write('<?xml version="1.0" standalone="yes"?>\n<OpenDRIVE>\n')
        f.write('  <header revMajor="1" revMinor="4" name="M0S0_mine" version="1.00"/>\n')
        f.write(f'  <road name="expert_route" length="{s[-1]:.3f}" id="1" junction="-1">\n    <planView>\n')
        for i in range(len(s)-1):
            length = s[i+1]-s[i]
            f.write(f'      <geometry s="{s[i]:.3f}" x="{xyz[i,0]:.3f}" y="{xyz[i,1]:.3f}" hdg="{headings[i]:.8f}" length="{length:.3f}"><line/></geometry>\n')
        f.write('    </planView>\n    <elevationProfile>\n')
        for i in range(len(s)-1):
            slope=(xyz[i+1,2]-xyz[i,2])/max(s[i+1]-s[i],1e-6)
            f.write(f'      <elevation s="{s[i]:.3f}" a="{xyz[i,2]:.3f}" b="{slope:.8f}" c="0" d="0"/>\n')
        half=road_width/2
        f.write(f'''    </elevationProfile>
    <lanes><laneSection s="0"><left><lane id="1" type="driving" level="false"><width sOffset="0" a="{half:.3f}" b="0" c="0" d="0"/></lane></left><center><lane id="0" type="none" level="false"/></center><right><lane id="-1" type="driving" level="false"><width sOffset="0" a="{half:.3f}" b="0" c="0" d="0"/></lane></right></laneSection></lanes>
  </road>
</OpenDRIVE>
''')


def preview(path, dem, count, lo, res, trajectory):
    valid=np.isfinite(dem); image=np.zeros((*dem.shape,3),np.uint8)
    if valid.any():
        a,b=np.percentile(dem[valid],[2,98]); norm=np.clip((dem-a)/max(b-a,1e-6),0,1)
        color=cv2.applyColorMap((np.nan_to_num(norm)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        image[valid]=color[valid]
    image[count<3]//=3
    uv=np.column_stack(((trajectory[:,0]-lo[0])/res,(trajectory[:,1]-lo[1])/res)).astype(np.int32)
    cv2.polylines(image,[uv.reshape(-1,1,2)],False,(255,255,255),max(1,int(2/res)),cv2.LINE_AA)
    image=cv2.flip(image,0)
    max_side=max(image.shape[:2])
    if max_side>2400:
        scale=2400/max_side;image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path),image)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--step-s',type=float,default=1.0);ap.add_argument('--max-scans',type=int,default=0)
    ap.add_argument('--max-points',type=int,default=24000);ap.add_argument('--resolution',type=float,default=.5)
    ap.add_argument('--padding',type=float,default=2.0);ap.add_argument('--road-width',type=float,default=12.0)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    traj=Trajectory(args.scene);files=select_scans(args.scene,args.step_s,args.max_scans)
    clouds=[];used=[]
    for i,p in enumerate(files,1):
        t=float(p.stem)
        if t < traj.start or t+.1 > traj.end: continue
        clouds.append(world_scan(p,traj,args.max_points));used.append(t)
        if i%50==0: print(f'{i}/{len(files)} scans',flush=True)
    if not clouds: raise RuntimeError('no scan overlaps the valid GNSS/odometry interval')
    points=np.concatenate(clouds); dem,count,confidence,lo=make_dem(points,args.resolution,args.padding)
    key=np.floor(points/.5).astype(np.int32)
    _,keep=np.unique(key,axis=0,return_index=True)
    global_points=points[np.sort(keep)].astype(np.float32)
    np.save(args.output/'global_points.npy',global_points)
    np.savez_compressed(args.output/'terrain_dem.npz',height=dem,count=count,confidence=confidence,
                        origin_xy=lo,resolution_m=args.resolution,
                        geodetic_origin=np.array([traj.lat0,traj.lon0,traj.alt0]))
    pose,_=traj.poses(np.asarray(used));s,center=resample_centerline(pose,2.0)
    with (args.output/'road_centerline.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['s_m','east_m','north_m','up_m']);w.writerows(np.column_stack((s,center)))
    write_xodr(args.output/'road.xodr',s,center,args.road_width)
    vertices=write_obj(args.output/'terrain.obj',dem,confidence,lo,args.resolution)
    preview(args.output/'terrain_dem.png',dem,count,lo,args.resolution,pose)
    report={'scene':str(args.scene),'selected_scans':len(files),'used_scans':len(clouds),
            'sampled_points':int(len(points)),'global_point_voxels':int(len(global_points)),'dem_shape':[int(x) for x in dem.shape],
            'resolution_m':args.resolution,'observed_cells':int(np.isfinite(dem).sum()),
            'mesh_vertices':vertices,'centerline_length_m':float(s[-1]),'road_width_m':args.road_width,
            'local_geodetic_origin_lat_lon_alt':[traj.lat0,traj.lon0,traj.alt0],
            'assumptions':['LiDAR coordinates are confirmed to be in the base_link/odometry child frame',
                           'ground height is the per-cell lower quartile',
                           'OpenDRIVE width is a configurable approximation'],
            'outputs':['terrain_dem.npz','global_points.npy','terrain_dem.png','terrain.obj','road_centerline.csv','road.xodr']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__': main()
