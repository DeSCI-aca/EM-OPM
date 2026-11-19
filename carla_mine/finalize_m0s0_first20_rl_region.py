#!/usr/bin/env python3
"""Finalize CARLA coordinates, OpenDRIVE and RL endpoints for the local ROI."""
import csv
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

ROOT = Path("carla_mine/outputs/m0s0_first20_rl_region")
DATA = Path("/media/ivan/Elements SE/06-28/scene_0")
TO_UTM = Transformer.from_crs(4326, 32650, always_xy=True)


def resample(points, spacing=2.0):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]
    target = np.arange(0, d[-1] + spacing*.5, spacing)
    return target, np.column_stack([np.interp(target, d, points[:, i]) for i in range(3)])


def main():
    report = json.loads((ROOT/"report.json").read_text())
    west, south, east, north = report["bounds_utm"]
    cx, cy = (west+east)/2, (south+north)/2
    dem = np.load(ROOT/"hybrid_dem.npz")
    z = dem["height"]; z0 = float(dem["z_origin_m"]); res = float(dem["resolution_m"])

    od = np.load(DATA/"gnss_odom.npy"); gps = np.load(DATA/"gps_fix.npy")
    od = od[np.isfinite(od[:,0])]; od = od[np.argsort(od[:,0])]
    good = np.isfinite(gps[:,[0,2,3,4]]).all(1); gps=gps[good]; gps=gps[np.argsort(gps[:,0])]
    t=od[:,0]; lon=np.interp(t,gps[:,0],gps[:,3]); lat=np.interp(t,gps[:,0],gps[:,2]); alt=np.interp(t,gps[:,0],gps[:,4])
    e,n=TO_UTM.transform(lon,lat); all_pose=np.c_[e,n,alt]
    cut=int(np.ceil(.2*len(all_pose))); core=all_pose[:cut]
    s, route = resample(core)
    np.savez_compressed(ROOT/"roi_definition.npz", core_utm=core, route_utm=route,
                        route_s_m=s, bounds_utm=np.array([west,south,east,north]),
                        fraction=np.array(.2), cut_row=np.array(cut))

    # Centered UE/CARLA coordinates; CARLA API mirrors Unreal Y.
    local = route.copy(); local[:,0]-=cx; local[:,1]-=cy
    local[:,2]-=z0
    heading=np.unwrap(np.arctan2(np.gradient(local[:,1]),np.gradient(local[:,0])))
    xodr=ROOT/"MineFirst20RL.xodr"
    with xodr.open("w") as f:
        f.write('<?xml version="1.0" standalone="yes"?>\n<OpenDRIVE>\n')
        f.write('  <header revMajor="1" revMinor="4" name="MineFirst20RL" version="1.00"/>\n')
        f.write(f'  <road name="M0S0_first20" length="{s[-1]:.3f}" id="1" junction="-1">\n    <planView>\n')
        for i in range(len(s)-1):
            f.write(f'      <geometry s="{s[i]:.3f}" x="{local[i,0]:.3f}" y="{-local[i,1]:.3f}" hdg="{-heading[i]:.8f}" length="{s[i+1]-s[i]:.3f}"><line/></geometry>\n')
        f.write('    </planView>\n    <elevationProfile>\n')
        for i in range(len(s)-1):
            slope=(local[i+1,2]-local[i,2])/max(s[i+1]-s[i],1e-6)
            f.write(f'      <elevation s="{s[i]:.3f}" a="{local[i,2]:.3f}" b="{slope:.8f}" c="0" d="0"/>\n')
        f.write('    </elevationProfile>\n    <lanes><laneSection s="0"><left><lane id="1" type="driving" level="false"><width sOffset="0" a="6" b="0" c="0" d="0"/></lane></left><center><lane id="0" type="none" level="false"/></center><right><lane id="-1" type="driving" level="false"><width sOffset="0" a="6" b="0" c="0" d="0"/></lane></right></laneSection></lanes>\n  </road>\n</OpenDRIVE>\n')

    navigation=ROOT/"navigation"; navigation.mkdir(exist_ok=True)
    starts=list(csv.DictReader((ROOT/"rl_start_points.csv").open()))
    with (navigation/"spawn_points.csv").open("w",newline="") as f:
        fields=["id","source_route","x_local_m","y_local_m","z_relative_m","yaw_deg","carla_x_m","carla_y_m","carla_yaw_deg"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,r in enumerate(starts):
            x=float(r["x_local_m"])+west; y=float(r["y_local_m"])+south
            yaw=float(r["yaw_deg"])
            w.writerow(dict(id=i,source_route="M0S0_first20",x_local_m=x-west,y_local_m=y-south,
                            z_relative_m=r["z_relative_m"],yaw_deg=yaw,
                            carla_x_m=x-cx,carla_y_m=-(y-cy),carla_yaw_deg=-yaw))
    goals=list(csv.DictReader((ROOT/"rl_terminal_points.csv").open()))
    with (navigation/"terminal_points.csv").open("w",newline="") as f:
        fields=["id","carla_x_m","carla_y_m","z_relative_m","radius_m","lateral_offset_m"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,r in enumerate(goals):
            x=float(r["x_local_m"])+west; y=float(r["y_local_m"])+south
            w.writerow(dict(id=i,carla_x_m=x-cx,carla_y_m=-(y-cy),z_relative_m=r["z_relative_m"],
                            radius_m=5.,lateral_offset_m=r["lateral_offset_m"]))
    (navigation/"rl_task.json").write_text((ROOT/"rl_task.json").read_text())
    report["carla_center_utm"]=[cx,cy]
    report["route_xodr_length_m"]=float(s[-1])
    report["outputs_finalized"]=["roi_definition.npz","MineFirst20RL.xodr","navigation/spawn_points.csv","navigation/terminal_points.csv"]
    (ROOT/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"center_utm":[cx,cy],"route_m":float(s[-1]),"starts":len(starts),"goals":len(goals)},indent=2))

if __name__ == "__main__": main()
