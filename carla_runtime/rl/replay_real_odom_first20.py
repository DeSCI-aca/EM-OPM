#!/usr/bin/env python3
"""Replay M0S6/M0S7 odometry in CARLA and compare OS1-64 ground ranges.

Only poses within 25 m of the M0S0 first-20% core path are simulated and
scored.  Samples outside that spatial interval are deliberately ignored.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import carla
from carla_mine.build_combined_dem import rigid_utm_alignment
from carla_mine.build_stage1 import Trajectory


SCENES = {"M0S6": 5, "M0S7": 6}


def angular_ranges(xyz, azimuth_bins=512, elevation_bins=64):
    """Closest ground-like range per angular bin in ROS LiDAR coordinates."""
    rxy = np.linalg.norm(xyz[:, :2], axis=1)
    rr = np.linalg.norm(xyz, axis=1)
    good = (np.isfinite(xyz).all(1) & (rr >= 3) & (rr <= 70) &
            (xyz[:, 2] <= .35) & (xyz[:, 2] >= -6))
    xyz, rxy, rr = xyz[good], rxy[good], rr[good]
    az = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2*np.pi)
    el = np.arctan2(xyz[:, 2], np.maximum(rxy, 1e-6))
    ia = np.floor(az/(2*np.pi)*azimuth_bins).astype(np.int32)
    ie = np.floor((el-np.deg2rad(-22.5))/np.deg2rad(45)*elevation_bins).astype(np.int32)
    good = (ie >= 0) & (ie < elevation_bins)
    ids = ie[good]*azimuth_bins + ia[good]
    out = np.full(elevation_bins*azimuth_bins, np.inf, np.float32)
    np.minimum.at(out, ids, rr[good])
    return out


def percentile_metrics(values):
    a=np.abs(np.asarray(values,np.float64))
    if not len(a): return {"count":0,"mae_m":None,"rmse_m":None,"p50_m":None,"p95_m":None}
    return {"count":int(len(a)),"mae_m":float(a.mean()),
            "rmse_m":float(np.sqrt(np.mean(a*a))),
            "p50_m":float(np.percentile(a,50)),"p95_m":float(np.percentile(a,95))}


def local_ground_intercept(xyz):
    """Robust local road plane z=ax+by+c; c is ground height at the sensor."""
    r=np.linalg.norm(xyz[:,:2],axis=1)
    m=(np.isfinite(xyz).all(1)&(r>=3)&(r<=15)&(np.abs(xyz[:,1])<=4)&
       (xyz[:,2]>=-5)&(xyz[:,2]<=-.5))
    p=xyz[m]
    if len(p)<80:return np.nan,0
    A=np.column_stack((p[:,0],p[:,1],np.ones(len(p))))
    keep=np.ones(len(p),bool);coef=np.zeros(3)
    for _ in range(4):
        if keep.sum()<50:break
        coef=np.linalg.lstsq(A[keep],p[keep,2],rcond=None)[0]
        e=p[:,2]-A@coef;med=np.median(e[keep]);mad=np.median(np.abs(e[keep]-med))
        keep=np.abs(e-med)<=max(.12,2.5*1.4826*mad)
    return float(coef[2]),int(keep.sum())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=2000)
    ap.add_argument('--data-root',type=Path,default=Path('/media/ivan/Elements SE/06-28'))
    ap.add_argument('--map-root',type=Path,default=ROOT/'carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined')
    ap.add_argument('--scenes',default='M0S6,M0S7');ap.add_argument('--roi-radius',type=float,default=25.)
    ap.add_argument('--compare-stride',type=int,default=10)
    ap.add_argument('--output',type=Path,default=ROOT/'carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined/carla_odom_replay')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)

    dem=np.load(args.map_root/'hybrid_dem.npz');z0=float(dem['z_origin_m'])
    west,south=float(dem['west']),float(dem['south']);z=dem['height']
    east=west+(z.shape[1]-1)*float(dem['resolution_m']);north=south+(z.shape[0]-1)*float(dem['resolution_m'])
    cx,cy=(west+east)/2,(south+north)/2
    core=np.load(args.map_root/'roi_definition.npz')['core_utm'];tree=cKDTree(core[:,:2])

    client=carla.Client(args.host,args.port);client.set_timeout(30.)
    world=client.get_world();original=world.get_settings()
    settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.1
    settings.no_rendering_mode=False;world.apply_settings(settings)
    lib=world.get_blueprint_library();actors=[];rows=[];all_errors=[];all_ground_errors=[]
    try:
        vehicle_bp=lib.find('vehicle.tesla.model3')
        lidar_bp=lib.find('sensor.lidar.ray_cast')
        for key,value in [('channels','64'),('range','70'),('rotation_frequency','10'),
                          ('points_per_second','655360'),('upper_fov','22.5'),('lower_fov','-22.5'),
                          ('sensor_tick','0.0')]: lidar_bp.set_attribute(key,value)
        for label in [x.strip() for x in args.scenes.split(',') if x.strip()]:
            raw_id=SCENES[label];folder=args.data_root/f'scene_{raw_id}'
            traj=Trajectory(folder);rot2,trans,alt,align=rigid_utm_alignment(folder)
            Ralign=np.eye(3);Ralign[:2,:2]=rot2
            files=sorted((folder/'LiDAR').glob('*.npy'),key=lambda p:float(p.stem))
            selected=[]
            for p in files:
                t=float(p.stem)
                if t<traj.start or t+.11>traj.end:continue
                pos,_=traj.poses([t]);utm=pos[0,:2]@rot2.T+trans
                if tree.query(utm)[0]<=args.roi_radius:selected.append(p)
            if not selected:continue
            t0=float(selected[0].stem);pos0,r0=traj.poses([t0]);utm0=pos0[0,:2]@rot2.T+trans
            Rc=np.diag([1,-1,1])@(Ralign@r0[0].as_matrix())@np.diag([1,-1,1])
            roll,pitch,yaw=Rotation.from_matrix(Rc).as_euler('xyz',degrees=True)
            sensor_z=pos0[0,2]+alt-z0
            tf=carla.Transform(carla.Location(float(utm0[0]-cx),float(-(utm0[1]-cy)),float(sensor_z-2.0)),
                               carla.Rotation(float(pitch),float(yaw),float(roll)))
            vehicle=world.spawn_actor(vehicle_bp,tf);actors.append(vehicle);vehicle.set_simulate_physics(False)
            q=queue.Queue(maxsize=8)
            lidar=world.spawn_actor(lidar_bp,carla.Transform(carla.Location(z=2.0)),attach_to=vehicle)
            actors.append(lidar);lidar.listen(lambda data:q.put(data) if not q.full() else None)
            world.tick()
            while not q.empty():q.get_nowait()
            scene_errors=[];scene_ground_errors=[]
            for idx,p in enumerate(selected):
                stamp=float(p.stem);pos,rr=traj.poses([stamp]);utm=pos[0,:2]@rot2.T+trans
                Rc=np.diag([1,-1,1])@(Ralign@rr[0].as_matrix())@np.diag([1,-1,1])
                roll,pitch,yaw=Rotation.from_matrix(Rc).as_euler('xyz',degrees=True)
                sensor_z=pos[0,2]+alt-z0
                vehicle.set_transform(carla.Transform(
                    carla.Location(float(utm[0]-cx),float(-(utm[1]-cy)),float(sensor_z-2.0)),
                    carla.Rotation(float(pitch),float(yaw),float(roll))))
                # Do not wait for a future sensor frame while synchronous CARLA
                # is paused.  Remove stale callbacks before advancing exactly
                # one step, then use the newest callback produced by that step.
                while not q.empty():
                    try:q.get_nowait()
                    except queue.Empty:break
                frame=world.tick();data=q.get(timeout=20)
                while not q.empty():
                    newer=q.get_nowait()
                    if newer.frame>=data.frame:data=newer
                nvirt=len(data.raw_data)//16
                virtual=np.frombuffer(data.raw_data,dtype=np.float32).reshape(nvirt,4)[:,:3].copy()
                # CARLA sensor +Y is right; recorded ROS/base_link +Y is left.
                virtual[:,1]*=-1
                err=np.empty(0,np.float32);shared=0;ground_error='';real_c='';virtual_c=''
                if idx%args.compare_stride==0:
                    real=np.asarray(np.load(p,mmap_mode='r')[:,:3],np.float32)
                    ar,av=angular_ranges(real),angular_ranges(virtual)
                    valid=np.isfinite(ar)&np.isfinite(av);shared=int(valid.sum());err=av[valid]-ar[valid]
                    scene_errors.extend(err.tolist());all_errors.extend(err.tolist())
                    real_c,nr=local_ground_intercept(real);virtual_c,nv=local_ground_intercept(virtual)
                    if np.isfinite(real_c) and np.isfinite(virtual_c):
                        ground_error=virtual_c-real_c;scene_ground_errors.append(ground_error);all_ground_errors.append(ground_error)
                rows.append([label,stamp,utm[0],utm[1],sensor_z,nvirt,shared,
                             float(np.mean(np.abs(err))) if len(err) else '',real_c,virtual_c,ground_error])
                if (idx+1)%500==0:print(f'{label}: {idx+1}/{len(selected)} ROI frames',flush=True)
            lidar.stop();lidar.destroy();actors.remove(lidar);vehicle.destroy();actors.remove(vehicle)
            (args.output/f'{label}_metrics.json').write_text(json.dumps({
                'scene':label,'roi_frames_replayed':len(selected),'comparison_stride':args.compare_stride,
                'range_consistency_all_ground_like_returns':percentile_metrics(scene_errors),
                'local_road_plane_height_consistency':percentile_metrics(scene_ground_errors),'utm_alignment':align},indent=2))
        with (args.output/'frames.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['scene','timestamp','utm_e','utm_n','sensor_z_abs_m','virtual_points','shared_angular_bins','range_mae_m','real_ground_z_sensor_m','virtual_ground_z_sensor_m','ground_height_error_m']);w.writerows(rows)
        report={'map':world.get_map().name,'scope':'Only poses within 25 m of M0S0 first-20% core; all outside poses ignored',
                'scenes':[x.strip() for x in args.scenes.split(',')],
                'roi_frames_replayed':len(rows),'virtual_lidar':'64 channels, 10 Hz, 655360 points/s, +/-22.5 deg, 70 m',
                'range_consistency_all_ground_like_returns':percentile_metrics(all_errors),
                'local_road_plane_height_consistency':percentile_metrics(all_ground_errors)}
        (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:
        for a in actors:
            try:a.destroy()
            except Exception:pass
        world.apply_settings(original)

if __name__=='__main__':main()
