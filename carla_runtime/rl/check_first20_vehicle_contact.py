#!/usr/bin/env python3
"""Check Tesla envelope collision and four-wheel terrain contact on real odom."""
from __future__ import annotations
import argparse,csv,json,queue,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import carla
from carla_mine.build_combined_dem import rigid_utm_alignment
from carla_mine.build_stage1 import Trajectory

SCENES={'M0S6':5,'M0S7':6}
S=np.diag([1.,-1.,1.])

def summary(x):
 a=np.asarray(x,float);a=a[np.isfinite(a)]
 if not len(a):return {'count':0}
 return {'count':int(len(a)),'mean_m':float(a.mean()),'std_m':float(a.std()),
         'p05_m':float(np.percentile(a,5)),'p50_m':float(np.percentile(a,50)),
         'p95_m':float(np.percentile(a,95)),'min_m':float(a.min()),'max_m':float(a.max())}

def to_transform(pos,rot,rot2,trans,alt,cx,cy,z0,sensor_to_root):
 utm=pos[:2]@rot2.T+trans;R3=np.eye(3);R3[:2,:2]=rot2
 Rc=S@(R3@rot.as_matrix())@S;roll,pitch,yaw=Rotation.from_matrix(Rc).as_euler('xyz',degrees=True)
 sensor_z=pos[2]+alt-z0
 tf=carla.Transform(carla.Location(float(utm[0]-cx),float(-(utm[1]-cy)),float(sensor_z-sensor_to_root)),
                    carla.Rotation(float(pitch),float(yaw),float(roll)))
 return tf,utm,sensor_z

def world_point(tf,p):
 # Explicit CARLA rotation avoids API-version differences in Transform.transform.
 r=np.deg2rad([tf.rotation.roll,tf.rotation.pitch,tf.rotation.yaw])
 cr,cp,cy=np.cos(r);sr,sp,sy=np.sin(r)
 R=np.array([[cp*cy,cy*sp*sr-sy*cr,-cy*sp*cr-sy*sr],
             [cp*sy,sy*sp*sr+cy*cr,-sy*sp*cr+cy*sr],
             [sp,-cp*sr,cp*cr]])
 q=R@np.asarray(p,float)+np.array([tf.location.x,tf.location.y,tf.location.z])
 return q

def ground_hit(world,x,y,z):
 hits=world.cast_ray(carla.Location(float(x),float(y),float(z+3.0)),
                     carla.Location(float(x),float(y),float(z-3.0)))
 if not hits:return np.nan
 vals=[h.location.z for h in hits if h.location.z<=z+1.0]
 return float(max(vals)) if vals else np.nan

def runs(mask):
 best=cur=0
 for v in mask:
  cur=cur+1 if v else 0;best=max(best,cur)
 return int(best)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=2000)
 ap.add_argument('--data-root',type=Path,default=Path('/media/ivan/Elements SE/06-28'))
 ap.add_argument('--map-root',type=Path,default=ROOT/'carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined')
 ap.add_argument('--output',type=Path,default=ROOT/'carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined/carla_vehicle_contact')
 ap.add_argument('--roi-radius',type=float,default=25.);ap.add_argument('--sensor-to-root',type=float,default=2.025)
 ap.add_argument('--envelope-stride',type=int,default=5)
 args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 d=np.load(args.map_root/'hybrid_dem.npz');z=d['height'];res=float(d['resolution_m']);z0=float(d['z_origin_m'])
 west,south=float(d['west']),float(d['south']);east=west+(z.shape[1]-1)*res;north=south+(z.shape[0]-1)*res
 cx,cy=(west+east)/2,(south+north)/2;core=np.load(args.map_root/'roi_definition.npz')['core_utm'];tree=cKDTree(core[:,:2])
 c=carla.Client(args.host,args.port);c.set_timeout(30);world=c.get_world();old=world.get_settings();st=world.get_settings();st.synchronous_mode=True;st.fixed_delta_seconds=.1;world.apply_settings(st)
 actors=[];rows=[];events=[];scene_reports={}
 try:
  bp=world.get_blueprint_library().find('vehicle.tesla.model3');cbp=world.get_blueprint_library().find('sensor.other.collision')
  for label,raw_id in SCENES.items():
   folder=args.data_root/f'scene_{raw_id}';traj=Trajectory(folder);rot2,trans,alt,align=rigid_utm_alignment(folder)
   files=sorted((folder/'LiDAR').glob('*.npy'),key=lambda p:float(p.stem));selected=[]
   for p in files:
    t=float(p.stem)
    if t<traj.start or t+.11>traj.end:continue
    pos,_=traj.poses([t]);utm=pos[0,:2]@rot2.T+trans
    if tree.query(utm)[0]<=args.roi_radius:selected.append(p)
   pos,r=traj.poses([float(selected[0].stem)]);tf,_,_=to_transform(pos[0],r[0],rot2,trans,alt,cx,cy,z0,args.sensor_to_root)
   vehicle=world.spawn_actor(bp,tf);actors.append(vehicle);vehicle.set_simulate_physics(False)
   cq=queue.Queue();collision=world.spawn_actor(cbp,carla.Transform(),attach_to=vehicle);actors.append(collision)
   collision.listen(lambda e,lab=label:cq.put((lab,e)))
   phys=vehicle.get_physics_control();wheels=phys.wheels
   # CARLA reports wheel positions in centimetres/world coordinates. Tesla local geometry is stable.
   wheel_local=np.array([[1.61824066,-.83341461,.28972973],[1.61824066,.83341461,.28972973],
                         [-1.38640564,-.83341461,.28972973],[-1.38640564,.83341461,.28972973]])
   radius=float(np.median([w.radius for w in wheels]))/100.
   bb=vehicle.bounding_box;bx,by=float(bb.extent.x),float(bb.extent.y);bz=float(bb.location.z-bb.extent.z)
   footprint=np.array([[bx,by,bz],[bx,-by,bz],[-bx,by,bz],[-bx,-by,bz],[0,0,bz]])
   gaps=[[],[],[],[]];spreads=[];envelope=[];jumps=[];missing=0;prev_ground=None;prev_t=None
   for i,p in enumerate(selected):
    stamp=float(p.stem);pos,r=traj.poses([stamp]);tf,utm,sensor_z=to_transform(pos[0],r[0],rot2,trans,alt,cx,cy,z0,args.sensor_to_root)
    # Temporarily remove the dynamic vehicle from the query segment so the
    # downward rays hit terrain rather than the Tesla body/wheels themselves.
    high=carla.Transform(carla.Location(tf.location.x,tf.location.y,tf.location.z+100.0),tf.rotation)
    vehicle.set_transform(high)
    wg=[]
    for wi,local in enumerate(wheel_local):
     wc=world_point(tf,local);gz=ground_hit(world,wc[0],wc[1],wc[2]);gap=wc[2]-radius-gz if np.isfinite(gz) else np.nan
     gaps[wi].append(gap);wg.append(gz)
    wg=np.asarray(wg);valid=np.isfinite(wg)
    if valid.all():
     spread=float(np.nanmax([gaps[j][-1] for j in range(4)])-np.nanmin([gaps[j][-1] for j in range(4)]));spreads.append(spread)
     if prev_ground is not None and stamp-prev_t<.2:jumps.extend(np.abs(wg-prev_ground).tolist())
     prev_ground=wg;prev_t=stamp
    else:missing+=1;prev_ground=None;prev_t=None
    env_min=''
    if i%args.envelope_stride==0:
     clear=[]
     for local in footprint:
      wp=world_point(tf,local);gz=ground_hit(world,wp[0],wp[1],wp[2]);
      if np.isfinite(gz):clear.append(wp[2]-gz)
     if clear:env_min=float(min(clear));envelope.append(env_min)
    # Put the complete envelope back on the recorded pose for a real PhysX
    # overlap/collision tick after all terrain-only rays have been evaluated.
    vehicle.set_transform(tf);frame=world.tick()
    while not cq.empty():
     lab,e=cq.get_nowait();imp=e.normal_impulse;events.append([lab,stamp,e.frame,e.transform.location.x,e.transform.location.y,e.transform.location.z,e.other_actor.type_id,float(np.sqrt(imp.x**2+imp.y**2+imp.z**2))])
    rows.append([label,stamp,utm[0],utm[1],sensor_z,*[g[-1] for g in gaps],spread if valid.all() else '',env_min])
    if (i+1)%500==0:print(f'{label}: {i+1}/{len(selected)} contact frames',flush=True)
   collision.stop();collision.destroy();actors.remove(collision);vehicle.destroy();actors.remove(vehicle)
   flat=np.concatenate([np.asarray(x,float) for x in gaps]);air=flat>.10;penetrate=flat<-.10
   perwheel={f'wheel_{i}':summary(gaps[i]) for i in range(4)}
   scene_reports[label]={'roi_frames':len(selected),'wheel_gap_m':summary(flat),'per_wheel_gap_m':perwheel,
    'four_wheel_gap_spread_m':summary(spreads),'ground_height_step_per_0p1s_m':summary(jumps),
    'envelope_min_clearance_m':summary(envelope),'missing_ground_ray_frames':missing,
    'wheel_air_fraction_gt_0p10m':float(air.mean()),'wheel_penetration_fraction_lt_minus0p10m':float(penetrate.mean()),
    'max_consecutive_air_samples_any_wheel':max(runs(np.asarray(x)>.10) for x in gaps),'collision_events':sum(1 for x in events if x[0]==label),'utm_alignment':align}
  with (args.output/'frames.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['scene','timestamp','utm_e','utm_n','sensor_z_abs_m','gap_fl_m','gap_fr_m','gap_rl_m','gap_rr_m','gap_spread_m','envelope_min_clearance_m']);w.writerows(rows)
  with (args.output/'collision_events.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['scene','timestamp','frame','x','y','z','other_actor','normal_impulse']);w.writerows(events)
  report={'map':world.get_map().name,'scope':'Only LiDAR/odom times within 25 m of M0S0 first-20% core; outside ignored',
          'sensor_to_vehicle_root_m':args.sensor_to_root,'tesla_wheel_radius_m':radius,
          'contact_thresholds_m':{'air':.10,'penetration':-.10},'scenes':scene_reports}
  (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 finally:
  for a in actors:
   try:a.destroy()
   except:pass
  world.apply_settings(old)

if __name__=='__main__':main()
