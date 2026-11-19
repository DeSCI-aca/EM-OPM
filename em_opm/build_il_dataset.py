#!/usr/bin/env python3
"""Create simulator-domain expert samples using the same point-map renderer as rollout."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carla_mine.build_stage1 import Trajectory
from run_closed_loop import Map,wrap
from il_core import HORIZONS,GOAL_HORIZON
SPLITS={'train':[0,1,2,3,5],'val':[6],'test':[7]}
def local(delta,yaw):
 c,s=np.cos(yaw),np.sin(yaw);return np.column_stack([c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path('/media/ivan/Elements SE/06-28'));ap.add_argument('--maps',type=Path,default=Path('carla_mine/outputs'));ap.add_argument('--output',type=Path,default=Path('em_opm/data/il_v1'));ap.add_argument('--sample-hz',type=float,default=.5);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);report={'domain':'accumulated point-map BEV used by closed-loop simulator','splits':{}}
 for split,ids in SPLITS.items():
  report['splits'][split]={}
  for sid in ids:
   scene=a.source/f'scene_{sid}';traj=Trajectory(scene);m=Map(a.maps/f'scene_{sid}_closed_loop');files=sorted((scene/'LiDAR').glob('*.npy'),key=lambda p:float(p.stem));times=[];last=-1e30
   for p in files:
    t=float(p.stem)
    if t<traj.start+.3 or t+GOAL_HORIZON>traj.end or t-last<1/a.sample_hz:continue
    times.append(t);last=t
   n=len(times);bevs=np.lib.format.open_memmap(a.output/f'M0S{sid}_bev.npy',mode='w+',dtype=np.float16,shape=(n,5,160,160));ways=np.empty((n,5,2),np.float32);goals=np.empty((n,2),np.float32);states=np.empty((n,3),np.float32)
   for i,t in enumerate(times):
    p,r=traj.poses([t]);yaw=r.as_euler('xyz')[0,2];future,_=traj.poses(t+HORIZONS);goal,_=traj.poses([t+GOAL_HORIZON]);ways[i]=local(future[:,:2]-p[0,:2],yaw);goals[i]=local((goal[:,:2]-p[0,:2]),yaw)[0];dt=.25;pm,rm=traj.poses([t-dt]);pp,rp=traj.poses([t+dt]);vel=local((pp[:,:2]-pm[:,:2])/(2*dt),yaw)[0];yr=wrap(rp.as_euler('xyz')[0,2]-rm.as_euler('xyz')[0,2])/(2*dt);states[i]=[vel[0],vel[1],yr];bevs[i]=m.bev(p[0,0],p[0,1],yaw)
    if (i+1)%100==0:print(f'M0S{sid} {i+1}/{n}',flush=True)
   del bevs;np.savez_compressed(a.output/f'M0S{sid}_labels.npz',waypoints=ways,goals=goals,states=states,timestamps=np.asarray(times));report['splits'][split][f'M0S{sid}']={'samples':n}
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
