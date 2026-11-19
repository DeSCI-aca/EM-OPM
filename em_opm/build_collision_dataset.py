#!/usr/bin/env python3
"""Build action-conditioned collision pairs from measured mine-vehicle motion.

Track velocities come from camera detections localized with LiDAR. Placement and
arrival time are randomized to create controlled interactions; these are replay
augmentations, not additional observed encounters.
"""
import argparse,json
from pathlib import Path
import numpy as np

SPLITS={'train':[0,1,2,3,5],'val':[6],'test':[7]}
T=np.arange(1,6,dtype=np.float32)

def dynamic_candidates(v0=5.):
 out=[];speeds=[]
 for target in [5.,2.5,1.,0.]:
  ramp=3.;acc=(target-v0)/ramp;tm=np.minimum(T,ramp);x=v0*tm+.5*acc*tm*tm+target*np.maximum(T-ramp,0)
  for angle in np.deg2rad([-7.5,0,7.5]):
   for lateral in [-3,-1.5,0,1.5,3]:
    y=np.tan(angle)*x+lateral*(T/5.)**2;out.append(np.column_stack([x,y]));speeds.append(target)
 return np.asarray(out,np.float32),np.asarray(speeds,np.float32)

def collision(candidate,actor,ego_length=6.0,ego_width=3.5,actor_length=8.0,actor_width=3.5):
 d=np.diff(np.vstack([np.zeros((1,2)),candidate]),axis=0); yaw=np.arctan2(d[:,1],d[:,0]); rel=actor-candidate
 co,si=np.cos(yaw),np.sin(yaw); longitudinal=co*rel[:,0]+si*rel[:,1]; lateral=-si*rel[:,0]+co*rel[:,1]
 hit=(np.abs(longitudinal)<(ego_length+actor_length)/2)&(np.abs(lateral)<(ego_width+actor_width)/2)
 return float(hit.any())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tracks',type=Path,default=Path('em_opm/data/dynamic_actors_v1'));ap.add_argument('--output',type=Path,default=Path('em_opm/data/collision_v2'));ap.add_argument('--scenarios-per-track',type=int,default=96);ap.add_argument('--seed',type=int,default=23);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed);bank,target_speeds=dynamic_candidates();report={'source':'camera-LiDAR track velocity with randomized relative placement and longitudinal actions','candidate_count':len(bank),'target_speeds_mps':sorted(set(target_speeds.tolist())),'splits':{}}
 for split,ids in SPLITS.items():
  actors=[];cands=[];labels=[];track_ids=[]
  for sid in ids:
   d=json.loads((a.tracks/f'scene_{sid}_tracks.json').read_text())
   usable=[x for x in d['tracks'] if x['dynamic'] and x['duration_s']>=2 and x['speed_mps']<=15]
   for tr in usable:
    speed=max(.2,float(tr['speed_mps']))
    for _ in range(a.scenarios_per_track):
     # Mix following, crossing and opposing interactions while retaining measured speed.
     mode=int(rng.integers(3)); x0=float(rng.uniform(6,34)); y0=float(rng.uniform(-9,9))
     if mode==0: heading=float(rng.normal(0,.12))
     elif mode==1: heading=float(rng.choice([-1,1])*rng.uniform(.9,1.5))
     else: heading=float(np.pi+rng.normal(0,.12))
     v=speed*np.array([np.cos(heading),np.sin(heading)]); actor=np.array([x0,y0])+T[:,None]*v
     for j,c in enumerate(bank):actors.append(actor);cands.append(c);labels.append(collision(c,actor));track_ids.append([sid,tr['track_id'],mode,j])
  actors=np.asarray(actors,np.float32);cands=np.asarray(cands,np.float32);labels=np.asarray(labels,np.float32);track_ids=np.asarray(track_ids,np.int32)
  np.savez_compressed(a.output/f'{split}.npz',actors=actors,candidates=cands,labels=labels,meta=track_ids)
  report['splits'][split]={'pairs':len(labels),'positive':int(labels.sum()),'positive_rate':float(labels.mean()),'source_tracks':len(set(map(tuple,track_ids[:,:2])))}
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
