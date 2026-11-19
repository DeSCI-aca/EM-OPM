#!/usr/bin/env python3
"""Fuse duplicate camera observations and form map-frame actor tracklets."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

VEHICLES={"car","truck","bus","motorcycle","bicycle"}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('detections',type=Path)
 ap.add_argument('--output',type=Path); ap.add_argument('--time-bin',type=float,default=.20)
 ap.add_argument('--fuse-radius',type=float,default=2.5); ap.add_argument('--match-radius',type=float,default=5.)
 ap.add_argument('--max-gap',type=float,default=1.0); ap.add_argument('--min-observations',type=int,default=4)
 ap.add_argument('--dynamic-speed',type=float,default=.5); a=ap.parse_args()
 rows=[json.loads(x) for x in a.detections.read_text().splitlines() if x.strip()]
 rows=[r for r in rows if r.get('class') in VEHICLES and 'position_map_m' in r]
 # Fuse multi-camera duplicates in short timestamp bins.
 bins={}
 for r in rows: bins.setdefault(round(r['timestamp']/a.time_bin),[]).append(r)
 observations=[]
 for _,rs in sorted(bins.items()):
  used=set()
  for i,r in enumerate(rs):
   if i in used: continue
   p=np.asarray(r['position_map_m'])[:2]; group=[i]
   for j,s in enumerate(rs[i+1:],i+1):
    if j not in used and s['class']==r['class'] and np.linalg.norm(np.asarray(s['position_map_m'])[:2]-p)<a.fuse_radius: group.append(j)
   used.update(group); g=[rs[j] for j in group]
   observations.append({'timestamp':float(np.mean([x['timestamp'] for x in g])),'class':r['class'],
    'position_map_m':np.median([x['position_map_m'] for x in g],axis=0).tolist(),
    'confidence':float(max(x['confidence'] for x in g)),'cameras':sorted(set(x['camera'] for x in g))})
 frames={}
 for o in observations: frames.setdefault(round(o['timestamp']/a.time_bin),[]).append(o)
 tracks=[]; active=[]
 for _,obs in sorted(frames.items()):
  t=np.mean([o['timestamp'] for o in obs]); active=[k for k in active if t-tracks[k][-1]['timestamp']<=a.max_gap]
  if active and obs:
   cost=np.full((len(active),len(obs)),1e6)
   for ii,k in enumerate(active):
    for jj,o in enumerate(obs):
     if tracks[k][-1]['class']==o['class']: cost[ii,jj]=np.linalg.norm(np.asarray(tracks[k][-1]['position_map_m'])[:2]-np.asarray(o['position_map_m'])[:2])
   ir,jc=linear_sum_assignment(cost); assigned=set()
   for ii,jj in zip(ir,jc):
    if cost[ii,jj]<=a.match_radius: tracks[active[ii]].append(obs[jj]); assigned.add(jj)
  else: assigned=set()
  for j,o in enumerate(obs):
   if j not in assigned: tracks.append([o]); active.append(len(tracks)-1)
 out=[]
 for i,tr in enumerate(tracks):
  if len(tr)<a.min_observations: continue
  ts=np.asarray([x['timestamp'] for x in tr]); xy=np.asarray([x['position_map_m'][:2] for x in tr]); duration=ts[-1]-ts[0]
  if duration>0:
   vx=np.polyfit(ts-ts[0],xy[:,0],1)[0]; vy=np.polyfit(ts-ts[0],xy[:,1],1)[0]
  else: vx=vy=0.
  speed=float(np.hypot(vx,vy)); displacement=float(np.linalg.norm(xy[-1]-xy[0]))
  out.append({'track_id':len(out),'class':tr[0]['class'],'observations':len(tr),'start_time':float(ts[0]),'end_time':float(ts[-1]),
   'duration_s':float(duration),'velocity_map_mps':[float(vx),float(vy)],'speed_mps':speed,'displacement_m':displacement,
   'dynamic':bool(speed>=a.dynamic_speed and displacement>=1.),'trajectory':tr})
 op=a.output or a.detections.with_name(a.detections.stem+'_tracks.json')
 op.write_text(json.dumps({'source':str(a.detections),'tracks':out,'dynamic_tracks':sum(x['dynamic'] for x in out),
  'automatic_pseudo_labels':True},indent=2))
 print(json.dumps({'observations':len(observations),'tracks':len(out),'dynamic_tracks':sum(x['dynamic'] for x in out),'output':str(op)},indent=2))
if __name__=='__main__': main()
