#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from run_closed_loop import Map
SPLITS={'train':[0,1,2,3,5],'val':[6],'test':[7]};H=np.arange(1,6,dtype=np.float32)
def candidates(speed=5.):
 out=[]
 for angle in np.deg2rad([-15,-7.5,0,7.5,15]):
  for lateral in [-3,-1.5,0,1.5,3]:
   x=speed*H;y=np.tan(angle)*x+lateral*(H/5.)**2;out.append(np.column_stack([x,y]))
 return np.asarray(out,np.float32)
def labels(m,x,y,yaw,cands,road=12.,width=3.5):
 co,si=np.cos(yaw),np.sin(yaw);out=[]
 for q in cands:
  samples=[];prev=np.zeros(2)
  for z in q:
   samples.extend(prev+(z-prev)*u for u in np.linspace(.1,1,10));prev=z
  p=np.asarray(samples);world=np.column_stack([x+co*p[:,0]-si*p[:,1],y+si*p[:,0]+co*p[:,1]])
  dist=m.route_tree.query(world)[0]
  boundary=float(np.max(dist)>road/2-width/2)
  d=np.diff(np.vstack([np.zeros((1,2)),q]),axis=0);head=np.unwrap(np.arctan2(d[:,1],d[:,0]));curv=np.abs(np.diff(head))/np.maximum(np.linalg.norm(d[1:],axis=1),.2);execution=float(np.any(curv>.25) or np.any(np.linalg.norm(d,axis=1)>12))
  out.append([boundary,execution])
 return np.asarray(out,np.float32)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--maps',type=Path,default=Path('carla_mine/outputs'));ap.add_argument('--output',type=Path,default=Path('em_opm/data/risk_v1'));ap.add_argument('--route-states',type=int,default=40);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);report={'labels':['boundary','execution'],'splits':{}}
 base=candidates()
 for split,ids in SPLITS.items():
  report['splits'][split]={}
  for sid in ids:
   m=Map(a.maps/f'scene_{sid}_closed_loop');valid=np.flatnonzero((m.s>80)&(m.s<m.s[-1]-100));route_ids=np.linspace(valid[0],valid[-1],a.route_states).astype(int);n=len(route_ids)*9;bevs=np.lib.format.open_memmap(a.output/f'M0S{sid}_bev.npy',mode='w+',dtype=np.float16,shape=(n,5,160,160));states=np.empty((n,3),np.float32);goals=np.empty((n,2),np.float32);cands=np.empty((n,25,5,2),np.float32);ys=np.empty((n,25,2),np.float32);meta=np.empty((n,3),np.float32);k=0
   for i in route_ids:
    p=m.xy[i];head=m.yaw[i];normal=np.array([-np.sin(head),np.cos(head)])
    for off in [-3,0,3]:
     for da in np.deg2rad([-10,0,10]):
      x,y=p+off*normal;yaw=head+da;goal=m.at_s(min(m.s[i]+40,m.s[-1]))-np.array([x,y]);co,si=np.cos(yaw),np.sin(yaw);goal=np.array([co*goal[0]+si*goal[1],-si*goal[0]+co*goal[1]],np.float32);bevs[k]=m.bev(x,y,yaw);states[k]=[5,0,0];goals[k]=goal;cands[k]=base;ys[k]=labels(m,x,y,yaw,base);meta[k]=[m.s[i],off,np.rad2deg(da)];k+=1
   del bevs;np.savez_compressed(a.output/f'M0S{sid}_labels.npz',states=states,goals=goals,candidates=cands,labels=ys,meta=meta);pos=ys.reshape(-1,2).mean(0);report['splits'][split][f'M0S{sid}']={'states':n,'candidates':int(n*25),'positive_rate':pos.tolist()}
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
