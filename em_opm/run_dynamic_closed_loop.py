#!/usr/bin/env python3
"""Closed-loop M0S7 replay with moving vehicles and collision screening."""
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent))
from run_closed_loop import Map,candidate_bank,wrap
from il_core import make_model
from risk import RiskNet
from train_collision import CollisionNet
from build_collision_dataset import dynamic_candidates

def actor_cases(track_file):
 d=json.loads(Path(track_file).read_text());sp=[x['speed_mps'] for x in d['tracks'] if x['dynamic'] and x['duration_s']>=2 and x['speed_mps']<=15]
 sp=(sp or [3.0])[:5];out=[]
 for i,v in enumerate(sp):
  out += [(f'follow_{i}',28.,max(.5,min(v,3.5)),1),(f'opposing_{i}',55.,max(1.,min(v,7.)), -1),(f'stopped_{i}',24.+2*i,0.,1)]
 return out

def local_future(m,actor_s,actor_v,actor_dir,x,y,yaw):
 q=[];co,si=np.cos(yaw),np.sin(yaw)
 for h in range(1,6):
  p=m.at_s(np.clip(actor_s+actor_dir*actor_v*h,m.s[0],m.s[-1]));d=p-np.array([x,y]);q.append([co*d[0]+si*d[1],-si*d[0]+co*d[1]])
 return np.asarray(q,np.float32)

def analytic_collision(cands,actor):
 out=[]
 for q in cands:
  d=np.diff(np.vstack([np.zeros((1,2)),q]),axis=0);yaw=np.arctan2(d[:,1],d[:,0]);r=actor-q;co,si=np.cos(yaw),np.sin(yaw);lon=co*r[:,0]+si*r[:,1];lat=-si*r[:,0]+co*r[:,1]
  out.append(float(np.any((np.abs(lon)<7.)&(np.abs(lat)<3.5))))
 return np.asarray(out)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--map',type=Path,default=Path('carla_mine/outputs/scene_7_closed_loop'));ap.add_argument('--il',type=Path,default=Path('em_opm/runs/il_v1/best.pt'));ap.add_argument('--boundary',type=Path,default=Path('em_opm/runs/risk_v1/best.pt'));ap.add_argument('--collision',type=Path,default=Path('em_opm/runs/collision_v2/best.pt'));ap.add_argument('--tracks',type=Path,default=Path('em_opm/data/dynamic_actors_v1/scene_7_tracks.json'));ap.add_argument('--mode',choices=['none','brake','geometric','learned'],required=True);ap.add_argument('--collision-weight',type=float,default=3.);ap.add_argument('--collision-threshold',type=float,default=-1.);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--starts',type=int,default=10);ap.add_argument('--dt',type=float,default=.1);ap.add_argument('--segment-length',type=float,default=120);ap.add_argument('--speed',type=float,default=5.);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=Map(a.map);il=make_model().to(dev);il.load_state_dict(torch.load(a.il,map_location=dev,weights_only=True)['model']);il.eval();br=RiskNet().to(dev);br.load_state_dict(torch.load(a.boundary,map_location=dev,weights_only=True)['model']);br.eval();cr=None
 if a.mode=='learned':cr=CollisionNet().to(dev);cr.load_state_dict(torch.load(a.collision,map_location=dev,weights_only=True)['model']);cr.eval()
 cases=actor_cases(a.tracks);valid=np.flatnonzero((m.s>100)&(m.s<m.s[-1]-a.segment_length-100));starts=[int(np.searchsorted(m.s,s)) for s in np.linspace(m.s[valid[0]],m.s[valid[-1]],a.starts)];bank,target_speeds=dynamic_candidates(a.speed);nc=len(bank);rows=[]
 for kk,start in enumerate(starts):
  n=len(cases);x=np.full(n,m.xy[start,0]);y=np.full(n,m.xy[start,1]);yaw=np.full(n,m.yaw[start]);v=np.full(n,a.speed);desired=np.full(n,a.speed);idx=np.full(n,start,int);active=np.ones(n,bool);outcome=np.array(['']*n,dtype='<U16');actor_s=np.array([m.s[start]+c[1] for c in cases]);actor_v=np.array([c[2] for c in cases]);actor_dir=np.array([c[3] for c in cases]);steer=np.zeros(n);min_clear=np.full(n,1e9);progress=np.zeros(n);target=m.s[start]+a.segment_length
  for tick in range(450):
   if tick%5==0 and active.any():
    ids=np.flatnonzero(active);bb=[];ss=[];gg=[];aa=[]
    for j in ids:
     idx[j]=m.nearest([x[j],y[j]],idx[j]);bb.append(m.bev(x[j],y[j],yaw[j]));g=m.at_s(min(m.s[idx[j]]+40,m.s[-1]))-[x[j],y[j]];co,si=np.cos(yaw[j]),np.sin(yaw[j]);gg.append([co*g[0]+si*g[1],-si*g[0]+co*g[1]]);ss.append([v[j],0,v[j]/4.5*np.tan(steer[j])]);aa.append(local_future(m,actor_s[j],actor_v[j],actor_dir[j],x[j],y[j],yaw[j]))
    bt=torch.from_numpy(np.stack(bb)).to(dev);st=torch.tensor(ss,dtype=torch.float32,device=dev);gt=torch.tensor(gg,dtype=torch.float32,device=dev)
    with torch.no_grad():ref,_=il(bt,st,gt);ref=ref.cpu().numpy()
    for k,j in enumerate(ids):
     ct=torch.from_numpy(bank).to(dev)
     with torch.no_grad():bs=torch.sigmoid(br(bt[k:k+1].expand(nc,-1,-1,-1),st[k:k+1].expand(nc,-1),gt[k:k+1].expand(nc,-1),ct)).cpu().numpy()[:,0]
     if a.mode=='none':cs=np.zeros(nc)
     elif a.mode=='brake':
      threat=analytic_collision(bank[target_speeds==5.],aa[k]).max()>0;cs=np.zeros(nc);cs[target_speeds>0]=float(threat)
     elif a.mode=='geometric':cs=analytic_collision(bank,aa[k])
     else:
      at=torch.from_numpy(aa[k]).to(dev).unsqueeze(0).expand(nc,-1,-1)
      with torch.no_grad():cs=torch.sigmoid(cr(ct,at)).cpu().numpy()
     deviate=np.linalg.norm(bank-ref[k][None],axis=2).mean(1);base=bs+.02*deviate+.03*(a.speed-target_speeds)
     if a.mode=='learned' and a.collision_threshold>=0:
      safe=cs<=a.collision_threshold;qi=int(np.flatnonzero(safe)[np.argmin(base[safe])]) if safe.any() else int(np.argmin(cs+.05*base))
     else:qi=int(np.argmin(base+a.collision_weight*cs))
     q=bank[qi];desired[j]=target_speeds[qi];tx,ty=q[1];ld=max(math.hypot(tx,ty),1);steer[j]=np.clip(math.atan2(9.*ty,ld*ld),-np.deg2rad(30),np.deg2rad(30))
   ids=np.flatnonzero(active);acc=np.clip(1.5*(desired[ids]-v[ids]),-3.,1.5);v[ids]=np.clip(v[ids]+acc*a.dt,0,a.speed);yaw[ids]=wrap(yaw[ids]+v[ids]/4.5*np.tan(steer[ids])*a.dt);x[ids]+=v[ids]*np.cos(yaw[ids])*a.dt;y[ids]+=v[ids]*np.sin(yaw[ids])*a.dt;actor_s[ids]+=actor_dir[ids]*actor_v[ids]*a.dt
   for j in ids:
    idx[j]=m.nearest([x[j],y[j]],idx[j]);ap=m.at_s(np.clip(actor_s[j],m.s[0],m.s[-1]));d=ap-[x[j],y[j]];co,si=np.cos(yaw[j]),np.sin(yaw[j]);lon=co*d[0]+si*d[1];lat=-si*d[0]+co*d[1];clear=max(abs(lon)-7.,abs(lat)-3.5);min_clear[j]=min(min_clear[j],clear);progress[j]=max(0,m.s[idx[j]]-m.s[start]);road=np.linalg.norm(np.array([x[j],y[j]])-m.xy[idx[j]])
    if abs(lon)<7 and abs(lat)<3.5:active[j]=False;outcome[j]='collision'
    elif road>4.25:active[j]=False;outcome[j]='boundary'
    elif m.s[idx[j]]>=target:active[j]=False;outcome[j]='success'
   if not active.any():break
  outcome[active]='timeout'
  for j,c in enumerate(cases):rows.append({'start':kk,'case':c[0],'mode':a.mode,'outcome':str(outcome[j]),'success':int(outcome[j]=='success'),'collision':int(outcome[j]=='collision'),'boundary':int(outcome[j]=='boundary'),'progress_m':float(progress[j]),'min_clearance_m':float(min_clear[j])})
  print(f'{kk+1}/{len(starts)}',flush=True)
 with (a.output/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 out={'mode':a.mode,'collision_weight':a.collision_weight,'collision_threshold':a.collision_threshold,'episodes':len(rows),'success_rate':float(np.mean([r['success'] for r in rows])),'collision_rate':float(np.mean([r['collision'] for r in rows])),'boundary_rate':float(np.mean([r['boundary'] for r in rows])),'timeout_rate':float(np.mean([r['outcome']=='timeout' for r in rows])),'mean_progress_m':float(np.mean([r['progress_m'] for r in rows])),'mean_min_clearance_m':float(np.mean([r['min_clearance_m'] for r in rows])),'source_tracks':len(actor_cases(a.tracks))//3,'scope':'controlled dynamic replay'};(a.output/'metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
