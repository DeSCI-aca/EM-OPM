#!/usr/bin/env python3
"""Kinematic closed-loop evaluation of the IL policy on reconstructed M0S7."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,torch
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from il_core import make_model
from risk import RiskNet

def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi

class Map:
 def __init__(self,root):
  z=np.load(root/'terrain_dem.npz');self.h=z['height'];self.c=z['confidence'];self.lo=z['origin_xy'];self.res=float(z['resolution_m']);self.route=np.loadtxt(root/'road_centerline.csv',delimiter=',',skiprows=1);self.xy=self.route[:,1:3];self.z=self.route[:,3];self.s=self.route[:,0];d=np.gradient(self.xy,axis=0);self.yaw=np.unwrap(np.arctan2(d[:,1],d[:,0]));self.route_tree=cKDTree(self.xy);self.valid=np.isfinite(self.h).astype(np.float32);hh=np.nan_to_num(self.h);gx=cv2.Sobel(hh,cv2.CV_32F,1,0,ksize=3)/(8*self.res);gy=cv2.Sobel(hh,cv2.CV_32F,0,1,ksize=3)/(8*self.res);self.rough=np.sqrt(gx*gx+gy*gy);self.points=np.load(root/'global_points.npy',mmap_mode='r');self.tree=cKDTree(np.asarray(self.points[:,:2]))
 def sample(self,a,world):
  col=(world[...,0]-self.lo[0])/self.res;row=(world[...,1]-self.lo[1])/self.res
  return map_coordinates(a,[row,col],order=1,mode='constant',cval=0)
 def bev(self,x,y,yaw):
  ids=self.tree.query_ball_point([x,y],58);p=np.asarray(self.points[ids],np.float32);j=int(np.argmin(np.sum((self.xy-np.array([x,y]))**2,axis=1)));co,si=np.cos(yaw),np.sin(yaw);dx=p[:,0]-x;dy=p[:,1]-y;local=np.column_stack([co*dx+si*dy,-si*dx+co*dy,p[:,2]-self.z[j]]);n=160;ix=np.floor((local[:,0]+40)/.5).astype(int);iy=np.floor((local[:,1]+40)/.5).astype(int);ok=(ix>=0)&(ix<n)&(iy>=0)&(iy<n)&np.isfinite(local).all(1);ix,iy,z=ix[ok],iy[ok],local[ok,2];f=iy*n+ix;count=np.bincount(f,minlength=n*n).astype(np.float32);zs=np.bincount(f,weights=z,minlength=n*n);z2=np.bincount(f,weights=z*z,minlength=n*n);mean=np.divide(zs,count,out=np.zeros_like(zs),where=count>0);var=np.maximum(np.divide(z2,count,out=np.zeros_like(z2),where=count>0)-mean*mean,0);zmax=np.full(n*n,-np.inf,np.float32);zmin=np.full(n*n,np.inf,np.float32);np.maximum.at(zmax,f,z);np.minimum.at(zmin,f,z);span=np.where(count>0,zmax-zmin,0);return np.stack([np.clip(mean/5,-1,1),np.clip(np.sqrt(var)/2,0,1),np.clip(np.log1p(count)/5,0,1),(span>.7).astype(np.float32),(count>0).astype(np.float32)]).reshape(5,n,n).astype(np.float32)
 def nearest(self,p,last):
  lo=max(0,last-5);hi=min(len(self.xy),last+25);return lo+int(np.argmin(np.sum((self.xy[lo:hi]-p)**2,axis=1)))
 def at_s(self,s):
  i=int(np.searchsorted(self.s,s).clip(1,len(self.s)-1));u=(s-self.s[i-1])/max(self.s[i]-self.s[i-1],1e-6);return (1-u)*self.xy[i-1]+u*self.xy[i]

def scenarios():
 out=[('Nominal',0.,0.)]
 for d in (1.,2.,3.):out += [(f'Lat {d:+.0f}m',d,0),(f'Lat {-d:+.0f}m',-d,0)]
 for d in (5.,10.,15.):out += [(f'Yaw {d:+.0f}deg',0,np.deg2rad(d)),(f'Yaw {-d:+.0f}deg',0,np.deg2rad(-d))]
 return out

def candidate_bank(speed=5.):
 h=np.arange(1,6,dtype=np.float32);out=[]
 for a in np.deg2rad([-15,-7.5,0,7.5,15]):
  for lat in [-3,-1.5,0,1.5,3]:out.append(np.column_stack([speed*h,np.tan(a)*speed*h+lat*(h/5.)**2]))
 return np.asarray(out,np.float32)

def run_group(m,model,risk_model,device,start_i,args):
 cases=scenarios();n=len(cases);p0=m.xy[start_i];yaw0=m.yaw[start_i];normal=np.array([-np.sin(yaw0),np.cos(yaw0)])
 x=np.array([p0[0]+d*normal[0] for _,d,_ in cases]);y=np.array([p0[1]+d*normal[1] for _,d,_ in cases]);yaw=np.array([yaw0+a for _,_,a in cases]);v=np.full(n,args.initial_speed);idx=np.full(n,start_i,int);active=np.ones(n,bool);failed=np.array(['']*n,dtype='<U16');recovered=np.zeros(n,bool);recover_count=np.zeros(n,int);lat_sum=np.zeros(n);lat_max=np.zeros(n);steps=np.zeros(n,int);target_s=m.s[start_i]+args.segment_length;paths=[[x[i],y[i]] for i in range(n)]
 steer=np.zeros(n);acc=np.zeros(n);policy_every=round(args.policy_dt/args.dt)
 for tick in range(int(args.time_limit/args.dt)):
  if tick%policy_every==0 and active.any():
   ids=np.flatnonzero(active);bb=[];ss=[];gg=[]
   for j in ids:
    idx[j]=m.nearest(np.array([x[j],y[j]]),idx[j]);bb.append(m.bev(x[j],y[j],yaw[j]));goal=m.at_s(min(m.s[idx[j]]+max(25.,v[j]*8),m.s[-1]));co,si=np.cos(yaw[j]),np.sin(yaw[j]);delta=goal-np.array([x[j],y[j]]);gg.append([co*delta[0]+si*delta[1],-si*delta[0]+co*delta[1]]);ss.append([v[j],0.,v[j]/args.wheelbase*np.tan(steer[j])])
   bt=torch.from_numpy(np.stack(bb)).to(device);st=torch.tensor(ss,dtype=torch.float32,device=device);gt=torch.tensor(gg,dtype=torch.float32,device=device)
   with torch.no_grad():wp,_=model(bt,st,gt);wp=wp.cpu().numpy()
   if risk_model is not None:
    bank=candidate_bank(args.initial_speed)
    with torch.no_grad():
     for k in range(len(ids)):
      ct=torch.from_numpy(bank).to(device);score=torch.sigmoid(risk_model(bt[k:k+1].expand(len(bank),-1,-1,-1),st[k:k+1].expand(len(bank),-1),gt[k:k+1].expand(len(bank),-1),ct)).cpu().numpy()[:,0];deviation=np.linalg.norm(bank-wp[k][None],axis=2).mean(1);wp[k]=bank[np.argmin(score+args.risk_lambda*deviation)]
   for k,j in enumerate(ids):
    if args.controller=='route':
     q=m.at_s(min(m.s[idx[j]]+10,m.s[-1]))-np.array([x[j],y[j]]);co,si=np.cos(yaw[j]),np.sin(yaw[j]);tx,ty=co*q[0]+si*q[1],-si*q[0]+co*q[1]
    else:tx,ty=wp[k,1]
    ld=max(math.hypot(tx,ty),1.0);steer[j]=np.clip(math.atan2(2*args.wheelbase*ty,ld*ld),-args.max_steer,args.max_steer);desired=args.initial_speed if args.speed_control=='fixed' else np.clip(np.linalg.norm(wp[k,0]),0,args.max_speed);acc[j]=np.clip(1.2*(desired-v[j]),-args.max_brake,args.max_accel)
  ids=np.flatnonzero(active);v[ids]=np.clip(v[ids]+acc[ids]*args.dt,0,args.max_speed);yaw[ids]=wrap(yaw[ids]+v[ids]/args.wheelbase*np.tan(steer[ids])*args.dt);x[ids]+=v[ids]*np.cos(yaw[ids])*args.dt;y[ids]+=v[ids]*np.sin(yaw[ids])*args.dt
  for j in ids:
   idx[j]=m.nearest(np.array([x[j],y[j]]),idx[j]);lat=np.linalg.norm(np.array([x[j],y[j]])-m.xy[idx[j]]);lat_sum[j]+=lat;lat_max[j]=max(lat_max[j],lat);steps[j]+=1
   if lat<1 and abs(wrap(yaw[j]-m.yaw[idx[j]]))<np.deg2rad(5):recover_count[j]+=1
   else:recover_count[j]=0
   recovered[j]|=recover_count[j]>=round(2/args.dt)
   if lat>args.road_width/2-args.vehicle_width/2:active[j]=False;failed[j]='boundary'
   elif m.s[idx[j]]>=target_s:active[j]=False;failed[j]='success'
   elif v[j]<.15 and tick*args.dt>10:active[j]=False;failed[j]='stalled'
   if tick%10==0:paths[j].extend([x[j],y[j]])
  if not active.any():break
 failed[active]='timeout'
 rows=[]
 for j,(name,off,ang) in enumerate(cases):rows.append({'condition':name,'start_s_m':float(m.s[start_i]),'offset_m':off,'heading_deg':float(np.rad2deg(ang)),'outcome':str(failed[j]),'success':int(failed[j]=='success'),'boundary':int(failed[j]=='boundary'),'recovered':int(recovered[j]) if off or ang else 0,'mean_lateral_m':float(lat_sum[j]/max(steps[j],1)),'max_lateral_m':float(lat_max[j]),'progress_m':float(max(0,m.s[idx[j]]-m.s[start_i]))})
 return rows,paths

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--map',type=Path,default=Path('carla_mine/outputs/scene_7_closed_loop'));ap.add_argument('--checkpoint',type=Path,default=Path('em_opm/runs/il_v1/best.pt'));ap.add_argument('--risk-checkpoint',type=Path,default=None);ap.add_argument('--risk-lambda',type=float,default=.02);ap.add_argument('--output',type=Path,default=Path('em_opm/runs/closed_loop_il'));ap.add_argument('--controller',choices=['il','route'],default='il');ap.add_argument('--starts',type=int,default=10);ap.add_argument('--segment-length',type=float,default=120);ap.add_argument('--time-limit',type=float,default=45);ap.add_argument('--dt',type=float,default=.1);ap.add_argument('--policy-dt',type=float,default=.5);ap.add_argument('--wheelbase',type=float,default=4.5);ap.add_argument('--vehicle-width',type=float,default=3.5);ap.add_argument('--road-width',type=float,default=12);ap.add_argument('--initial-speed',type=float,default=5);ap.add_argument('--speed-control',choices=['fixed','policy'],default='fixed');ap.add_argument('--max-speed',type=float,default=12);ap.add_argument('--max-steer',type=float,default=np.deg2rad(30));ap.add_argument('--max-accel',type=float,default=1.5);ap.add_argument('--max-brake',type=float,default=3);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=make_model().to(device);model.load_state_dict(torch.load(a.checkpoint,map_location=device,weights_only=True)['model']);model.eval();risk_model=None
 if a.risk_checkpoint:risk_model=RiskNet().to(device);risk_model.load_state_dict(torch.load(a.risk_checkpoint,map_location=device,weights_only=True)['model']);risk_model.eval()
 m=Map(a.map);valid=np.flatnonzero((m.s>100)&(m.s<m.s[-1]-a.segment_length-100));start_s=np.linspace(m.s[valid[0]],m.s[valid[-1]],a.starts);starts=[int(np.searchsorted(m.s,s)) for s in start_s];rows=[];allpaths=[]
 for k,i in enumerate(starts):r,p=run_group(m,model,risk_model,device,i,a);rows+=r;allpaths+=p;print(f'{k+1}/{len(starts)} starts',flush=True)
 keys=list(rows[0]);
 with (a.output/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 groups=defaultdict(list)
 for r in rows:groups[r['condition']].append(r)
 summary={k:{'episodes':len(v),'success_rate':float(np.mean([x['success'] for x in v])),'boundary_rate':float(np.mean([x['boundary'] for x in v])),'recovery_rate':float(np.mean([x['recovered'] for x in v])) if k!='Nominal' else None,'mean_lateral_m':float(np.mean([x['mean_lateral_m'] for x in v])),'mean_progress_m':float(np.mean([x['progress_m'] for x in v]))} for k,v in groups.items()};overall={'episodes':len(rows),'success_rate':float(np.mean([x['success'] for x in rows])),'boundary_rate':float(np.mean([x['boundary'] for x in rows])),'recovery_rate_perturbed':float(np.mean([x['recovered'] for x in rows if x['condition']!='Nominal']))};report={'setting':'kinematic closed loop on reconstructed M0S7 map','device':str(device),'parameters':vars(a)|{'map':str(a.map),'checkpoint':str(a.checkpoint),'output':str(a.output)},'overall':overall,'conditions':summary,'limitations':['BEV is cropped from an accumulated point map rather than ray-cast LiDAR','12 m road width and vehicle parameters are approximations','no dynamic obstacles in v1']};(a.output/'metrics.json').write_text(json.dumps(report,indent=2,default=str));print(json.dumps({'overall':overall,'conditions':summary},indent=2))
if __name__=='__main__':main()
