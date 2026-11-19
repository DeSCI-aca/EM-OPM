#!/usr/bin/env python3
"""Apply a bounded, feathered DEM smoothing only at CARLA contact hotspots."""
from pathlib import Path
import argparse,csv,json,shutil
import cv2
import numpy as np
from scipy import ndimage

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--input',type=Path,default=Path('carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined'))
 ap.add_argument('--frames',type=Path,default=Path('carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined/carla_vehicle_contact/frames.csv'))
 ap.add_argument('--output',type=Path,default=Path('carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined_v2'))
 ap.add_argument('--core-radius',type=float,default=4.0);ap.add_argument('--feather',type=float,default=3.0)
 ap.add_argument('--max-correction',type=float,default=.5)
 args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 d=np.load(args.input/'hybrid_dem.npz');z=d['height'].astype(np.float32);roi=d['roi_mask'].astype(bool)
 res=float(d['resolution_m']);west,south=float(d['west']),float(d['south']);north=south+(z.shape[0]-1)*res
 rows=list(csv.DictReader(args.frames.open()));centers=[]
 for r in rows:
  gaps=np.array([float(r[k]) for k in ['gap_fl_m','gap_fr_m','gap_rl_m','gap_rr_m']])
  if gaps.min()<-.25 or gaps.max()>.20 or np.ptp(gaps)>.40:centers.append((float(r['utm_e']),float(r['utm_n'])))
 seed=np.zeros(z.shape,bool)
 for e,n in centers:
  x=int(round((e-west)/res));y=int(round((north-n)/res))
  if 0<=x<z.shape[1] and 0<=y<z.shape[0]:seed[y,x]=True
 dist=ndimage.distance_transform_edt(~seed)*res
 core=dist<=args.core_radius;support=(dist<=args.core_radius+args.feather)&roi
 weight=np.clip((args.core_radius+args.feather-dist)/args.feather,0,1);weight[core]=1;weight*=roi
 # A 5x5 robust median rejects isolated triangles; the 2 m Gaussian joins the
 # remaining cells into a suspension-friendly surface without flattening grade.
 med=ndimage.median_filter(z,size=5,mode='nearest');target=ndimage.gaussian_filter(med,sigma=2.0)
 delta=np.clip(target-z,-args.max_correction,args.max_correction)*weight
 out=z+delta
 payload={k:d[k] for k in d.files if k!='height'};payload['height']=out
 np.savez_compressed(args.output/'hybrid_dem.npz',**payload)
 for name in ['MineFullTexture.png','drivable.npy','roi_definition.npz','routes_roi.csv','lidar_consistency_report.json']:
  p=args.input/name
  if p.exists():shutil.copy2(p,args.output/name)
 if (args.input/'navigation').exists():shutil.copytree(args.input/'navigation',args.output/'navigation',dirs_exist_ok=True)
 vis=np.zeros((*z.shape,3),np.uint8);scale=max(float(np.max(np.abs(delta))),.01)
 vis=cv2.applyColorMap((np.clip(delta/(2*scale)+.5,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO);vis[~support]//=6
 cv2.imwrite(str(args.output/'contact_hotspot_smoothing.png'),vis)
 report={'source_hotspot_frames':len(centers),'core_radius_m':args.core_radius,'feather_m':args.feather,
  'support_cells':int(support.sum()),'changed_cells':int((np.abs(delta)>1e-4).sum()),
  'correction_abs_p50_p95_max_m':[float(np.percentile(np.abs(delta[support]),p)) for p in [50,95,100]],
  'outside_support_unchanged':bool(np.array_equal(out[~support],z[~support])),
  'policy':'Only severe wheel contact hotspots; full terrain, texture, ROI and routes retained.'}
 (args.output/'contact_smoothing_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
