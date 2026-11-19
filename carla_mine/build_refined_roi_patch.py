#!/usr/bin/env python3
"""Build a 1 m UE mesh patch for the LiDAR-refined first-20% ROI."""
from pathlib import Path
import argparse
import json
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined'));args=ap.parse_args()
ROOT=args.root
d=np.load(ROOT/'hybrid_dem.npz');z=d['height'];roi=d['roi_mask'].astype(bool)
z0=float(d['z_origin_m']);H,W=z.shape
y,x=np.nonzero(roi);pad=3
y0=max(0,y.min()-pad);y1=min(H,y.max()+pad+1)
x0=max(0,x.min()-pad);x1=min(W,x.max()+pad+1)
q=z[y0:y1,x0:x1];h,w=q.shape
obj=ROOT/'MineFirst20RefinedPatch.obj';mtl=obj.with_suffix('.mtl')
mtl.write_text('newmtl MineFirst20Refined\nKd 1 1 1\nKa 0 0 0\nKs 0 0 0\nmap_Kd MineFullTexture.png\n')
with obj.open('w') as f:
 f.write(f'mtllib {mtl.name}\no MineFirst20RefinedPatch\n')
 # South-to-north vertex order, matching the existing full-map OBJ convention.
 for lr in range(h-1,-1,-1):
  gr=y0+lr
  uy=(H-1-gr-(H-1)/2)*100.0
  for lc in range(w):
   gc=x0+lc;ux=(gc-(W-1)/2)*100.0
   f.write(f'v {ux:.3f} {uy:.3f} {(q[lr,lc]-z0)*100:.3f}\n')
 for lr in range(h-1,-1,-1):
  gr=y0+lr;v=(H-1-gr)/(H-1)
  for lc in range(w):
   gc=x0+lc;f.write(f'vt {gc/(W-1):.8f} {v:.8f}\n')
 f.write('usemtl MineFirst20Refined\ns 1\n')
 for iy in range(h-1):
  a=iy*w+1;c=(iy+1)*w+1
  for ix in range(w-1):
   p,qv,r,s=a+ix,a+ix+1,c+ix,c+ix+1
   f.write(f'f {p}/{p} {qv}/{qv} {s}/{s}\nf {p}/{p} {s}/{s} {r}/{r}\n')
report={'grid_resolution_m':float(d['resolution_m']),'vertices':[w,h],
        'vertex_count':w*h,'triangle_count':2*(w-1)*(h-1),
        'source_rows':[int(y0),int(y1)],'source_cols':[int(x0),int(x1)],
        'purpose':'High-resolution collision and LiDAR ray surface inside first20 ROI; full satellite mesh remains visual context.'}
(ROOT/'roi_patch_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
