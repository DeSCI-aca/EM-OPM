#!/usr/bin/env python3
"""Write OpenDRIVE for the ROI route in full-map-centered CARLA coordinates."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('carla_mine/outputs/m0s0_first20_rl_full_map')
ROI=Path('carla_mine/outputs/m0s0_first20_rl_region')

def main():
    d=np.load(ROOT/'hybrid_dem.npz');z=d['height'];west,south=float(d['west']),float(d['south']);res=float(d['resolution_m']);north=south+(z.shape[0]-1)*res;east=west+(z.shape[1]-1)*res;cx,cy=(west+east)/2,(south+north)/2;z0=float(d['z_origin_m'])
    core=np.load(ROI/'roi_definition.npz')['core_utm'];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(core[:,:2],axis=0),axis=1))];s=np.arange(0,arc[-1]+1,2.)
    p=np.column_stack([np.interp(s,arc,core[:,i]) for i in range(3)])
    col=np.clip(np.rint((p[:,0]-west)/res).astype(int),0,z.shape[1]-1);row=np.clip(np.rint((north-p[:,1])/res).astype(int),0,z.shape[0]-1)
    x=p[:,0]-cx;y=-(p[:,1]-cy);h=z[row,col]-z0;hdg=np.unwrap(np.arctan2(np.gradient(y),np.gradient(x)))
    out=ROOT/'MineFirst20RLFull.xodr'
    with out.open('w') as f:
        f.write('<?xml version="1.0" standalone="yes"?>\n<OpenDRIVE>\n  <header revMajor="1" revMinor="4" name="MineFirst20RLFull" version="1.00"/>\n')
        f.write(f'  <road name="M0S0_first20" length="{s[-1]:.3f}" id="1" junction="-1">\n    <planView>\n')
        for i in range(len(s)-1):f.write(f'      <geometry s="{s[i]:.3f}" x="{x[i]:.3f}" y="{y[i]:.3f}" hdg="{hdg[i]:.8f}" length="{s[i+1]-s[i]:.3f}"><line/></geometry>\n')
        f.write('    </planView>\n    <elevationProfile>\n')
        for i in range(len(s)-1):f.write(f'      <elevation s="{s[i]:.3f}" a="{h[i]:.3f}" b="{(h[i+1]-h[i])/max(s[i+1]-s[i],1e-6):.8f}" c="0" d="0"/>\n')
        f.write('    </elevationProfile>\n    <lanes><laneSection s="0"><left><lane id="1" type="driving" level="false"><width sOffset="0" a="6" b="0" c="0" d="0"/></lane></left><center><lane id="0" type="none" level="false"/></center><right><lane id="-1" type="driving" level="false"><width sOffset="0" a="6" b="0" c="0" d="0"/></lane></right></laneSection></lanes>\n  </road>\n</OpenDRIVE>\n')
    print(out,s[-1])
if __name__=='__main__':main()
