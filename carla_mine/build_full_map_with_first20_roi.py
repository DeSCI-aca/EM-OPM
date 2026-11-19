#!/usr/bin/env python3
"""Keep the full satellite map, but apply LiDAR/drivable optimization only in the first20 ROI."""
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path("carla_mine/outputs/carla_route_buffer_1km_utm_aligned/lidar_all_points_1m/hybrid_dem.npz")
ROI = Path("carla_mine/outputs/m0s0_first20_rl_region")
SAT_DEM = Path("carla_mine/external_dem/copernicus_glo30/mine_dem_glo30_utm50n.tif")
FULL_TEXTURE = Path("carla_mine/outputs/carla_route_buffer_1km_utm_aligned/lidar_all_points_1m/carla_map/carla_preview_m0s7_smoothed/MineHybridTexture.png")
OUT = Path("carla_mine/outputs/m0s0_first20_rl_full_map")


def write_obj(path, z, z0, res=1., stride=4):
    q=np.flipud(z)[::stride,::stride];h,w=q.shape;step=res*stride*100
    x0,y0=-(w-1)*step/2,-(h-1)*step/2
    mtl=path.with_suffix('.mtl');mtl.write_text('newmtl MineFirst20Full\nKd 1 1 1\nKa 0 0 0\nKs 0 0 0\nmap_Kd MineFullTexture.png\n')
    with path.open('w') as f:
        f.write(f'mtllib {mtl.name}\no MineFirst20FullTerrain\n')
        for iy in range(h):
            for ix in range(w):f.write(f'v {x0+ix*step:.3f} {y0+iy*step:.3f} {(q[iy,ix]-z0)*100:.3f}\n')
        for iy in range(h):
            for ix in range(w):f.write(f'vt {ix/(w-1):.7f} {iy/(h-1):.7f}\n')
        f.write('usemtl MineFirst20Full\ns 1\n')
        for iy in range(h-1):
            a=iy*w+1;c=(iy+1)*w+1
            for ix in range(w-1):
                p,qv,r,s=a+ix,a+ix+1,c+ix,c+ix+1
                f.write(f'f {p}/{p} {qv}/{qv} {s}/{s}\nf {p}/{p} {s}/{s} {r}/{r}\n')
    return [w,h]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    b=np.load(BASE);shape=b['height'].shape;res=float(b['resolution_m']);west=float(b['west']);south=float(b['south'])
    north=south+(shape[0]-1)*res;east=west+(shape[1]-1)*res
    transform=from_origin(west-res/2,north+res/2,res,res)
    satellite=np.full(shape,np.nan,np.float32)
    with rasterio.open(SAT_DEM) as src:
        reproject(rasterio.band(src,1),satellite,src_transform=src.transform,src_crs=src.crs,
                  dst_transform=transform,dst_crs='EPSG:32650',src_nodata=src.nodata,dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    valid=np.isfinite(satellite)
    if not valid.all():
        _,ind=distance_transform_edt(~valid,return_indices=True);satellite[~valid]=satellite[ind[0][~valid],ind[1][~valid]]

    r=np.load(ROI/'hybrid_dem.npz');rz=r['height'].astype(np.float32);trusted=r['lidar_trusted'].astype(bool);roi=r['roi_mask'].astype(bool)
    rw,rs=float(r['west']),float(r['south']);rn=rs+(rz.shape[0]-1)*res
    c0=int(round((rw-west)/res)); south0=int(round((rs-south)/res)); row0=shape[0]-1-(south0+rz.shape[0]-1)
    full_lidar=np.full(shape,np.nan,np.float32);full_trusted=np.zeros(shape,bool);full_roi=np.zeros(shape,bool)
    sl=np.s_[row0:row0+rz.shape[0],c0:c0+rz.shape[1]]
    full_lidar[sl]=rz;full_trusted[sl]=trusted;full_roi[sl]=roi
    delta=(full_lidar-satellite)[full_trusted];med=float(np.median(delta));mad=float(np.median(np.abs(delta-med)))
    keep=np.abs(delta-med)<=max(3.,3*1.4826*mad);offset=float(np.median(delta[keep]));satellite+=offset
    dist,nearest=distance_transform_edt(~full_trusted,return_indices=True)
    near=full_lidar[nearest[0],nearest[1]];near=gaussian_filter(near,1.0);near[full_trusted]=full_lidar[full_trusted]
    weight=np.clip(1-dist/50.,0,1)**2;weight[full_trusted]=1;weight*=full_roi
    fused=satellite*(1-weight)+near*weight
    z0=float(np.floor(np.min(fused)-1))

    full_drive=np.zeros(shape,bool);local_drive=np.load(ROI/'drivable.npy');full_drive[sl]=local_drive
    np.save(OUT/'drivable.npy',full_drive);cv2.imwrite(str(OUT/'drivable.png'),full_drive.astype(np.uint8)*255)
    np.savez_compressed(OUT/'hybrid_dem.npz',height=fused,resolution_m=res,crs='EPSG:32650',west=west,south=south,
                        z_origin_m=z0,lidar_weight=weight,roi_mask=full_roi,drivable=full_drive)
    shutil.copy2(FULL_TEXTURE,OUT/'MineFullTexture.png')
    mesh=write_obj(OUT/'MineFirst20FullTerrain.obj',fused,z0,res,4)

    # Keep only ROI routes, converted from ROI-local coordinates to full-map local coordinates.
    with (ROI/'routes_roi.csv').open() as src,(OUT/'routes_roi.csv').open('w',newline='') as dst:
        reader=csv.DictReader(src);fields=['scene','interval','x_east_m','y_north_m','z_relative_m'];w=csv.DictWriter(dst,fieldnames=fields);w.writeheader()
        for q in reader:
            e=float(q['x_east_m'])+rw;n=float(q['y_north_m'])+rs
            w.writerow(dict(scene=q['scene'],interval=q['interval'],x_east_m=e-west,y_north_m=n-south,
                            z_relative_m=float(q['z_m'])-z0))
    # Translate start/goal coordinates from local-ROI center to full-map center.
    local_report=json.loads((ROI/'report.json').read_text());lcx,lcy=local_report['carla_center_utm'];fcx,fcy=(west+east)/2,(south+north)/2
    nav=OUT/'navigation';nav.mkdir(exist_ok=True)
    for filename in ['spawn_points.csv','terminal_points.csv']:
        rows=list(csv.DictReader((ROI/'navigation'/filename).open()))
        for q in rows:
            # Source CARLA y is mirrored.
            e=lcx+float(q['carla_x_m']);n=lcy-float(q['carla_y_m'])
            q['carla_x_m']=e-fcx;q['carla_y_m']=-(n-fcy)
            q['z_relative_m']=float(q['z_relative_m'])+float(r['z_origin_m'])-z0
        with (nav/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    shutil.copy2(ROI/'navigation/rl_task.json',nav/'rl_task.json')
    shutil.copy2(ROI/'roi_definition.npz',OUT/'roi_definition.npz')
    shutil.copy2(ROI/'camera_manifest.csv',OUT/'camera_manifest.csv')
    shutil.copy2(ROI/'lidar_manifest.csv',OUT/'lidar_manifest.csv')

    # QA: full satellite terrain, ROI boundary, and ROI-only drivable area.
    lo,hi=np.percentile(fused,[1,99]);gray=(np.clip((fused-lo)/(hi-lo),0,1)*180).astype(np.uint8);qa=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    qa[full_roi]=(0.7*qa[full_roi]+0.3*np.array([150,100,20])).astype(np.uint8)
    qa[full_drive]=(0.25*qa[full_drive]+0.75*np.array([30,210,60])).astype(np.uint8)
    cv2.imwrite(str(OUT/'full_map_roi_drivable_preview.png'),qa)
    report={'terrain_extent_preserved':True,'satellite_texture_preserved':True,'bounds_utm':[west,south,east,north],
            'extent_m':[east-west,north-south],'shape':list(shape),'resolution_m':res,'roi_area_m2':float(full_roi.sum()*res*res),
            'drivable_outside_roi_cells':int((full_drive&~full_roi).sum()),'lidar_trusted_cells':int(full_trusted.sum()),
            'satellite_vertical_offset_from_roi_m':offset,'alignment_mad_m':mad,'mesh_vertices_xy':mesh,
            'policy':'Full satellite DEM/texture retained; only first20 ROI receives selected seven-run LiDAR optimization and drivable routes.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
