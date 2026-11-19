#!/usr/bin/env python3
"""Extract camera-detected dynamic actors and associate them with LiDAR points.

The output is JSONL: one record per detection with image, base-frame and map-frame
measurements.  It is intended to construct replayable mine-traffic scenarios, not
to provide manually verified ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
]
KEEP = {0, 1, 2, 3, 5, 7}

# X_cam0 = R_i0 X_cami + t_i0
R_I0 = [
    np.eye(3),
    np.array([[.498563814,.009789365,.866797738],[-.005205524,.999951850,-.008298820],[-.866837811,-.000375087,.498590099]]),
    np.array([[-.512235,.000303,.858846],[-.007906,.999956,-.005067],[-.858809,-.009386,-.512211]]),
    np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]),
    np.array([[-.47215392,.01972107,-.88129551],[-.00294644,.9997086,.02394907],[.88151197,.01390425,-.47195765]]),
    np.array([[.512235245,.000302668,-.858845985],[.007905981,.999955973,.005066733],[.858808829,-.009385755,.512210534]]),
]
T_I0 = [
    np.zeros(3), np.array([.187680,.005917,-.082962]),
    np.array([.188480,.001000,-.362641]), np.array([0.,.005917,-.442623]),
    np.array([-.176386,.000944,-.362641]), np.array([-.179698,.001028,-.090938]),
]
# FAST-LIVO2 file declares camera-to-LiDAR: X_l = R_cl X_cam0 + P_cl.
R_CL = np.array([[.021616,-.999762,-.003075],[.011367,.003321,-.999930],[.999702,.021580,.011436]])
P_CL = np.array([.002104,-.130557,-.208898])


def stamp(path: Path) -> float:
    return float(path.stem)


def nearest(files, times, t, max_dt):
    i = int(np.searchsorted(times, t))
    ids = [j for j in (i - 1, i) if 0 <= j < len(files)]
    if not ids:
        return None
    j = min(ids, key=lambda k: abs(times[k] - t))
    return files[j] if abs(times[j] - t) <= max_dt else None


def load_poses(scene: Path):
    rows = []
    with (scene / "gnss_odom.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append([float(r[k]) for k in ("timestamp", "x", "y", "z", "qx", "qy", "qz", "qw")])
    return np.asarray(rows, np.float64)


def quat_R(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    s = 2. / n
    return np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],
                     [s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],
                     [s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])


def pose_at(poses, t):
    i = int(np.argmin(np.abs(poses[:, 0] - t)))
    r = poses[i]
    return quat_R(r[4:8]), r[1:4], abs(r[0] - t)


def intrinsics(scene, cam):
    names = ["front", "right_front", "right_back", "back", "left_back", "left_front"]
    p = scene / "calib" / "Intrinsic" / f"{names[cam]}-camchain.yaml"
    d = yaml.safe_load(p.read_text())
    c = next(iter(d.values()))
    fx, fy, cx, cy = c["intrinsics"]
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]], np.float64), np.asarray(c["distortion_coeffs"], np.float64)


def read_points(path):
    a = np.load(path)
    if isinstance(a, np.lib.npyio.NpzFile):
        a = a[a.files[0]]
    if getattr(a.dtype, "names", None):
        a = np.column_stack([a[n] for n in ("x", "y", "z")])
    return np.asarray(a[:, :3], np.float64)


def project_lidar(points_l, cam, K, dist):
    # LiDAR -> cam0 -> cam_i.
    p0 = (R_CL.T @ (points_l - P_CL).T).T
    pi = (R_I0[cam].T @ (p0 - T_I0[cam]).T).T
    valid = pi[:, 2] > .5
    if not np.any(valid):
        return np.empty((0, 2), np.float64), np.empty((0, 3), np.float64)
    uv, _ = cv2.projectPoints(pi[valid], np.zeros(3), np.zeros(3), K, dist)
    if uv is None:
        return np.empty((0, 2), np.float64), np.empty((0, 3), np.float64)
    return uv[:, 0, :], points_l[valid]


def detect_darknet(net, image, conf, nms):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255., (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    boxes, scores, classes = [], [], []
    for out in net.forward(net.getUnconnectedOutLayersNames()):
        for d in out:
            cls = int(np.argmax(d[5:])); score = float(d[4] * d[5 + cls])
            if cls not in KEEP or score < conf: continue
            cx, cy, bw, bh = d[:4] * [w,h,w,h]
            boxes.append([int(cx-bw/2),int(cy-bh/2),int(bw),int(bh)])
            scores.append(score); classes.append(cls)
    ids = cv2.dnn.NMSBoxes(boxes, scores, conf, nms)
    return [(boxes[int(i)], scores[int(i)], classes[int(i)]) for i in np.asarray(ids).reshape(-1)] if len(ids) else []

def detect_ultralytics(model, image, conf, device):
    result = model.predict(image, conf=conf, classes=sorted(KEEP), device=device,
                           imgsz=960, verbose=False)[0]
    out=[]
    if result.boxes is None: return out
    for xyxy, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        x1,y1,x2,y2=xyxy; out.append(([int(x1),int(y1),int(x2-x1),int(y2-y1)],float(score),int(cls)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--model", type=Path, default=Path("dust_removal/models/yolov4-tiny"))
    ap.add_argument("--backend", choices=("ultralytics","darknet"), default="ultralytics")
    ap.add_argument("--yolo-model", default="yolo11n.pt")
    ap.add_argument("--device", default="0", help="CUDA device such as 0 or cpu")
    ap.add_argument("--output", type=Path, default=Path("em_opm/data/dynamic_actors"))
    ap.add_argument("--cams", default="0,1,2,3,4,5")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=.25)
    ap.add_argument("--nms", type=float, default=.45)
    ap.add_argument("--max-lidar-dt", type=float, default=.12)
    ap.add_argument("--min-points", type=int, default=3)
    ap.add_argument("--visualize", action="store_true")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    visdir = a.output / "overlays"; visdir.mkdir(exist_ok=True)
    if a.backend == "ultralytics":
        from ultralytics import YOLO
        net=YOLO(a.yolo_model)
        print(f"detector=ultralytics model={a.yolo_model} device={a.device}", flush=True)
    else:
        net=cv2.dnn.readNetFromDarknet(str(a.model/"yolov4-tiny.cfg"),str(a.model/"yolov4-tiny.weights"))
        print("detector=opencv-darknet device=cpu", flush=True)
    poses = load_poses(a.scene)
    lidar = sorted((a.scene/"LiDAR").glob("*.npy"), key=stamp)
    lt = np.array([stamp(p) for p in lidar])
    scene_name = a.scene.name
    fout = (a.output/f"{scene_name}.jsonl").open("w")
    count = frames = localized = 0
    for cam in map(int, a.cams.split(',')):
        K, dist = intrinsics(a.scene, cam)
        images = sorted((a.scene/f"mav0/cam{cam}/data").glob("*.png"), key=stamp)[::a.stride]
        if a.limit: images = images[:a.limit]
        for image_path in images:
            t = stamp(image_path); im = cv2.imread(str(image_path))
            if im is None: continue
            ds=(detect_ultralytics(net,im,a.conf,a.device) if a.backend=="ultralytics" else detect_darknet(net,im,a.conf,a.nms)); frames += 1
            lp = nearest(lidar, lt, t, a.max_lidar_dt)
            uv = pl = None
            if lp is not None and ds:
                uv, pl = project_lidar(read_points(lp), cam, K, dist)
            Rmb, tmb, pose_dt = pose_at(poses, t)
            for box, score, cls in ds:
                x,y,w,h = box; rec = {"scene":scene_name,"timestamp":t,"camera":cam,
                    "image":str(image_path),"class":COCO[cls],"class_id":cls,
                    "confidence":score,"bbox_xywh":[x,y,w,h],"lidar_timestamp":stamp(lp) if lp else None}
                if uv is not None:
                    inside=(uv[:,0]>=x)&(uv[:,0]<=x+w)&(uv[:,1]>=y)&(uv[:,1]<=y+h)
                    q=pl[inside]
                    # Prefer the nearest coherent depth group to reject background points.
                    if len(q)>=a.min_points:
                        ranges=np.linalg.norm(q[:,:2],axis=1); near=np.percentile(ranges,25)
                        q=q[ranges <= near+2.5]
                    if len(q)>=a.min_points:
                        pb=np.median(q,axis=0); pm=Rmb@pb+tmb
                        rec.update({"num_lidar_points":int(len(q)),"position_base_m":pb.tolist(),
                                    "position_map_m":pm.tolist(),"pose_dt_s":pose_dt})
                        localized += 1
                fout.write(json.dumps(rec)+"\n"); count += 1
                if a.visualize:
                    color=(0,255,0) if "position_base_m" in rec else (0,165,255)
                    cv2.rectangle(im,(x,y),(x+w,y+h),color,3)
                    label=f"{COCO[cls]} {score:.2f}"+(f" {np.linalg.norm(rec['position_base_m'][:2]):.1f}m" if "position_base_m" in rec else "")
                    cv2.putText(im,label,(max(0,x),max(25,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
            if a.visualize and ds:
                cv2.imwrite(str(visdir/f"{scene_name}_cam{cam}_{image_path.stem}.jpg"),im)
    fout.close()
    summary={"scene":scene_name,"backend":a.backend,"device":a.device,"frames":frames,"detections":count,"lidar_localized":localized,
             "localization_rate":localized/max(count,1),"output":str(a.output/f'{scene_name}.jsonl'),
             "note":"Automatic pseudo-labels; visually verify overlays before training."}
    (a.output/f"{scene_name}_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
