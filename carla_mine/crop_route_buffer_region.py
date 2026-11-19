#!/usr/bin/env python3
"""Crop the hybrid DEM to the seven-route envelope plus a metric buffer."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path,
                    default=Path("carla_mine/outputs/carla_hybrid_dem/hybrid_dem.npz"))
    ap.add_argument("--routes", type=Path,
                    default=Path("carla_mine/outputs/carla_hybrid_dem/routes_carla.csv"))
    ap.add_argument("--output", type=Path,
                    default=Path("carla_mine/outputs/carla_route_buffer_1km"))
    ap.add_argument("--buffer-m", type=float, default=1000.0)
    ap.add_argument("--mesh-stride", type=int, default=4)
    ap.add_argument("--texture", type=Path,
                    default=Path("carla_mine/outputs/carla_route_buffer_1km/government_ortho/MineHybridTexture_GovernmentOrtho.png"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    src = np.load(args.source)
    dem = src["height"].astype(np.float32)  # north-up
    west0, south0 = float(src["west"]), float(src["south"])
    res, z0 = float(src["resolution_m"]), float(src["z_origin_m"])
    routes = {}
    with args.routes.open() as stream:
        for row in csv.DictReader(stream):
            routes.setdefault(row["scene"], []).append(
                [float(row["x_east_m"]), float(row["y_north_m"]),
                 float(row["z_relative_m"])])
    all_points = np.concatenate([np.asarray(v) for v in routes.values()])
    route_min = all_points[:, :2].min(0) + [west0, south0]
    route_max = all_points[:, :2].max(0) + [west0, south0]
    desired_min = route_min - args.buffer_m
    desired_max = route_max + args.buffer_m

    # Snap outward to the existing 2 m grid.
    col0 = max(0, int(np.floor((desired_min[0] - west0) / res)))
    col1 = min(dem.shape[1] - 1, int(np.ceil((desired_max[0] - west0) / res)))
    # Convert south-up vertex indices to north-up array rows.
    south_index0 = max(0, int(np.floor((desired_min[1] - south0) / res)))
    south_index1 = min(dem.shape[0] - 1, int(np.ceil((desired_max[1] - south0) / res)))
    row0 = dem.shape[0] - 1 - south_index1
    row1 = dem.shape[0] - south_index0
    cropped = dem[row0:row1, col0:col1 + 1]
    west = west0 + col0 * res
    south = south0 + south_index0 * res
    east = west + (cropped.shape[1] - 1) * res
    north = south + (cropped.shape[0] - 1) * res

    np.savez_compressed(args.output / "hybrid_dem.npz", height=cropped,
                        resolution_m=res, crs="EPSG:32650", west=west,
                        south=south, z_origin_m=z0)
    transform = from_origin(west - res / 2, north + res / 2, res, res)
    profile = {"driver": "GTiff", "height": cropped.shape[0],
               "width": cropped.shape[1], "count": 1, "dtype": "float32",
               "crs": "EPSG:32650", "transform": transform,
               "compress": "deflate", "tiled": True}
    with rasterio.open(args.output / "hybrid_dem_utm50n.tif", "w", **profile) as dst:
        dst.write(cropped, 1)

    # Preserve routes in coordinates local to the cropped CARLA map.
    with (args.output / "routes_carla.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "x_east_m", "y_north_m", "z_relative_m"])
        for name, points in routes.items():
            for x, y, z in points:
                writer.writerow([name, x + west0 - west, y + south0 - south, z])

    # Unreal 16-bit heightmap using the original vertical scale convention.
    z_min, z_max = float(np.floor(cropped.min() - 2)), float(np.ceil(cropped.max() + 2))
    z_center = (z_min + z_max) / 2
    z_scale = (z_max - z_min) / 5.12
    height16 = np.clip(np.rint(32768 + (cropped - z_center) * 12800 / z_scale),
                       0, 65535).astype(np.uint16)
    cv2.imwrite(str(args.output / "carla_heightmap.png"), np.flipud(height16))

    preview = args.output / "carla_preview"
    preview.mkdir(parents=True, exist_ok=True)
    texture_out = preview / "MineHybridTexture.png"
    if args.texture.exists():
        shutil.copy2(args.texture, texture_out)
    stride = args.mesh_stride
    mesh_z = np.flipud(cropped)[::stride, ::stride]
    mh, mw = mesh_z.shape
    step_cm = res * stride * 100.0
    x0, y0 = -(mw - 1) * step_cm / 2, -(mh - 1) * step_cm / 2
    mtl = preview / "MineHybridTerrain.mtl"
    mtl.write_text("newmtl MineHybridSurface\nKd 1 1 1\nKa 0 0 0\nKs 0 0 0\n"
                   "map_Kd MineHybridTexture.png\n")
    obj = preview / "MineHybridTerrain.obj"
    with obj.open("w") as stream:
        stream.write("mtllib MineHybridTerrain.mtl\no MineHybridTerrain\n")
        for iy in range(mh):
            for ix in range(mw):
                stream.write(f"v {x0 + ix*step_cm:.3f} {y0 + iy*step_cm:.3f} "
                             f"{(mesh_z[iy,ix]-z0)*100:.3f}\n")
        for iy in range(mh):
            for ix in range(mw):
                stream.write(f"vt {ix/(mw-1):.7f} {iy/(mh-1):.7f}\n")
        stream.write("usemtl MineHybridSurface\ns 1\n")
        for iy in range(mh - 1):
            row, nxt = iy * mw + 1, (iy + 1) * mw + 1
            for ix in range(mw - 1):
                a, b, c, d = row+ix, row+ix+1, nxt+ix, nxt+ix+1
                stream.write(f"f {a}/{a} {b}/{b} {d}/{d}\n"
                             f"f {a}/{a} {d}/{d} {c}/{c}\n")

    report = {
        "buffer_each_side_m": args.buffer_m,
        "route_bounds_utm": [*route_min.tolist(), *route_max.tolist()],
        "crop_bounds_utm": [west, south, east, north],
        "extent_m": [east - west, north - south],
        "size_vertices": [int(cropped.shape[1]), int(cropped.shape[0])],
        "resolution_m": res, "crs": "EPSG:32650",
        "unreal_z_scale": z_scale, "unreal_actor_z_m": z_center,
        "preview_mesh_vertices": [mw, mh],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
