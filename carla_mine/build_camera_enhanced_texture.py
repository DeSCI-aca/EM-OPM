#!/usr/bin/env python3
"""Blend the M0S0 camera-ground mosaic over the CARLA satellite texture.

Geometry remains the seven-traversal LiDAR/Copernicus hybrid DEM.  Only pixels
that were projected onto LiDAR-supported ground are trusted as camera texture;
the confidence is feathered into the single-date Sentinel-2 fallback.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter


def robust_color_match(camera: np.ndarray, satellite: np.ndarray, mask: np.ndarray):
    """Match camera low-frequency color to satellite while retaining detail."""
    out = camera.astype(np.float32)
    if mask.sum() < 100:
        return out
    for channel in range(3):
        c = out[:, :, channel][mask]
        s = satellite[:, :, channel].astype(np.float32)[mask]
        c_lo, c_hi = np.percentile(c, [10, 90])
        s_lo, s_hi = np.percentile(s, [10, 90])
        gain = np.clip((s_hi - s_lo) / max(c_hi - c_lo, 1.0), 0.65, 1.45)
        bias = np.median(s) - gain * np.median(c)
        out[:, :, channel] = out[:, :, channel] * gain + bias
    return np.clip(out, 0, 255)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=Path,
                    default=Path("carla_mine/outputs/m0_combined_dem/terrain_rgb.npz"))
    ap.add_argument("--local-dem", type=Path,
                    default=Path("carla_mine/outputs/m0_combined_dem/terrain_dem.npz"))
    ap.add_argument("--hybrid-dem", type=Path,
                    default=Path("carla_mine/outputs/carla_hybrid_dem/hybrid_dem.npz"))
    ap.add_argument("--preview-dir", type=Path,
                    default=Path("carla_mine/outputs/carla_hybrid_dem/carla_preview"))
    ap.add_argument("--texture-size", type=int, default=4096)
    ap.add_argument("--feather-m", type=float, default=12.0)
    args = ap.parse_args()

    camera_pkg = np.load(args.camera)
    local_pkg = np.load(args.local_dem)
    hybrid = np.load(args.hybrid_dem)
    camera_rgb = camera_pkg["rgb"].astype(np.uint8)
    measured = camera_pkg["measured_mask"].astype(np.uint8)
    filled = camera_pkg["filled_mask"].astype(np.uint8)
    local_lo = local_pkg["origin_xy"].astype(np.float64)
    local_res = float(local_pkg["resolution_m"])
    lat0, lon0, _ = local_pkg["geodetic_origin"].astype(np.float64)
    ref_e, ref_n = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True).transform(lon0, lat0)

    west, south = float(hybrid["west"]), float(hybrid["south"])
    res = float(hybrid["resolution_m"])
    height = hybrid["height"]
    east = west + (height.shape[1] - 1) * res
    north = south + (height.shape[0] - 1) * res
    n = args.texture_size
    dst_transform = from_bounds(west, south, east, north, n, n)
    src_transform = from_origin(ref_e + local_lo[0] - local_res / 2,
                                ref_n + local_lo[1] + (camera_rgb.shape[0] - 0.5) * local_res,
                                local_res, local_res)

    camera_north_up = np.flipud(camera_rgb)
    trusted_north_up = np.flipud(np.maximum(measured, filled)).astype(np.uint8)
    direct_north_up = np.flipud(measured).astype(np.uint8)
    warped = np.zeros((n, n, 3), np.uint8)
    trusted = np.zeros((n, n), np.uint8)
    direct = np.zeros((n, n), np.uint8)
    for channel in range(3):
        reproject(camera_north_up[:, :, channel], warped[:, :, channel],
                  src_transform=src_transform, src_crs="EPSG:32650",
                  dst_transform=dst_transform, dst_crs="EPSG:32650",
                  resampling=Resampling.bilinear)
    reproject(trusted_north_up, trusted, src_transform=src_transform, src_crs="EPSG:32650",
              dst_transform=dst_transform, dst_crs="EPSG:32650", resampling=Resampling.nearest)
    reproject(direct_north_up, direct, src_transform=src_transform, src_crs="EPSG:32650",
              dst_transform=dst_transform, dst_crs="EPSG:32650", resampling=Resampling.nearest)

    args.preview_dir.mkdir(parents=True, exist_ok=True)
    active = args.preview_dir / "MineHybridTexture.png"
    satellite_backup = args.preview_dir / "MineHybridTexture_Satellite.png"
    if not satellite_backup.exists():
        shutil.copy2(active, satellite_backup)
    satellite = cv2.cvtColor(cv2.imread(str(satellite_backup)), cv2.COLOR_BGR2RGB)
    satellite = cv2.resize(satellite, (n, n), interpolation=cv2.INTER_CUBIC)

    valid = trusted > 0
    warped = robust_color_match(warped, satellite, direct > 0)
    # Rasterized LiDAR returns are sparse even with dense temporal sampling.
    # Normalized convolution removes point/scan-line stippling without leaking
    # black invalid pixels into the camera mosaic.
    kernel_sigma = 2.5
    norm = gaussian_filter(valid.astype(np.float32), kernel_sigma)
    smooth_camera = np.empty_like(warped, dtype=np.float32)
    for channel in range(3):
        numerator = gaussian_filter(warped[:, :, channel] * valid, kernel_sigma)
        smooth_camera[:, :, channel] = numerator / np.maximum(norm, 1e-4)
    warped = np.where(valid[..., None], 0.25 * warped + 0.75 * smooth_camera,
                      smooth_camera)
    # Suppress isolated projection speckles and feather only inward from the
    # camera-supported boundary; satellite remains untouched elsewhere.
    support = (gaussian_filter(valid.astype(np.float32), 1.0) > 0.35)
    inside_distance = distance_transform_edt(support)
    metres_per_pixel = (east - west) / n
    alpha = np.clip(inside_distance * metres_per_pixel / args.feather_m, 0, 1)
    alpha *= np.where(direct > 0, 1.0, 0.82)
    alpha = gaussian_filter(alpha.astype(np.float32), 0.8)
    result = satellite.astype(np.float32) * (1 - alpha[..., None]) + warped * alpha[..., None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    camera_path = args.preview_dir / "MineHybridTexture_CameraEnhanced.png"
    mask_path = args.preview_dir / "MineHybridTexture_CameraConfidence.png"
    geotiff_path = args.preview_dir / "MineHybridTexture_CameraEnhanced_utm50n.tif"
    confidence_tif = args.preview_dir / "MineHybridTexture_CameraConfidence_utm50n.tif"
    cv2.imwrite(str(camera_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), np.clip(alpha * 255, 0, 255).astype(np.uint8))
    rgb_profile = {
        "driver": "GTiff", "height": n, "width": n, "count": 3,
        "dtype": "uint8", "crs": "EPSG:32650", "transform": dst_transform,
        "compress": "deflate", "tiled": True, "photometric": "RGB",
    }
    with rasterio.open(geotiff_path, "w", **rgb_profile) as dst:
        for channel in range(3):
            dst.write(result[:, :, channel], channel + 1)
    mask_profile = dict(rgb_profile)
    mask_profile.update(count=1, photometric="MINISBLACK")
    with rasterio.open(confidence_tif, "w", **mask_profile) as dst:
        dst.write(np.clip(alpha * 255, 0, 255).astype(np.uint8), 1)
    yy, xx = np.nonzero(support)
    comparison_path = args.preview_dir / "MineHybridTexture_CameraComparison.png"
    if len(xx):
        pad = 120
        x0, x1 = max(0, int(xx.min()) - pad), min(n, int(xx.max()) + pad + 1)
        y0, y1 = max(0, int(yy.min()) - pad), min(n, int(yy.max()) + pad + 1)
        before = satellite[y0:y1, x0:x1]
        after = result[y0:y1, x0:x1]
        target_h = 900
        scale = min(1.0, target_h / max(before.shape[0], 1))
        size = (max(1, int(before.shape[1] * scale)), max(1, int(before.shape[0] * scale)))
        before = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
        after = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
        panel = np.concatenate([before, after], axis=1)
        panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.putText(panel, "Sentinel-2", (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(panel, "M0S0 camera + Sentinel-2", (size[0] + 20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imwrite(str(comparison_path), panel)
    shutil.copy2(camera_path, active)
    report = {
        "geometry": "seven-traversal LiDAR plus Copernicus GLO-30 hybrid DEM",
        "camera_texture_source": "M0S0 synchronized six-camera imagery only",
        "fallback_texture": "single-date Sentinel-2 visual image",
        "texture_size": [n, n],
        "camera_supported_pixels": int(support.sum()),
        "camera_supported_percent": float(100 * support.mean()),
        "direct_camera_pixels": int((direct > 0).sum()),
        "feather_m": args.feather_m,
        "camera_normalized_smoothing_sigma_px": kernel_sigma,
        "active_texture": str(active),
        "satellite_backup": str(satellite_backup),
        "comparison": str(comparison_path),
        "georeferenced_rgb_tif": str(geotiff_path),
        "georeferenced_confidence_tif": str(confidence_tif),
    }
    (args.preview_dir / "camera_texture_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
