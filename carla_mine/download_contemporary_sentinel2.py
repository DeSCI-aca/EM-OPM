#!/usr/bin/env python3
"""Download all useful Sentinel-2 L2A bands cropped around the mine."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


BBOX = (115.9645, 43.9519, 116.0104, 43.9767)  # GPS extent plus about 1 km
RASTER_ASSETS = (
    "visual", "blue", "green", "red", "nir", "coastal", "rededge1",
    "rededge2", "rededge3", "nir08", "nir09", "swir16", "swir22",
    "scl", "aot", "wvp",
)
META_ASSETS = ("granule_metadata", "tileinfo_metadata", "thumbnail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stac", type=Path, default=Path("/tmp/earthsearch.json"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("carla_mine/external_imagery/sentinel2_20260628/crop_1km"))
    args = ap.parse_args()
    catalog = json.loads(args.stac.read_text())
    item = catalog["features"][0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "item.json").write_text(json.dumps(item, indent=2))

    for name in RASTER_ASSETS:
        asset = item["assets"][name]
        url = asset["href"]
        print(f"[{name}] {url}", flush=True)
        with rasterio.open(url) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *BBOX, densify_pts=21)
            win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            data = src.read(window=win)
            profile = src.profile.copy()
            profile.update(width=int(win.width), height=int(win.height),
                           transform=src.window_transform(win), driver="GTiff",
                           compress="deflate", tiled=True)
            with rasterio.open(args.output_dir / f"{name}.tif", "w", **profile) as dst:
                dst.write(data)

    for name in META_ASSETS:
        url = item["assets"][name]["href"]
        suffix = Path(url).suffix or ".bin"
        print(f"[{name}] {url}", flush=True)
        urllib.request.urlretrieve(url, args.output_dir / f"{name}{suffix}")

    summary = {
        "item": item["id"], "datetime": item["properties"]["datetime"],
        "cloud_cover_scene_percent": item["properties"]["eo:cloud_cover"],
        "bbox_wgs84": BBOX, "raster_assets": RASTER_ASSETS,
        "note": "Cropped to the seven-run GPS envelope plus approximately 1 km.",
    }
    (args.output_dir / "download_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
