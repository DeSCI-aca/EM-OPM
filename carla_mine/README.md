# Data-driven CARLA mine reconstruction

This directory builds the first-stage terrain assets for reproducing EM-OPM.
It does not require CARLA to be installed. The generated OBJ and OpenDRIVE
files are inputs for a later Unreal/CARLA import step.

## Stage 1: one-scene terrain pilot

```bash
cd /path/to/mine_navigation_release
dust_removal/.conda_night/bin/python carla_mine/build_stage1.py \
  --scene "/media/ivan/Elements SE/06-28/scene_0" \
  --output carla_mine/outputs/scene_0_pilot \
  --step-s 5 --max-scans 120 --max-points 12000 --resolution 1.0
```

After checking the pilot, build the denser scene-0 base terrain with:

```bash
dust_removal/.conda_night/bin/python carla_mine/build_stage1.py \
  --scene "/media/ivan/Elements SE/06-28/scene_0" \
  --output carla_mine/outputs/scene_0 \
  --step-s 1 --max-points 24000 --resolution 0.5
```

Outputs:

- `terrain_dem.npz`: height, confidence, observation count and metric axes.
- `terrain_dem.png`: DEM and trajectory preview.
- `terrain.obj`: triangulated observed terrain in local metric coordinates.
- `road_centerline.csv`: smoothed trajectory reference line.
- `road.xodr`: simple one-road OpenDRIVE approximation.
- `report.json`: coordinate origin, parameters, coverage and limitations.

The reconstruction uses fixed-solution GNSS positions and odometry
orientations. Per-column LiDAR timestamps are used for motion compensation.
Unobserved cells remain holes in the OBJ instead of being presented as measured
terrain. The LiDAR coordinates are confirmed to be expressed in the vehicle
base frame, which is also the odometry child frame.

## Camera-colored seven-traversal DEM

Build the common seven-traversal DEM and add surface color from the six
synchronized M0S0 cameras:

```bash
bash em_opm/reproduce.sh combined-dem
bash em_opm/reproduce.sh camera-dem
```

All seven LiDAR traversals determine the geometry. M0S0 is the sole camera
texture source to avoid exposure and white-balance seams across sessions.
Camera pixels are accepted only when
their associated LiDAR return agrees with the DEM height. The camera stage
adds `terrain_rgb.npz`, `terrain_rgb.png`, `terrain_colored.obj`, and
`camera_dem_report.json` under `carla_mine/outputs/m0_combined_dem/`.
The reproducible default uses 60 s keyframes; reduce `--step-s` in
`colorize_combined_dem.py` for a denser offline texture pass.

## Local terrain editor

Start the dependency-free browser UI with:

```bash
bash em_opm/reproduce.sh terrain-ui
```

The editor displays elevation, hillshade, LiDAR confidence, M0S0 camera color,
and edited cells. Its brush can raise, lower, flatten, smooth, erase, or create
terrain. High-confidence LiDAR cells are protected by default. Each export is
saved under `carla_mine/outputs/m0_combined_dem/edits/<timestamp>/` as an NPZ
with source masks, a 16-bit CARLA/Unreal heightmap, metadata, and a colored OBJ.

## Complete CARLA terrain

Fuse the seven-run LiDAR DEM with the downloaded Copernicus GLO-30 terrain:

```bash
MPLCONFIGDIR=/tmp/matplotlib carla_runtime/venv/bin/python \
  carla_mine/build_carla_hybrid_dem.py
```

The result is written to `carla_mine/outputs/carla_hybrid_dem/`. Import
`carla_heightmap_2017.png` as an Unreal Landscape with the XY and Z scales in
`report.json`. The 2017-by-2017 heightmap represents a 4032 m square at 2 m
vertex spacing. `routes_carla.csv` is the orientation and alignment reference;
check it before packaging the map in CARLA.
