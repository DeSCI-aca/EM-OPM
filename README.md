# Mine Navigation: IL and Risk-Screened CARLA Evaluation

This repository is the curated release of the current V2 mine-navigation
pipeline. It contains the paper implementation, terrain-generation scripts,
CARLA replay and contact validation tools, final V2 UE map resources, and the
small reports used by the paper.

## Current map

The retained Unreal map is:

`/Game/Carla/Maps/MineFirst20RLFull/MineFirst20RLFullRefinedV2`

Copy `assets/ue_map/MineFirst20RLFull` into the corresponding CARLA UE4
content directory. The source terrain products and validation reports are in
`artifacts/v2`.

## Contents

- `carla_mine/`: DEM, texture, ROI, LiDAR refinement, smoothing, and UE import.
- `carla_runtime/rl/`: real-odometry replay, vehicle-contact checks, and RL environment.
- `em_opm/`: imitation learning, boundary/collision risk models, evaluation, and paper sections.
- `assets/ue_map/`: final V2 Unreal map and OpenDRIVE resources.
- `artifacts/v2/`: final terrain mesh, texture, arrays, navigation task, and validation reports.

## Data policy

The seven raw traversals, the full CARLA/Unreal source trees, virtual
environments, logs, caches, and intermediate maps are intentionally excluded.
They are too large or cannot be redistributed as ordinary Git content. The
scripts expect users to configure the raw-data paths locally.

Large final map resources are tracked with Git LFS. Install Git LFS before
cloning or pushing this repository.

See `carla_mine/CURRENT_PIPELINE.md` for the retained V2 build and validation
sequence.
