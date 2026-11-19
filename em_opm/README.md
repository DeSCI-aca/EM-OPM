# EM-OPM: IL and action-conditioned risk reproduction

This directory is the executable implementation corresponding to
`method_updated.tex` and `experiments_updated.tex`. It reproduces the reported
non-RL pipeline:

1. LiDAR BEV construction and trajectory-level imitation learning;
2. action-conditioned transportation-boundary prediction;
3. camera--LiDAR vehicle extraction and tracking;
4. action-conditioned collision prediction;
5. boundary-recovery and dynamic-replay closed-loop evaluation.

The code does **not** claim CARLA-quality rendering, reinforcement learning,
pedestrian interaction, or physical vehicle deployment.

## Data split

Complete traversals are separated before any learning or replay augmentation:

| Split | Traversals |
|---|---|
| Train | M0S0, M0S1, M0S2, M0S3, M0S5 |
| Validation | M0S6 |
| Test | M0S7 |

The default source root is `/media/ivan/Elements SE/06-28`. Change
`DATA_ROOT` in `configs/paper.env`, or create another configuration and set
`EM_OPM_CONFIG`.

## Environment

The recorded environment uses Python 3.11, CUDA-enabled PyTorch 2.5.1,
NumPy 2.4.6, SciPy 1.15.3, OpenCV 5.0.0, PyYAML 6.0.3, and Ultralytics
8.4.210. The default interpreter is
`dust_removal/.conda_night/bin/python`. Package versions are listed in
`requirements.txt` for reference.

## One entry point

Run all commands from the repository root:

```bash
bash em_opm/reproduce.sh check
bash em_opm/reproduce.sh help
```

The complete reproduction is:

```bash
bash em_opm/reproduce.sh all
```

Because actor extraction and training are expensive, stages can be run
individually and resumed from their saved artifacts:

```bash
bash em_opm/reproduce.sh maps
bash em_opm/reproduce.sh combined-dem
bash em_opm/reproduce.sh camera-dem
bash em_opm/reproduce.sh terrain-ui
bash em_opm/reproduce.sh il-data
bash em_opm/reproduce.sh il
bash em_opm/reproduce.sh boundary-data
bash em_opm/reproduce.sh boundary
bash em_opm/reproduce.sh actors
bash em_opm/reproduce.sh collision-data
bash em_opm/reproduce.sh collision
bash em_opm/reproduce.sh tune-collision
bash em_opm/reproduce.sh closed-loop
bash em_opm/reproduce.sh dynamic-loop
bash em_opm/reproduce.sh figures
```

`DEVICE=0` in the configuration selects the physical GPU exposed to the
process. The scripts themselves use logical CUDA device 0, so they work with
one GPU and do not assume a second GPU exists.

## Stage dependencies and outputs

| Stage | Main output |
|---|---|
| `maps` | `carla_mine/outputs/scene_*_closed_loop/` |
| `combined-dem` | `carla_mine/outputs/m0_combined_dem/` |
| `camera-dem` | RGB DEM from the M0S0 cameras, colored OBJ, and coverage report |
| `terrain-ui` | Local confidence-aware terrain editor and CARLA asset export |
| `il-data` | `em_opm/data/il_v1/` (BEVs rendered from reconstructed maps) |
| `il` | `em_opm/runs/il_v1/` |
| `boundary-data` | `em_opm/data/risk_v1/` |
| `boundary` | `em_opm/runs/risk_v1/` |
| `actors` | `em_opm/data/dynamic_actors_v1/` |
| `collision-data` | `em_opm/data/collision_v2/` |
| `collision` | `em_opm/runs/collision_v2*/` |
| `tune-collision` | `em_opm/runs/collision_threshold_selection.json` |
| `closed-loop` | `em_opm/runs/closed_loop_{il,risk,route_upper}/` |
| `dynamic-loop` | `em_opm/runs/dynamic_closed_loop_*/` and `paper_results.json` |
| `figures` | `em_opm/figures/*.pdf` and `*.png` |

Training and validation determine checkpoints and the collision threshold.
M0S7 is read only by final evaluation stages.

## Paper result provenance

The final tables are backed by the following machine-readable files:

- open-loop IL: `runs/il_v1/test_metrics.json`;
- boundary prediction: `runs/risk_v1/test_metrics.json`;
- collision prediction: `runs/collision_v2/multiseed_metrics.json`;
- static closed loop: `runs/closed_loop_*/metrics.json`;
- dynamic replay: `runs/dynamic_closed_loop_*/metrics.json`.

After all stages, `runs/paper_results.json` collects these results without
recomputing them.

## Repository layout

The active directory contains only the current reproduction chain. Previous
smoke tests, superseded datasets, parameter sweeps, and wrapper scripts are
stored under `_legacy/` so the cleanup is reversible. They are not imported or
read by `reproduce.sh`.
