# Current CARLA mine pipeline (V2)

The retained map is:

`/Game/Carla/Maps/MineFirst20RLFull/MineFirst20RLFullRefinedV2`

Final data and reports are under:

`carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined_v2/`

## Retained build sequence

1. `build_m0s0_first20_rl_region.py` selects the M0S0 first-20% spatial ROI and matching data from seven traversals.
2. `build_full_map_with_first20_roi.py` retains the complete satellite terrain/texture and restricts navigation to the ROI.
3. `refine_first20_terrain_lidar_consistency.py` replays 10 Hz OS1-64 scans with odometry deskew.
4. `smooth_first20_contact_hotspots.py` applies the bounded V2 contact correction.
5. `build_refined_roi_patch.py --root carla_mine/outputs/m0s0_first20_rl_full_map_lidar_refined_v2` builds the 1 m ROI mesh.
6. `import_refined_first20_into_ue.py` imports the V2 patch and map.

## Retained validation

- `carla_runtime/rl/replay_real_odom_first20.py`: M0S6/M0S7 real-odom and virtual LiDAR replay.
- `carla_runtime/rl/check_first20_vehicle_contact.py`: Tesla envelope and four-wheel contact checks.

Only poses within 25 m of the M0S0 first-20% core are evaluated. Outside poses are ignored.
