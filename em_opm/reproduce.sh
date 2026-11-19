#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${EM_OPM_CONFIG:-em_opm/configs/paper.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing configuration: $CONFIG_FILE" >&2
  exit 2
fi
# shellcheck source=configs/paper.env
source "$CONFIG_FILE"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-/media/ivan/Elements SE/06-28}"
DEVICE="${DEVICE:-0}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

require_scene() {
  [[ -d "$DATA_ROOT/scene_$1" ]] || { echo "Missing scene: $DATA_ROOT/scene_$1" >&2; exit 2; }
}

stage="${1:-help}"

case "$stage" in
  maps)
    for sid in $ALL_SCENES; do
      require_scene "$sid"
      "$PYTHON_BIN" carla_mine/build_stage1.py \
        --scene "$DATA_ROOT/scene_$sid" \
        --output "carla_mine/outputs/scene_${sid}_closed_loop" \
        --step-s 4 --max-points 12000 --resolution 1.0 --road-width 12
    done
    ;;

  combined-dem)
    for sid in $ALL_SCENES; do
      require_file "carla_mine/outputs/scene_${sid}_closed_loop/global_points.npy"
      require_file "carla_mine/outputs/scene_${sid}_closed_loop/report.json"
    done
    "$PYTHON_BIN" carla_mine/build_combined_dem.py \
      --maps carla_mine/outputs --output carla_mine/outputs/m0_combined_dem \
      --scenes 0,1,2,3,5,6,7 --reference-scene 0 \
      --resolution 1.0 --voxel-size 0.5
    ;;

  camera-dem)
    require_file carla_mine/outputs/m0_combined_dem/terrain_dem.npz
    require_file carla_mine/outputs/m0_combined_dem/report.json
    for sid in $ALL_SCENES; do require_scene "$sid"; done
    "$PYTHON_BIN" carla_mine/colorize_combined_dem.py \
      --data-root "$DATA_ROOT" --dem-dir carla_mine/outputs/m0_combined_dem \
      --scenes 0 --step-s 60 --max-points 8000
    ;;

  terrain-ui)
    require_file carla_mine/outputs/m0_combined_dem/terrain_dem.npz
    "$PYTHON_BIN" carla_mine/terrain_editor.py
    ;;

  il-data)
    "$PYTHON_BIN" em_opm/build_il_dataset.py \
      --source "$DATA_ROOT" --maps carla_mine/outputs \
      --output em_opm/data/il_v1 --sample-hz 0.5
    ;;

  il)
    require_file em_opm/data/il_v1/manifest.json
    "$PYTHON_BIN" em_opm/train_il.py --data em_opm/data/il_v1 \
      --output em_opm/runs/il_v1 --epochs "$IL_EPOCHS" --batch-size 64 --workers 4
    "$PYTHON_BIN" em_opm/evaluate_il.py --data em_opm/data/il_v1 \
      --checkpoint em_opm/runs/il_v1/best.pt \
      --output em_opm/runs/il_v1/test_metrics.json --batch-size 128
    ;;

  boundary-data)
    for sid in $ALL_SCENES; do
      require_file "carla_mine/outputs/scene_${sid}_closed_loop/global_points.npy"
    done
    "$PYTHON_BIN" em_opm/build_risk_dataset.py \
      --maps carla_mine/outputs --output em_opm/data/risk_v1 --route-states 40
    ;;

  boundary)
    require_file em_opm/data/risk_v1/manifest.json
    "$PYTHON_BIN" em_opm/train_risk.py --data em_opm/data/risk_v1 \
      --output em_opm/runs/risk_v1 --epochs "$RISK_EPOCHS" --batch-size 128
    "$PYTHON_BIN" em_opm/evaluate_risk.py --data em_opm/data/risk_v1 \
      --checkpoint em_opm/runs/risk_v1/best.pt \
      --output em_opm/runs/risk_v1/test_metrics.json
    ;;

  actors)
    require_file "$YOLO_MODEL"
    mkdir -p em_opm/data/dynamic_actors_v1/logs
    for sid in $ALL_SCENES; do
      require_scene "$sid"
      CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" em_opm/extract_dynamic_actors.py \
        "$DATA_ROOT/scene_$sid" --cams 0,1,2,3,4,5 --stride 3 \
        --backend ultralytics --yolo-model "$YOLO_MODEL" --device 0 \
        --output em_opm/data/dynamic_actors_v1 \
        >"em_opm/data/dynamic_actors_v1/logs/scene_${sid}.log" 2>&1
      "$PYTHON_BIN" em_opm/track_dynamic_actors.py \
        "em_opm/data/dynamic_actors_v1/scene_$sid.jsonl" \
        --output "em_opm/data/dynamic_actors_v1/scene_${sid}_tracks.json"
    done
    "$PYTHON_BIN" em_opm/summarize_actor_tracks.py
    ;;

  collision-data)
    require_file em_opm/data/dynamic_actors_v1/summary.json
    "$PYTHON_BIN" em_opm/build_collision_dataset.py \
      --tracks em_opm/data/dynamic_actors_v1 \
      --output em_opm/data/collision_v2 --scenarios-per-track 96 --seed 23
    ;;

  collision)
    require_file em_opm/data/collision_v2/manifest.json
    for seed in $COLLISION_SEEDS; do
      CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" em_opm/train_collision.py \
        --data em_opm/data/collision_v2 \
        --output "em_opm/runs/collision_v2_seed${seed}" \
        --epochs "$COLLISION_EPOCHS" --batch-size 1024 --seed "$seed"
    done
    cp em_opm/runs/collision_v2_seed31/best.pt em_opm/runs/collision_v2/best.pt
    "$PYTHON_BIN" em_opm/summarize_results.py collision
    ;;

  tune-collision)
    require_file em_opm/runs/collision_v2/best.pt
    for threshold in 0.1 0.2 0.3 0.5 0.7; do
      CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" em_opm/run_dynamic_closed_loop.py \
        --mode learned --map carla_mine/outputs/scene_6_closed_loop \
        --tracks em_opm/data/dynamic_actors_v1/scene_6_tracks.json \
        --collision-threshold "$threshold" \
        --output "em_opm/runs/tuning/collision_threshold_${threshold}"
    done
    "$PYTHON_BIN" em_opm/summarize_results.py threshold
    ;;

  closed-loop)
    require_file em_opm/runs/il_v1/best.pt
    require_file em_opm/runs/risk_v1/best.pt
    "$PYTHON_BIN" em_opm/run_closed_loop.py --controller il \
      --output em_opm/runs/closed_loop_il
    "$PYTHON_BIN" em_opm/run_closed_loop.py --controller il \
      --risk-checkpoint em_opm/runs/risk_v1/best.pt --risk-lambda 0.02 \
      --output em_opm/runs/closed_loop_risk
    "$PYTHON_BIN" em_opm/run_closed_loop.py --controller route \
      --output em_opm/runs/closed_loop_route_upper
    ;;

  dynamic-loop)
    require_file em_opm/runs/collision_v2/best.pt
    for mode in none brake geometric; do
      CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" em_opm/run_dynamic_closed_loop.py \
        --mode "$mode" --output "em_opm/runs/dynamic_closed_loop_${mode}"
    done
    CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" em_opm/run_dynamic_closed_loop.py \
      --mode learned --collision-threshold "$COLLISION_THRESHOLD" \
      --output em_opm/runs/dynamic_closed_loop_learned
    "$PYTHON_BIN" em_opm/summarize_results.py paper
    ;;

  figures)
    require_file em_opm/runs/paper_results.json
    MPLCONFIGDIR=/tmp/em_opm_matplotlib "$PYTHON_BIN" em_opm/generate_paper_figures.py
    ;;

  all)
    for s in maps combined-dem camera-dem il-data il boundary-data boundary actors collision-data collision tune-collision closed-loop dynamic-loop figures; do
      "$0" "$s"
    done
    ;;

  check)
    "$PYTHON_BIN" -m compileall -q em_opm
    "$PYTHON_BIN" em_opm/summarize_results.py check
    ;;

  help|*)
    cat <<'EOF'
Usage: bash em_opm/reproduce.sh STAGE

Stages:
  maps             Reconstruct point maps for all seven traversals
  combined-dem     Fuse all seven traversals into one georeferenced DEM
  camera-dem       Color the seven-traversal DEM with the six M0S0 cameras
  terrain-ui       Open the local terrain visualization and editing UI
  il-data          Build the imitation-learning dataset
  il               Train and evaluate the imitation policy
  boundary-data    Build action-conditioned boundary examples
  boundary         Train and evaluate the boundary-risk model
  actors           Extract and track camera--LiDAR dynamic vehicles
  collision-data   Build controlled collision replay pairs
  collision        Train three collision-model seeds and summarize them
  tune-collision   Select the collision threshold on M0S6
  closed-loop      Run IL, boundary-screened, and privileged route tests
  dynamic-loop     Run all four M0S7 dynamic-replay baselines
  figures          Generate publication figures from held-out artifacts
  check            Validate code and required paper artifacts
  all              Run every stage in dependency order

Override defaults with EM_OPM_CONFIG=/path/to/config.env.
EOF
    ;;
esac
