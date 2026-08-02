#!/usr/bin/env bash
# High-priority pass over the *-stitch-slice50 checkpoints under
# ckpts/empowerment: empowerment map AND per-skill ant paths for each run.
#
# Both jobs for a run are launched together (map on one GPU, paths on another)
# so the whole slice50 set finishes as fast as the box allows. Everything runs
# concurrently -- 6 runs x 2 jobs = 12 processes -- so the per-process memory
# fraction is kept low.
set -uo pipefail

cd "$(dirname "$0")/.."   # -> impls/

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NGPU=${#GPUS[@]}
STEPS=${STEPS:-3000}
GRID_RES=${GRID_RES:-200}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.12}

# Root fs (/tmp) runs at 100% on this box; keep ptxas scratch on the NAS.
export TMPDIR=${TMPDIR:-/nas/ucb/ishirgarg/tmp}
export JAX_COMPILATION_CACHE_DIR="$(pwd)/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

ANTMAZE_XY=${ANTMAZE_XY:-8,8}
ANTSOCCER_XY=${ANTSOCCER_XY:-6,6}
# Antsoccer rollouts start the ball at a fixed spot too, so the ant-path and
# ball-path figures are comparable across runs.
ANTSOCCER_BALL_XY=${ANTSOCCER_BALL_XY:-3,3}

mkdir -p logs
i=0
n=0
for run_dir in ckpts/empowerment/*slice50*/*/; do
  [ -d "$run_dir" ] || continue
  ls "$run_dir"params_*.pkl >/dev/null 2>&1 || { echo "SKIP (no params): $run_dir"; continue; }
  ball_arg=""
  case "$run_dir" in
    *antsoccer*) script="plot_empowerment_map_antsoccer.py"; ant_xy="$ANTSOCCER_XY"; ball_arg="--video_ball_xy $ANTSOCCER_BALL_XY" ;;
    *antmaze*)   script="plot_empowerment_map_antmaze.py";   ant_xy="$ANTMAZE_XY" ;;
    *) echo "SKIP (no plot script for env): $run_dir"; continue ;;
  esac
  tag=$(echo "$run_dir" | tr '/' '_')

  gpu=${GPUS[$((i % NGPU))]}; i=$((i + 1))
  echo "[gpu $gpu] MAP   $script  $run_dir"
  CUDA_VISIBLE_DEVICES=$gpu python "$script" \
    --run_dir "$run_dir" --grid_res "$GRID_RES" \
    --no-skill_video --no-skill_paths > "logs/emp_map_$tag.log" 2>&1 &

  gpu=${GPUS[$((i % NGPU))]}; i=$((i + 1))
  echo "[gpu $gpu] PATHS $script  $run_dir  ant_xy=$ant_xy"
  CUDA_VISIBLE_DEVICES=$gpu python "$script" \
    --run_dir "$run_dir" --video_steps "$STEPS" --video_ant_xy "$ant_xy" $ball_arg \
    --no-skill_video --no-skill_map --skill_paths > "logs/emp_paths_$tag.log" 2>&1 &

  n=$((n + 1))
done

echo "Launched $n slice50 runs (map + paths each) across GPUs: ${GPUS[*]}"
wait
echo "ALL DONE (slice50 priority)."
