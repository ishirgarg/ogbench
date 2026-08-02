#!/usr/bin/env bash
# Per-skill ant-path plots for the DDS-trained policies under ckpts/dds.
#
# Uses plot_dds_skill_paths.py (DDS-specific: rolls out each VQ codebook skill
# via the low-level decoder). Produces dds_skill_paths_e{epoch}.png per run.
# Each path gets color-coded dots at the 500/1000/1500/2000/2500/3000 step
# intervals; rollouts run for STEPS (default 3000) env steps.
#
# Covers antmaze + antsoccer (8 checkpoints). pointmaze-teleport is skipped:
# no skill-rollout support exists for that env.
#
# Fans jobs out across GPUS (default 0..7), one checkpoint per GPU at a time.
set -uo pipefail

cd "$(dirname "$0")/.."   # -> impls/

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NGPU=${#GPUS[@]}
STEPS=${STEPS:-3000}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}

# Central ant start position per env family (world x,y). Overridable via env vars.
# antmaze-medium: (8,8) is the most central *free* maze cell (true center borders a wall).
# antsoccer-arena: (10,10) is the true center of the open arena.
ANTMAZE_XY=${ANTMAZE_XY:-8,8}
ANTSOCCER_XY=${ANTSOCCER_XY:-10,10}

ENV_DIRS=(
  "antmaze-medium-navigate"
  "antmaze-medium-stitch"
  "antsoccer-arena-navigate"
  "antsoccer-arena-stitch"
)

RUN_DIRS=()
for env_dir in "${ENV_DIRS[@]}"; do
  for run_dir in ckpts/dds/"$env_dir"/*/; do
    [ -d "$run_dir" ] || continue
    ls "$run_dir"params_*.pkl >/dev/null 2>&1 || { echo "SKIP (no params): $run_dir"; continue; }
    RUN_DIRS+=("$run_dir")
  done
done

mkdir -p logs
echo "Dispatching ${#RUN_DIRS[@]} DDS runs across GPUs: ${GPUS[*]}  (steps=$STEPS)"

i=0
for run_dir in "${RUN_DIRS[@]}"; do
  gpu=${GPUS[$((i % NGPU))]}
  if [[ "$run_dir" == *antsoccer* ]]; then ant_xy="$ANTSOCCER_XY"; else ant_xy="$ANTMAZE_XY"; fi
  log="logs/dds_paths_$(echo "$run_dir" | tr '/' '_').log"
  echo "[gpu $gpu] $run_dir  ant_xy=$ant_xy -> $log"
  CUDA_VISIBLE_DEVICES=$gpu python plot_dds_skill_paths.py \
    --run_dir "$run_dir" --steps "$STEPS" --ant_xy "$ant_xy" > "$log" 2>&1 &
  i=$((i + 1))
  (( i % NGPU == 0 )) && wait
done
wait

echo "ALL DONE (dds skill paths)."
