#!/usr/bin/env bash
# Per-skill ant-path plots for the empowerment_skill policies under
# ckpts/empowerment. Produces skill_ant_paths_e{epoch}.png per run.
#
# Uses plot_empowerment_map_antmaze.py for antmaze runs and
# plot_empowerment_map_antsoccer.py for antsoccer runs. Runs paths-only
# (--no-skill_video --no-skill_map) so no rendering / slow map compute is done.
# Each path gets color-coded dots at the 500/1000/1500/2000/2500/3000 step
# intervals; rollouts run for STEPS (default 3000) env steps.
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

# Map each env-dir prefix to its plot script.
declare -A SCRIPT_FOR=(
  ["antmaze-medium-navigate"]="plot_empowerment_map_antmaze.py"
  ["antsoccer-arena-navigate"]="plot_empowerment_map_antsoccer.py"
)

RUN_DIRS=()
SCRIPTS=()
for env_dir in "${!SCRIPT_FOR[@]}"; do
  for run_dir in ckpts/empowerment/"$env_dir"/*/; do
    [ -d "$run_dir" ] || continue
    ls "$run_dir"params_*.pkl >/dev/null 2>&1 || { echo "SKIP (no params): $run_dir"; continue; }
    RUN_DIRS+=("$run_dir")
    SCRIPTS+=("${SCRIPT_FOR[$env_dir]}")
  done
done

mkdir -p logs
echo "Dispatching ${#RUN_DIRS[@]} empowerment runs across GPUs: ${GPUS[*]}  (steps=$STEPS)"

i=0
for idx in "${!RUN_DIRS[@]}"; do
  run_dir=${RUN_DIRS[$idx]}
  script=${SCRIPTS[$idx]}
  gpu=${GPUS[$((i % NGPU))]}
  if [[ "$script" == *antsoccer* ]]; then ant_xy="$ANTSOCCER_XY"; else ant_xy="$ANTMAZE_XY"; fi
  log="logs/emp_paths_$(echo "$run_dir" | tr '/' '_').log"
  echo "[gpu $gpu] $script  $run_dir  ant_xy=$ant_xy -> $log"
  CUDA_VISIBLE_DEVICES=$gpu python "$script" \
    --run_dir "$run_dir" --video_steps "$STEPS" --video_ant_xy "$ant_xy" \
    --no-skill_video --no-skill_map --skill_paths > "$log" 2>&1 &
  i=$((i + 1))
  (( i % NGPU == 0 )) && wait
done
wait

echo "ALL DONE (empowerment skill paths)."
