#!/usr/bin/env bash
# Regenerate the 2D empowerment map for every checkpoint under
# ckpts/empowerment and ckpts/empowerment_crl.
#
# Picks the plotting script by env family (antmaze vs antsoccer) and runs
# map-only (--no-skill_video --no-skill_paths) so no rendering or rollouts
# happen here -- skill paths are handled by run_empowerment_skill_rollouts.sh.
#
# Output per run dir: empowerment_antmaze_e{epoch}.png/.npy (antmaze) or
# empowerment_e{epoch}.png/.npy (antsoccer).
#
# Fans jobs out across GPUS (default 0..7), one checkpoint per GPU at a time.
set -uo pipefail

cd "$(dirname "$0")/.."   # -> impls/

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NGPU=${#GPUS[@]}
GRID_RES=${GRID_RES:-200}
# Empty -> let each plot script use its own default (16 antmaze / 128 antsoccer).
BATCH_SIZE=${BATCH_SIZE:-}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}

# Runs of the same env/agent share the (pathologically slow ~10min ptxas)
# empowerment kernel, so compile once and let the rest hit the on-disk cache.
export JAX_COMPILATION_CACHE_DIR="$(pwd)/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

# Override with e.g. CKPT_ROOTS="ckpts/empowerment_final" to target one tree.
read -r -a CKPT_ROOTS <<< "${CKPT_ROOTS:-ckpts/empowerment ckpts/empowerment_crl}"

RUN_DIRS=()
SCRIPTS=()
for root in "${CKPT_ROOTS[@]}"; do
  for run_dir in "$root"/*/*/; do
    [ -d "$run_dir" ] || continue
    ls "$run_dir"params_*.pkl >/dev/null 2>&1 || { echo "SKIP (no params): $run_dir"; continue; }
    case "$run_dir" in
      *antsoccer*) script="plot_empowerment_map_antsoccer.py" ;;
      *antmaze*)   script="plot_empowerment_map_antmaze.py" ;;
      *) echo "SKIP (no plot script for env): $run_dir"; continue ;;
    esac
    RUN_DIRS+=("$run_dir")
    SCRIPTS+=("$script")
  done
done

mkdir -p logs
echo "Dispatching ${#RUN_DIRS[@]} empowerment-map jobs across GPUs: ${GPUS[*]}  (grid_res=$GRID_RES)"

fail=0
i=0
for idx in "${!RUN_DIRS[@]}"; do
  run_dir=${RUN_DIRS[$idx]}
  script=${SCRIPTS[$idx]}
  gpu=${GPUS[$((i % NGPU))]}
  log="logs/emp_map_$(echo "$run_dir" | tr '/' '_').log"
  echo "[gpu $gpu] $script  $run_dir -> $log"
  bs_arg=()
  [ -n "$BATCH_SIZE" ] && bs_arg=(--batch_size "$BATCH_SIZE")
  CUDA_VISIBLE_DEVICES=$gpu python "$script" \
    --run_dir "$run_dir" --grid_res "$GRID_RES" "${bs_arg[@]}" \
    --no-skill_video --no-skill_paths > "$log" 2>&1 &
  i=$((i + 1))
  (( i % NGPU == 0 )) && wait
done
wait

for idx in "${!RUN_DIRS[@]}"; do
  run_dir=${RUN_DIRS[$idx]}
  log="logs/emp_map_$(echo "$run_dir" | tr '/' '_').log"
  grep -qE "^Saved (3x3 )?image:" "$log" || { echo "FAILED: $run_dir (see $log)"; fail=1; }
done

echo "ALL DONE (empowerment maps, fail=$fail)."
exit $fail
