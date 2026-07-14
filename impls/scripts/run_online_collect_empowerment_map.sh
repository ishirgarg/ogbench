#!/usr/bin/env bash
# 2D empowerment map only (no skill rollouts) for the 3 online-collect runs.
# All 3 are antsoccer-arena-navigate-v0 / empowerment_skill, so use the
# antsoccer plotting script. Output: empowerment_map_e{epoch}.png per run.
set -uo pipefail
cd "$(dirname "$0")/.."   # -> impls/

GPU=${GPU:-1}
# export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}

# All 3 runs share identical env/agent/config, so the (pathologically slow ~10min
# ptxas) empowerment-map kernel compiles once and the other two reuse it from
# JAX's on-disk compilation cache instead of recompiling.
export JAX_COMPILATION_CACHE_DIR="$(pwd)/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

fail=0
for run_dir in ckpts/online-collect/*/; do
  [ -d "$run_dir" ] || continue
  ls "$run_dir"params_*.pkl >/dev/null 2>&1 || { echo "SKIP (no params): $run_dir"; continue; }
  echo "=================================================================="
  echo "RUN_DIR=$run_dir"
  echo "=================================================================="
  CUDA_VISIBLE_DEVICES=$GPU python plot_empowerment_map_antsoccer.py \
    --run_dir "$run_dir" \
    --no-skill_video --no-skill_paths \
    || { echo "FAILED: $run_dir"; fail=1; }
done
echo "ALL DONE (fail=$fail)."
exit $fail
