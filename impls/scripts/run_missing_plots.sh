#!/usr/bin/env bash
# Generate every plot that scripts/audit_plots.py reports as missing.
#
# The audit decides what applies per run (empowerment map, skill paths, or both)
# from the agent + env in flags.json, so this covers every ckpt root -- including
# ones the earlier per-root drivers never looked at, like ckpts/online_soccer.
#
# Runs whose plot job is already in flight are skipped, so this is safe to run
# alongside or straight after another driver.
set -uo pipefail

cd "$(dirname "$0")/.."   # -> impls/

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NGPU=${#GPUS[@]}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}
# Root fs (/tmp) runs at 100% on this box; keep ptxas scratch on the NAS.
export TMPDIR=${TMPDIR:-/nas/ucb/ishirgarg/tmp}
export JAX_COMPILATION_CACHE_DIR="$(pwd)/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

mkdir -p logs

# FORCE_SOCCER_PATHS=1 also re-runs the antsoccer path jobs whose plots already
# exist -- needed whenever the rollout start positions or the figure style change.
AUDIT_ARGS=(--print-cmds)
[ "${FORCE_SOCCER_PATHS:-0}" = 1 ] && AUDIT_ARGS+=(--force-soccer-paths)

mapfile -t CMDS < <(python scripts/audit_plots.py "${AUDIT_ARGS[@]}")
echo "Audit reports ${#CMDS[@]} plot job(s) to run."

i=0
launched=0
for cmd in "${CMDS[@]}"; do
  [ -n "$cmd" ] || continue
  run_dir=$(echo "$cmd" | sed -E 's/.*--run_dir ([^ ]+).*/\1/')
  case "$cmd" in
    *--skill_paths*|plot_dds_skill_paths*) kind=paths ;;
    *) kind=map ;;
  esac
  # Match on kind as well as run dir: a run's map and paths jobs are separate
  # work and must not mask each other (a map job is "--no-skill_paths", which
  # is why the two patterns cannot collide).
  if [ "$kind" = map ]; then
    inflight_pat="plot_.*--run_dir $run_dir.*--no-skill_paths"
  else
    inflight_pat="plot_.*--run_dir $run_dir.*(--skill_paths|--steps)"
  fi
  if pgrep -f "$inflight_pat" >/dev/null 2>&1; then
    echo "SKIP ($kind already in flight): $run_dir"
    continue
  fi
  tag=$(echo "$run_dir" | tr '/' '_')
  log="logs/missing_${kind}_${tag}.log"
  gpu=${GPUS[$((i % NGPU))]}
  echo "[gpu $gpu] $kind  $cmd -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu python $cmd > "$log" 2>&1 &
  i=$((i + 1)); launched=$((launched + 1))
  (( i % NGPU == 0 )) && wait
done
wait

echo "Launched $launched job(s). Re-auditing:"
python scripts/audit_plots.py | tail -n 3
echo "ALL DONE (missing plots)."
