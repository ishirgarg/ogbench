#!/bin/bash
# Runs eval_skill_plan.py (selector=plan, the "with replanning" hierarchical policy)
# on every trained run in noisy_policy/noisy_q/new_runs_loginterval8k, in parallel.
set -u
cd "$(dirname "$0")/.."

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2
export MUJOCO_GL=egl

RUN_DIRS=(
  ckpts/empowerment/antmaze-medium-navigate/noisy_policy/noisy_q/new_runs_loginterval8k/*/
  ckpts/empowerment/antmaze-medium-stitch/noisy_policy/noisy_q/new_runs_loginterval8k/*/
)

mkdir -p logs/skill_plan_eval
pids=()
for d in "${RUN_DIRS[@]}"; do
  d="${d%/}"
  if ! ls "$d"/params_*.pkl >/dev/null 2>&1; then
    echo "skip (no checkpoint): $d"
    continue
  fi
  tag=$(basename "$d")
  logf="logs/skill_plan_eval/${tag}.log"
  echo "launching: $d -> $logf"
  python -u eval_skill_plan.py --run_dir "$d" --selector plan --eval_episodes 50 \
    > "$logf" 2>&1 &
  pids+=($!)
done

echo "waiting on ${#pids[@]} jobs..."
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "done. fail=$fail"
