#!/usr/bin/env bash
# Fill in missing empowerment maps (all empowerment-family algos) and missing
# skill paths (empowerment_skill / ckpts/empowerment only, incl. noisy_policy
# sweep subdirs) across ckpts/empowerment{,_dads,_crl,_crl_flowbc,_dv}.
#
# Recursively finds every run dir (any params_*.pkl, any depth) under the
# CKPT_ROOTS below, decides what's missing per run via existing output files,
# and skips runs that already have everything. Only agent_name=='empowerment_skill'
# runs (those with a num_skills config) get skill-path rollouts -- other
# empowerment variants (crl / crl_flowbc / dv / dads) are map-only here since
# crl/crl_flowbc/dv aren't skill-conditioned (dads already has its maps).
#
# Fans jobs out across GPUS (default 0..7), one checkpoint per GPU at a time.
set -uo pipefail

cd "$(dirname "$0")/.."   # -> impls/

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NGPU=${#GPUS[@]}
GRID_RES=${GRID_RES:-200}
STEPS=${STEPS:-3000}
BATCH_SIZE=${BATCH_SIZE:-}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}

export JAX_COMPILATION_CACHE_DIR="$(pwd)/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

ANTMAZE_XY=${ANTMAZE_XY:-8,8}
ANTSOCCER_XY=${ANTSOCCER_XY:-14,14}
ANTSOCCER_BALL_XY=${ANTSOCCER_BALL_XY:-7,7}

CKPT_ROOTS=("ckpts/empowerment" "ckpts/empowerment_dads" "ckpts/empowerment_crl" "ckpts/empowerment_crl_flowbc" "ckpts/empowerment_dv")

mkdir -p logs

# Build the job list in python: for each run dir, decide need_map / need_paths.
JOBS_FILE=$(mktemp)
python3 - "${CKPT_ROOTS[@]}" > "$JOBS_FILE" <<'PYEOF'
import glob, json, os, sys

roots = sys.argv[1:]
for root in roots:
    pkls = glob.glob(os.path.join(root, "**", "params_*.pkl"), recursive=True)
    run_dirs = sorted(set(os.path.dirname(p) for p in pkls))
    for run_dir in run_dirs:
        flags_path = os.path.join(run_dir, "flags.json")
        if not os.path.exists(flags_path):
            print(f"SKIP\t{run_dir}\tno flags.json", file=sys.stderr)
            continue
        with open(flags_path) as f:
            flags = json.load(f)
        agent_cfg = flags.get("agent", {})
        agent_name = agent_cfg.get("agent_name")

        has_map = bool(glob.glob(os.path.join(run_dir, "empowerment*.png")))
        has_paths = bool(glob.glob(os.path.join(run_dir, "skill_*paths*.png")))

        need_map = not has_map
        need_paths = (agent_name == "empowerment_skill") and not has_paths

        if not need_map and not need_paths:
            continue

        if "antsoccer" in run_dir:
            family = "antsoccer"
        elif "antmaze" in run_dir:
            family = "antmaze"
        else:
            print(f"SKIP\t{run_dir}\tunknown env family", file=sys.stderr)
            continue

        print(f"{run_dir}\t{family}\t{int(need_map)}\t{int(need_paths)}")
PYEOF

echo "=== Job list ===" >&2
cat "$JOBS_FILE" >&2

mapfile -t JOBS < "$JOBS_FILE"
rm -f "$JOBS_FILE"

echo "Dispatching ${#JOBS[@]} empowerment fill-gap jobs across GPUs: ${GPUS[*]}"

fail=0
i=0
declare -a LOGS
for job in "${JOBS[@]}"; do
  IFS=$'\t' read -r run_dir family need_map need_paths <<< "$job"
  [[ "$family" == "antsoccer" ]] && script="plot_empowerment_map_antsoccer.py" || script="plot_empowerment_map_antmaze.py"

  gpu=${GPUS[$((i % NGPU))]}
  log="logs/emp_fillgap_$(echo "$run_dir" | tr '/' '_').log"
  LOGS+=("$log")

  args=(--run_dir "$run_dir" --grid_res "$GRID_RES" --no-skill_video)
  [ -n "$BATCH_SIZE" ] && args+=(--batch_size "$BATCH_SIZE")

  if [[ "$need_map" == "1" ]]; then args+=(--skill_map); else args+=(--no-skill_map); fi
  if [[ "$need_paths" == "1" ]]; then
    args+=(--skill_paths --video_steps "$STEPS")
    if [[ "$family" == "antsoccer" ]]; then
      args+=(--video_ant_xy "$ANTSOCCER_XY" --video_ball_xy "$ANTSOCCER_BALL_XY")
    else
      args+=(--video_ant_xy "$ANTMAZE_XY")
    fi
  else
    args+=(--no-skill_paths)
  fi

  echo "[gpu $gpu] $script  $run_dir  map=$need_map paths=$need_paths -> $log"
  CUDA_VISIBLE_DEVICES=$gpu python "$script" "${args[@]}" > "$log" 2>&1 &
  i=$((i + 1))
  (( i % NGPU == 0 )) && wait
done
wait

for log in "${LOGS[@]:-}"; do
  [ -z "${log:-}" ] && continue
  grep -qE "^Saved image:|^Saved skill ant-path plot:" "$log" || { echo "FAILED (or incomplete): $log"; fail=1; }
done

echo "ALL DONE (empowerment fill-gaps, fail=$fail)."
exit $fail
