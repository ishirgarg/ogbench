#!/usr/bin/env bash
# Resume the empowerment-map / skill-path regeneration for every checkpoint
# whose output is not already fresh.
#
# "Fresh" = the expected output PNG exists AND is newer than MARKER (default
# logs/.regen_marker, stamped at the start of this regeneration pass). That is
# what makes this a *regeneration*: plots left over from earlier passes are
# older than the marker and get redone, while anything produced during this
# pass is left alone. Safe to re-run at any time.
#
# Covers, in one dispatch loop across GPUS:
#   ckpts/empowerment      -> map + skill paths
#   ckpts/empowerment_crl  -> map only (no skill policy in that agent)
#   ckpts/dds              -> skill paths only (plot_dds_skill_paths.py)
set -uo pipefail

cd "$(dirname "$0")/.."   # -> impls/

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NGPU=${#GPUS[@]}
STEPS=${STEPS:-3000}
GRID_RES=${GRID_RES:-200}
MARKER=${MARKER:-logs/.regen_marker}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}
# The antsoccer map autotuner needs ~2GB of scratch per config; its default
# 128-point batch OOMs when jobs share a GPU, so cap it.
MAP_BATCH=${MAP_BATCH:-48}

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

[ -f "$MARKER" ] || { echo "Marker $MARKER missing; refusing to run (it decides what counts as fresh)."; exit 1; }

latest_epoch() {  # run_dir -> highest params_<N>.pkl
  ls "$1"params_*.pkl 2>/dev/null | sed -E 's/.*params_([0-9]+)\.pkl/\1/' | sort -n | tail -1
}
fresh() {  # output path -> 0 if it exists and postdates the marker
  [ -f "$1" ] && [ "$1" -nt "$MARKER" ]
}
in_flight() {  # run_dir -> 0 if any plot job for it is already running
  pgrep -f "plot_.*--run_dir $1" >/dev/null 2>&1
}
# NOTE: this driver queues a run's map and paths together, so the coarse
# per-run-dir check above is correct here. run_missing_plots.sh queues them as
# independent items and needs the finer per-kind check it defines locally.

CMDS=()   # each entry: "<label>\t<logfile>\t<python args...>"
queue() { CMDS+=("$1"); }

for run_dir in ckpts/empowerment/*/*/ ckpts/empowerment_crl/*/*/ ckpts/dds/*/*/; do
  [ -d "$run_dir" ] || continue
  epoch=$(latest_epoch "$run_dir")
  [ -n "$epoch" ] || { echo "SKIP (no params): $run_dir"; continue; }
  # A job launched by hand outside this driver may still be working on it.
  in_flight "$run_dir" && { echo "SKIP (already in flight): $run_dir"; continue; }
  tag=$(echo "$run_dir" | tr '/' '_')

  case "$run_dir" in
    *antsoccer*) script="plot_empowerment_map_antsoccer.py"; ant_xy="$ANTSOCCER_XY"; map_png="empowerment_map_e$epoch.png" ;;
    *antmaze*)   script="plot_empowerment_map_antmaze.py";   ant_xy="$ANTMAZE_XY";   map_png="empowerment_antmaze_e$epoch.png" ;;
    *) echo "SKIP (no plot script for env): $run_dir"; continue ;;
  esac

  case "$run_dir" in
    ckpts/dds/*)
      out="$run_dir/dds_skill_paths_e$epoch.png"
      ball_arg=""; case "$run_dir" in *antsoccer*) ball_arg=" --ball_xy $ANTSOCCER_BALL_XY" ;; esac
      fresh "$out" || queue "PATHS|logs/dds_paths_$tag.log|plot_dds_skill_paths.py --run_dir $run_dir --steps $STEPS --ant_xy $ant_xy$ball_arg"
      ;;
    ckpts/empowerment_crl/*)
      out="$run_dir/$map_png"
      fresh "$out" || queue "MAP|logs/emp_map_$tag.log|$script --run_dir $run_dir --grid_res $GRID_RES --batch_size $MAP_BATCH --no-skill_video --no-skill_paths"
      ;;
    ckpts/empowerment/*)
      out="$run_dir/$map_png"
      fresh "$out" || queue "MAP|logs/emp_map_$tag.log|$script --run_dir $run_dir --grid_res $GRID_RES --batch_size $MAP_BATCH --no-skill_video --no-skill_paths"
      ball_arg=""; case "$run_dir" in *antsoccer*) ball_arg=" --video_ball_xy $ANTSOCCER_BALL_XY" ;; esac
      out="$run_dir/skill_ant_paths_e$epoch.png"
      fresh "$out" || queue "PATHS|logs/emp_paths_$tag.log|$script --run_dir $run_dir --video_steps $STEPS --video_ant_xy $ant_xy$ball_arg --no-skill_video --no-skill_map --skill_paths"
      ;;
  esac
done

echo "Dispatching ${#CMDS[@]} remaining jobs across GPUs: ${GPUS[*]}"
i=0
for entry in "${CMDS[@]}"; do
  kind=${entry%%|*}; rest=${entry#*|}
  log=${rest%%|*}; pyargs=${rest#*|}
  gpu=${GPUS[$((i % NGPU))]}
  echo "[gpu $gpu] $kind  $pyargs -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu python $pyargs > "$log" 2>&1 &
  i=$((i + 1))
  (( i % NGPU == 0 )) && wait
done
wait
echo "ALL DONE (regen resume)."
