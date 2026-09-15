#!/usr/bin/env bash
# Resume the 8 unfinished ckpts/final/opal pretraining runs, LOCALLY on this box,
# on 2 GPUs with 2 concurrent jobs per GPU (4 simultaneous, remaining 4 queued
# behind within each GPU's lane).
#
# These runs originally trained on a different machine; their latest checkpoint
# (params_*.pkl + flags.json + wandb_run_id.txt) was copied over here. main.py's
# --resume_dir replays the full agent config + training flags from the run's own
# flags.json and re-attaches to the SAME wandb run via wandb_run_id.txt (verified
# 2026-09-14: wandb resume is purely server-side by run id, independent of the
# local save_dir/machine). So the only flags needed per run are --agent and
# --resume_dir; everything else (env_name, train_steps, log/eval/save intervals,
# eval_episodes, video_episodes=0, ...) comes back from the saved flags.json.
#
# The 8 checkpoints (ckpts/final/opal/<env>/sd000_*), latest epoch as of 2026-09-14:
#   antmaze-medium-navigate-v0     925000 / 1000000
#   antmaze-medium-stitch-v0       950000 / 1000000
#   antsoccer-arena-navigate-v0    875000 / 1000000
#   antsoccer-arena-stitch-v0      875000 / 1000000
#   cube-double-play-v0            950000 / 1000000
#   cube-single-play-v0            400000 / 1000000
#   pointmaze-teleport-navigate-v0 850000 / 1000000
#   pointmaze-teleport-stitch-v0   875000 / 1000000
#
# SIZE THIS AGAINST WHAT IS ACTUALLY FREE -- this box is shared. Default GPUs
# below (2, 4) were picked 2026-09-14 when GPU2 was ~idle and GPU4 was already at
# ~74% util/4.5GB from other jobs; re-check with
#   nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
# opal pretraining is small (well under 5 GB per run observed elsewhere in this
# repo), so PER_GPU=2 at MEM_FRACTION=0.45 (0.9/PER_GPU) leaves ample headroom.
#
# Run from impls/, detached so it survives an ssh disconnect:
#   setsid nohup bash scripts/resume_opal_final_local.sh > logs/resume_opal_final.out 2>&1 &
#   GPUS="0 1" PER_GPU=3 RUN_IDS="0 1"   to override the GPU set / cap / subset.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> impls/

PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
OPAL_ROOT=${OPAL_ROOT:-ckpts/final/opal}

read -r -a GPUS <<< "${GPUS:-2 4}"
PER_GPU=${PER_GPU:-2}
NGPU=${#GPUS[@]}

# ── The 8 runs (one per final opal checkpoint) ───────────────────────────────
# Resolved by glob so a re-copy with a different job id still works; each env
# dir must hold exactly one sd000_* run.
ENV_DIRS=(
    antmaze-medium-navigate
    antmaze-medium-stitch
    antsoccer-arena-navigate
    antsoccer-arena-stitch
    cube-double-play
    cube-single-play
    pointmaze-teleport-navigate
    pointmaze-teleport-stitch
)
RUN_DIRS=()
for d in "${ENV_DIRS[@]}"; do
    matches=("$OPAL_ROOT/$d"/sd000_*/)
    if (( ${#matches[@]} != 1 )) || [[ ! -d "${matches[0]}" ]]; then
        echo "ERROR: expected exactly one sd000_* run under $OPAL_ROOT/$d, found ${#matches[@]}" >&2
        exit 1
    fi
    RUN_DIRS+=("${matches[0]%/}")
done

read -r -a RUN_IDS <<< "${RUN_IDS:-0 1 2 3 4 5 6 7}"

# ── Environment ───────────────────────────────────────────────────────────────
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
# PER_GPU JAX processes share a card: cap each one's preallocation.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-$(awk -v n="$PER_GPU" 'BEGIN{printf "%.2f", 0.9/n}')}
# Root fs (/tmp) runs at 100% on this box; keep ptxas scratch on the NAS.
export TMPDIR=${TMPDIR:-/nas/ucb/ishirgarg/tmp}
export JAX_COMPILATION_CACHE_DIR="$(pwd)/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$TMPDIR"

LOG_DIR=logs/resume_opal_final
mkdir -p "$LOG_DIR"

# ── Preflight: every selected checkpoint must be resumable ───────────────────
for IDX in "${RUN_IDS[@]}"; do
    if (( IDX < 0 || IDX >= ${#RUN_DIRS[@]} )); then
        echo "ERROR: run id $IDX out of range 0..$(( ${#RUN_DIRS[@]} - 1 ))" >&2; exit 1
    fi
    R=${RUN_DIRS[$IDX]}
    [[ -f "$R/flags.json" ]] || { echo "ERROR: missing $R/flags.json" >&2; exit 1; }
    [[ -f "$R/wandb_run_id.txt" ]] || { echo "ERROR: missing $R/wandb_run_id.txt (would start a NEW wandb run)" >&2; exit 1; }
    compgen -G "$R/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $R" >&2; exit 1; }
done

# ── Per-run launcher ──────────────────────────────────────────────────────────
launch_run() {
    local IDX=$1 GPU=$2
    local RUN_DIR=${RUN_DIRS[$IDX]}
    local ENV_NAME
    ENV_NAME=$(basename "$(dirname "$RUN_DIR")")
    local LOG="$LOG_DIR/idx${IDX}_${ENV_NAME}.log"

    echo "[gpu $GPU] IDX=$IDX  ENV=$ENV_NAME"
    echo "          resume_dir=$RUN_DIR  log=$LOG"

    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main.py \
        --agent=agents/opal.py \
        --resume_dir="$RUN_DIR" \
        > "$LOG" 2>&1 &
    LAST_PID=$!
}

# ── Scheduler: round-robin over GPUs, at most PER_GPU live runs per GPU ──────
# One background "lane" per GPU runs its assigned indices, keeping up to PER_GPU
# of them alive at once. Lanes themselves run concurrently.
gpu_lane() {
    local GPU=$1; shift
    local -a queue=("$@")
    local -a live=()
    local fail=0
    for IDX in "${queue[@]}"; do
        while (( ${#live[@]} >= PER_GPU )); do
            local -a still=()
            for p in "${live[@]}"; do
                if kill -0 "$p" 2>/dev/null; then still+=("$p"); else wait "$p" || fail=1; fi
            done
            live=("${still[@]}")
            (( ${#live[@]} >= PER_GPU )) && sleep 30
        done
        launch_run "$IDX" "$GPU"
        live+=("$LAST_PID")
        sleep 5
    done
    for p in "${live[@]}"; do wait "$p" || fail=1; done
    return $fail
}

declare -A ASSIGN
for k in "${!RUN_IDS[@]}"; do
    IDX=${RUN_IDS[$k]}
    GPU=${GPUS[$((k % NGPU))]}
    ASSIGN[$GPU]+="$IDX "
done

lane_pids=()
for GPU in "${GPUS[@]}"; do
    [[ -n "${ASSIGN[$GPU]:-}" ]] || continue
    read -r -a ids <<< "${ASSIGN[$GPU]}"
    echo "GPU $GPU lane: run ids ${ids[*]} (max $PER_GPU concurrent)"
    gpu_lane "$GPU" "${ids[@]}" &
    lane_pids+=($!)
    sleep 2
done

echo "waiting on ${#lane_pids[@]} GPU lanes (${#RUN_IDS[@]} runs)..."
fail=0
for p in "${lane_pids[@]}"; do wait "$p" || fail=1; done
echo "ALL DONE (fail=$fail)"
exit $fail
