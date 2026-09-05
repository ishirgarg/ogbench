#!/usr/bin/env bash
# DDS high-level skill controller (offline) on the 5 final DDS checkpoints, run
# LOCALLY on this box with at most 2 GPUs.
#
# This is the DDS paper's "relabel with the encoder, then IQL + AWR over the
# codebook" stage (arXiv:2503.20176 Sec. 4.2-4.4): each frozen dds checkpoint is
# loaded read-only (encoder / codebook / diffusion decoder only) and only the
# gciql high level over the K=50 codes is trained, with the paper's Table 7
# hyperparameters (agents/dds_controller.py defaults). Same flags as
# scripts/run_dds_controller.sh -- this script only changes *which* checkpoints
# and *how they are scheduled*.
#
# The 5 checkpoints (ckpts/final/dds/<env>/sd000_*), all K=50, H=10, 1M steps:
#   0  antmaze-medium-navigate-v0
#   1  antmaze-medium-stitch-v0
#   2  antsoccer-arena-navigate-v0
#   3  pointmaze-teleport-navigate-v0
#   4  pointmaze-teleport-stitch-v0
#
# Scheduling: 5 runs dealt round-robin over GPUS (default "0 1" -- 2 GPUs max),
# with at most PER_GPU concurrent runs on a card. With the defaults that is
# GPU 0 <- runs 0,2,4 and GPU 1 <- runs 1,3, all live at once.
# XLA_PYTHON_CLIENT_MEM_FRACTION is derived from PER_GPU so the stacked
# processes fit on one A6000 (48 GB); override it if you change PER_GPU.
#
# Checkpoints go to <SKILL_CKPT>/controller/OGBench/Debug/sd000_<ts>/, so the
# pretrained params_*.pkl in each SKILL_CKPT is never touched.
#
# Run from impls/:  bash scripts/run_dds_controller_final5_local.sh
#   GPUS="4 5" PER_GPU=2 RUN_IDS="0 1"   to override the GPU set / cap / subset.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> impls/

PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
DDS_ROOT=${DDS_ROOT:-ckpts/final/dds}

read -r -a GPUS <<< "${GPUS:-0 1}"
PER_GPU=${PER_GPU:-3}
NGPU=${#GPUS[@]}

# ── The 5 runs (one per final DDS checkpoint) ────────────────────────────────
# Resolved by glob so a re-rsync with a different job id still works; each env
# dir must hold exactly one sd000_* run.
ENV_DIRS=(
    antmaze-medium-navigate
    antmaze-medium-stitch
    antsoccer-arena-navigate
    pointmaze-teleport-navigate
    pointmaze-teleport-stitch
)
SKILL_CKPTS=()
for d in "${ENV_DIRS[@]}"; do
    matches=("$DDS_ROOT/$d"/sd000_*/)
    if (( ${#matches[@]} != 1 )) || [[ ! -d "${matches[0]}" ]]; then
        echo "ERROR: expected exactly one sd000_* run under $DDS_ROOT/$d, found ${#matches[@]}" >&2
        exit 1
    fi
    SKILL_CKPTS+=("${matches[0]%/}")
done

SEED=${SEED:-0}
# The paper trains Q-learning for 1M steps and AWR for 500k more; gciql trains all
# three heads jointly, so one 1M-step run covers the Q-learning budget.
TRAIN_STEPS=${TRAIN_STEPS:-1000000}
read -r -a RUN_IDS <<< "${RUN_IDS:-0 1 2 3 4}"

# ── Environment ─────────────────────────────────────────────────────────────
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=${MUJOCO_GL:-egl}   # video_episodes=0, so no rendering is needed
# PER_GPU JAX processes share a card: cap each one's preallocation.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-$(awk -v n="$PER_GPU" 'BEGIN{printf "%.2f", 0.9/n}')}
# Root fs (/tmp) runs at 100% on this box; keep ptxas scratch on the NAS.
export TMPDIR=${TMPDIR:-/nas/ucb/ishirgarg/tmp}
export JAX_COMPILATION_CACHE_DIR="$(pwd)/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$TMPDIR"

LOG_DIR=logs/dds_controller_final5
mkdir -p "$LOG_DIR"

# ── Preflight: every selected checkpoint must be complete ───────────────────
# The rsync of ckpts/final/dds landed short for at least one env, and a missing
# params_*.pkl / flags.json would otherwise only surface minutes into the run.
for IDX in "${RUN_IDS[@]}"; do
    if (( IDX < 0 || IDX >= ${#SKILL_CKPTS[@]} )); then
        echo "ERROR: run id $IDX out of range 0..$(( ${#SKILL_CKPTS[@]} - 1 ))" >&2; exit 1
    fi
    C=${SKILL_CKPTS[$IDX]}
    [[ -f "$C/flags.json" ]] || { echo "ERROR: missing $C/flags.json" >&2; exit 1; }
    compgen -G "$C/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $C" >&2; exit 1; }
done

# ── Per-run launcher ─────────────────────────────────────────────────────────
launch_run() {
    local IDX=$1 GPU=$2
    local SKILL_CKPT=${SKILL_CKPTS[$IDX]}

    local ENV_NAME SKILL_EPOCH
    ENV_NAME=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    SKILL_EPOCH=$($PYTHON -c "
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
" "$SKILL_CKPT")
    if [[ -z "$ENV_NAME" || -z "$SKILL_EPOCH" ]]; then
        echo "could not read env_name / latest epoch from $SKILL_CKPT" >&2; return 1
    fi

    local SAVE_DIR="$SKILL_CKPT/controller"
    mkdir -p "$SAVE_DIR"
    local LOG="$LOG_DIR/idx${IDX}_${ENV_NAME}_s${SEED}.log"

    echo "[gpu $GPU] IDX=$IDX  ENV=$ENV_NAME  SEED=$SEED"
    echo "          ckpt=$SKILL_CKPT  epoch=$SKILL_EPOCH"
    echo "          save_dir=$SAVE_DIR  log=$LOG"

    # agent.chunk_horizon defaults to 10 and dds_controller.create asserts it equals
    # the checkpoint's sequence_length (10 for all five), so it is left at the default.
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main.py \
        --env_name="$ENV_NAME" \
        --save_dir="$SAVE_DIR" \
        --agent=agents/dds_controller.py:gciql \
        --agent.skill_checkpoint_path="$SKILL_CKPT" \
        --agent.skill_restore_epoch="$SKILL_EPOCH" \
        --seed="$SEED" \
        --train_steps="$TRAIN_STEPS" \
        --log_interval=5000 \
        --eval_interval=100000 \
        --save_interval=100000 \
        --eval_episodes=50 \
        --video_episodes=0 \
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
        # exp_name has 1-second resolution: stagger so no two runs share a dir name.
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
    sleep 2   # keep lanes' first launches on different seconds
done

echo "waiting on ${#lane_pids[@]} GPU lanes (${#RUN_IDS[@]} runs)..."
fail=0
for p in "${lane_pids[@]}"; do wait "$p" || fail=1; done
echo "ALL DONE (fail=$fail)"
exit $fail
