#!/usr/bin/env bash
# Train the DDS high-level policy (dds_controller) on top of frozen DDS skill models.
#
# This is the paper's "relabel with the encoder, then IQL + AWR over the codebook" stage
# (arXiv:2503.20176 Sec. 4.2-4.4) run as its own job: the pretrained dds checkpoint is
# loaded read-only (only its encoder / codebook / diffusion decoder are used) and only the
# gciql high level over the K codes is trained, with the paper's Table 7 hyperparameters
# (agents/dds_controller.py defaults). Checkpoints go to
# <SKILL_CKPT>/controller/OGBench/Debug/sd000_<timestamp>/, so the pretrained
# params_*.pkl in each SKILL_CKPT itself is never touched.
#
# At most NGPU jobs run at once: jobs are dispatched one per GPU and each batch is waited
# on before the next starts. -e so a mistyped SKILL_CKPT fails here rather than launching
# a job with an empty --env_name / --skill_restore_epoch.
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=egl

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

read -r -a GPUS <<< "${GPUS:-0 1}"
NGPU=${#GPUS[@]}

# One DDS run per env family (K=15, H=10; see ckpts/dds). Override with SKILL_CKPTS="a b c".
DEFAULT_SKILL_CKPTS=(
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/dds/antmaze-medium-navigate/sd000_s_35757533.0.20260721_190340
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/dds/antmaze-medium-stitch/sd000_s_35757537.0.20260721_190341
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/dds/antsoccer-arena-navigate/sd000_s_35757535.0.20260721_190341
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/dds/antsoccer-arena-stitch/sd000_s_35757532.0.20260721_190340
)
if [[ -n "${SKILL_CKPTS:-}" ]]; then
    read -r -a SKILL_CKPTS <<< "$SKILL_CKPTS"
else
    SKILL_CKPTS=("${DEFAULT_SKILL_CKPTS[@]}")
fi

# The paper trains Q-learning for 1M steps and AWR for 500k more; gciql trains all three
# heads jointly, so one 1M-step run covers the Q-learning budget.
TRAIN_STEPS=${TRAIN_STEPS:-1000000}

LOG_DIR=logs/dds_controller
mkdir -p "$LOG_DIR"

pids=()
logs=()
fail=0
i=0
for SKILL_CKPT in "${SKILL_CKPTS[@]}"; do
    ENV_NAME=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    SKILL_EPOCH=$($PYTHON -c "
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl\$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
" "$SKILL_CKPT")
    SAVE_DIR="$SKILL_CKPT/controller"

    if [[ -z "$ENV_NAME" || -z "$SKILL_EPOCH" ]]; then
        echo "could not read env_name / latest epoch from $SKILL_CKPT" >&2
        exit 1
    fi

    mkdir -p "$SAVE_DIR"
    GPU=${GPUS[$((i % NGPU))]}
    LOG="$LOG_DIR/$(basename "$(dirname "$SKILL_CKPT")")_$(basename "$SKILL_CKPT").log"

    echo "ckpt=$SKILL_CKPT"
    echo "  epoch=$SKILL_EPOCH env=$ENV_NAME save_dir=$SAVE_DIR gpu=$GPU log=$LOG"

    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main.py \
        --env_name="$ENV_NAME" \
        --save_dir="$SAVE_DIR" \
        --agent=agents/dds_controller.py:gciql \
        --agent.skill_checkpoint_path="$SKILL_CKPT" \
        --agent.skill_restore_epoch="$SKILL_EPOCH" \
        --train_steps="$TRAIN_STEPS" \
        --log_interval=5000 \
        --eval_interval=100000 \
        --save_interval=100000 \
        --eval_episodes=50 \
        --video_episodes=0 \
        > "$LOG" 2>&1 &
    pids+=($!)
    logs+=("$LOG")
    i=$((i + 1))
    # Stagger: exp_name is sd000_<YYYYmmdd_HHMMSS>, and two same-second launches would
    # also race mujoco-EGL init. Then throttle to NGPU jobs at once.
    sleep 5
    if (( i % NGPU == 0 )); then
        echo "waiting on batch of ${NGPU} jobs..."
        for pid in "${pids[@]}"; do
            wait "$pid" || fail=1
        done
        pids=()
    fi
done

if [ "${#pids[@]}" -gt 0 ]; then
    echo "waiting on trailing batch of ${#pids[@]} jobs..."
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
fi

echo "done. fail=${fail}"
for log in "${logs[@]}"; do
    echo "--- tail $log ---"
    tail -n 5 "$log" || true
done
