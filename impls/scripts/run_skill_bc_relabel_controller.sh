#!/usr/bin/env bash
# Train the BC-relabelled goal-conditioned skill controller (skill_bc_relabel_controller)
# on top of frozen empowerment_skill checkpoints.
#
# The pretrained agent is loaded read-only; only the controller is trained. Checkpoints
# go to <SKILL_CKPT>/controller/OGBench/Debug/sd000_<timestamp>/, so the pretrained
# params_*.pkl in each SKILL_CKPT itself is never touched.
#
# At most 2 GPUs are used at once: jobs are dispatched two at a time (one per GPU),
# waiting for each pair to finish before launching the next. -e so a mistyped
# SKILL_CKPT fails here rather than launching a job with an empty --env_name /
# --skill_restore_epoch.
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=egl

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

GPUS=(0 1)
NGPU=${#GPUS[@]}

SKILL_CKPTS=(
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/empowerment_final/antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/empowerment_final/antmaze-medium-stitch/sd000_s_37866313.0.20260821_030454_k50_s0.01_bc0.001
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/empowerment_final/antsoccer-medium-stitch/sd000_s_38006052.0.20260825_035401_k50_s0.01_bc0.001
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/empowerment_final/antsoccer-medium-navigate/sd000_s_38005166.0.20260825_023027_k50_s0.01_bc0.001
)

LOG_DIR=logs/skill_bc_relabel_controller
mkdir -p "$LOG_DIR"

pids=()
logs=()
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
        --agent=agents/skill_bc_relabel_controller.py:gciql \
        --agent.skill_checkpoint_path="$SKILL_CKPT" \
        --agent.skill_restore_epoch="$SKILL_EPOCH" \
        --train_steps=1000000 \
        --log_interval=5000 \
        --eval_interval=100000 \
        --save_interval=100000 \
        --eval_episodes=50 \
        --video_episodes=0 \
        > "$LOG" 2>&1 &
    pids+=($!)
    logs+=("$LOG")
    i=$((i + 1))
    # Stagger: exp_name is sd000_<YYYYmmdd_HHMMSS>, so two same-second launches on
    # different ckpts (different save_dir, so no clobber risk) would still race
    # mujoco-EGL init. Also throttles to "at most 2 GPUs at once": wait for the
    # pair before starting the next two.
    sleep 5
    if (( i % NGPU == 0 )); then
        echo "waiting on batch of ${NGPU} jobs..."
        fail=0
        for pid in "${pids[@]}"; do
            wait "$pid" || fail=1
        done
        pids=()
    fi
done

# Wait on any trailing partial batch (odd number of checkpoints).
if [ "${#pids[@]}" -gt 0 ]; then
    echo "waiting on trailing batch of ${#pids[@]} jobs..."
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
fi

echo "done. fail=${fail:-0}"
for log in "${logs[@]}"; do
    echo "--- tail $log ---"
    tail -n 5 "$log" || true
done
