#!/usr/bin/env bash
# Train the OPAL task policy (opal_controller: paper Sec. 4.2 / App. F, IQL instead of
# CQL by default) on top of frozen, finished OPAL runs.
#
# Every start index of the dataset is labelled with one skill drawn from the OPAL
# posterior (p(z|tau) on the discrete path, q(z|tau) on the continuous one), and the
# inner goal-conditioned algorithm is trained on the option MDP those labels define.
# The pretrained agent is loaded read-only; only the task policy is trained.
# Checkpoints go to <OPAL_CKPT>/controller/OGBench/Debug/sd000_<timestamp>/, so the
# pretrained params_*.pkl in each OPAL_CKPT itself is never touched.
#
# At most 2 GPUs are used at once: jobs are dispatched two at a time (one per GPU),
# waiting for each pair to finish before launching the next.
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=egl

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

GPUS=(0 1)
NGPU=${#GPUS[@]}

# Discrete (App. F offline-DADS, k=15) OPAL runs. The continuous (VAE) runs in the same
# folders (the *_c10 dirs) work too: the agent reads latent_type from flags.json.
OPAL_CKPTS=(
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/opal/antmaze-medium-navigate/sd000_s_36595195.0.20260807_234025_k15
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/opal/antmaze-medium-stitch/sd000_s_36595199.0.20260807_234037_k15
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/opal/antsoccer-arena-navigate/sd000_s_36595197.0.20260807_234031_k15
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/opal/antsoccer-arena-stitch/sd000_s_36595180.0.20260807_234044_k15
)

# Inner algorithm (gciql | crl) and any of its knobs, e.g. --agent.base.expectile=0.9.
BASE_AGENT=${BASE_AGENT:-gciql}

LOG_DIR=logs/opal_controller
mkdir -p "$LOG_DIR"

pids=()
logs=()
i=0
for OPAL_CKPT in "${OPAL_CKPTS[@]}"; do
    ENV_NAME=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$OPAL_CKPT")
    CHUNK_SIZE=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['agent']['chunk_size'])" "$OPAL_CKPT")
    OPAL_EPOCH=$($PYTHON -c "
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl\$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
" "$OPAL_CKPT")
    SAVE_DIR="$OPAL_CKPT/controller"

    if [[ -z "$ENV_NAME" || -z "$OPAL_EPOCH" || -z "$CHUNK_SIZE" ]]; then
        echo "could not read env_name / chunk_size / latest epoch from $OPAL_CKPT" >&2
        exit 1
    fi

    mkdir -p "$SAVE_DIR"
    GPU=${GPUS[$((i % NGPU))]}
    LOG="$LOG_DIR/$(basename "$(dirname "$OPAL_CKPT")")_$(basename "$OPAL_CKPT").log"

    echo "ckpt=$OPAL_CKPT"
    echo "  epoch=$OPAL_EPOCH env=$ENV_NAME chunk=$CHUNK_SIZE base=$BASE_AGENT save_dir=$SAVE_DIR gpu=$GPU log=$LOG"

    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main.py \
        --env_name="$ENV_NAME" \
        --save_dir="$SAVE_DIR" \
        --agent=agents/opal_controller.py:"$BASE_AGENT" \
        --agent.skill_checkpoint_path="$OPAL_CKPT" \
        --agent.skill_restore_epoch="$OPAL_EPOCH" \
        --agent.chunk_horizon="$CHUNK_SIZE" \
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
    # Stagger: exp_name is sd000_<YYYYmmdd_HHMMSS>, so two same-second launches would
    # race mujoco-EGL init. Also throttles to "at most 2 GPUs at once".
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
