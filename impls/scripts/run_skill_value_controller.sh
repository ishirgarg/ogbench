#!/usr/bin/env bash
# Train the goal-conditioned skill controller pi_hi(z | s, g) ~= argmax_z V(s, z, g)
# on top of a frozen empowerment_skill checkpoint.
#
# The pretrained agent is loaded read-only; the only thing trained is the controller.
# Checkpoints go to <SKILL_CKPT>/controller/OGBench/<run_group>/sd000_<timestamp>/,
# so the pretrained params_*.pkl in SKILL_CKPT itself is never touched.
#
# Two jobs are launched: identical training (same seed -> same controller up to GPU
# nondeterminism, since skill_horizon enters nothing but sample_actions_with_state),
# differing only in the eval-time skill commitment horizon. h=1 reselects the skill
# every env step (the literal pi_hi(z | s, g)); h=10 holds it for 10 steps, matching
# how the online greedy selector in eval_skill_value_policy.py is scored, so the
# amortised controller is directly comparable to the selector it distills. The extra
# training is duplicated on purpose -- it keeps main.py as the only entry point.
# -e so a mistyped SKILL_CKPT fails here rather than launching two jobs with an
# empty --env_name / --skill_restore_epoch.
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=egl

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

SKILL_CKPT=${SKILL_CKPT:-/nas/ucb/ishirgarg/ogbench/impls/ckpts/empowerment/antmaze-medium-stitch/noisy_policy/noisy_q/new_runs_loginterval8k/sd000_s_37866313.0.20260821_030454_k50_s0.01_bc0.001}
ENV_NAME=${ENV_NAME:-$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")}
# Resolve the pretrained epoch here and pass it explicitly. main.py writes flags.json
# before constructing the agent, so a None epoch would be recorded as null and the run
# would keep no record of which pretrained epoch it distilled.
SKILL_EPOCH=${SKILL_EPOCH:-$($PYTHON -c "
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl\$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
" "$SKILL_CKPT")}
SAVE_DIR="$SKILL_CKPT/controller"

if [[ -z "$ENV_NAME" || -z "$SKILL_EPOCH" ]]; then
    echo "could not read env_name / latest epoch from $SKILL_CKPT" >&2
    exit 1
fi

LOG_DIR=logs/skill_value_controller
mkdir -p "$LOG_DIR" "$SAVE_DIR"

GPUS=(0 1)
HORIZONS=(1 10)

echo "ckpt=$SKILL_CKPT"
echo "epoch=$SKILL_EPOCH"
echo "env=$ENV_NAME"
echo "save_dir=$SAVE_DIR"

pids=()
for i in "${!HORIZONS[@]}"; do
    H=${HORIZONS[$i]}
    GPU=${GPUS[$i]}
    LOG="$LOG_DIR/$(basename "$SKILL_CKPT")_h${H}.log"
    echo "launching h=${H} on GPU ${GPU} -> ${LOG}"
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main.py \
        --env_name="$ENV_NAME" \
        --save_dir="$SAVE_DIR" \
        --agent=agents/skill_value_controller.py \
        --agent.skill_checkpoint_path="$SKILL_CKPT" \
        --agent.skill_restore_epoch="$SKILL_EPOCH" \
        --agent.skill_horizon="$H" \
        --train_steps=1000000 \
        --log_interval=5000 \
        --eval_interval=100000 \
        --save_interval=100000 \
        --eval_episodes=50 \
        --video_episodes=0 \
        > "$LOG" 2>&1 &
    pids+=($!)
    # Stagger: exp_name is sd000_<YYYYmmdd_HHMMSS>, so two same-second launches would
    # share one run dir and clobber each other. (Also avoids racing mujoco-EGL inits.)
    sleep 5
done

echo "waiting on ${#pids[@]} jobs..."
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
echo "done. fail=$fail"
