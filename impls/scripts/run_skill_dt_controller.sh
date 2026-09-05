#!/usr/bin/env bash
# Train the Skill-DT high-level controller (skill_dt_controller) on top of frozen,
# finished Skill-DT runs.
#
# Skill-DT (arXiv:2301.13573, agents/skill_dt.py) is reward-free and has no high level
# of its own -- the paper reports the best of a per-skill sweep. This job is the
# hierarchical stage the paper leaves as future work: every start index of the dataset
# is labelled with a codebook skill read off the frozen VQ encoder (LABEL_MODE, default
# the most frequent skill over the H-step window), and the inner goal-conditioned
# algorithm (BASE_AGENT, gciql by default) is trained on the option MDP those labels
# define, exactly as run_dds_controller.sh / run_opal_controller.sh do for their
# skill models. At eval the controller picks a skill every SKILL_HORIZON env steps and
# the frozen Transformer executes it with the paper's Sec. A.5 rollout.
#
# The pretrained agent is loaded read-only; only the controller is trained.
# Checkpoints go to <SKILL_CKPT>/controller/OGBench/Debug/sd000_<timestamp>/, so the
# pretrained params_*.pkl in each SKILL_CKPT itself is never touched.
#
# Usage (from anywhere):
#     SKILL_CKPTS="/path/to/skill_dt_run_a /path/to/skill_dt_run_b" \
#         scripts/run_skill_dt_controller.sh
# SKILL_CKPTS defaults to every run under impls/ckpts/skill_dt/*/ (the rsynced copies
# of run_skill_dt_sweep.sh's Savio runs). Other knobs, all optional:
#     GPUS="0 1"          GPU ids; at most one job per GPU at a time.
#     BASE_AGENT=gciql    inner algorithm (gciql | crl); add its knobs via EXTRA_FLAGS.
#     LABEL_MODE=window_mode   window_mode | end_state | future_hist.
#     CHUNK_HORIZON=20    H, the label window and option length (any value; nothing ties
#                         it to the checkpoint, unlike DDS/OPAL).
#     SKILL_HORIZON=      env steps a chosen skill is held for (default: CHUNK_HORIZON).
#     TRAIN_STEPS=1000000
#     EXTRA_FLAGS="--agent.base.expectile=0.9 --agent.base.alpha=3"
#
# The checkpoint's own `eval_max_steps` (201 on the -stitch datasets, see
# run_skill_dt_sweep.sh) is inherited by the controller's rollouts, so nothing needs
# to be passed for it here.
#
# At most NGPU jobs run at once: jobs are dispatched one per GPU and each batch is waited
# on before the next starts. -e so a mistyped SKILL_CKPT fails here rather than launching
# a job with an empty --env_name / --skill_restore_epoch.
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=${MUJOCO_GL:-egl}

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

read -r -a GPUS <<< "${GPUS:-0 1}"
NGPU=${#GPUS[@]}

if [[ -n "${SKILL_CKPTS:-}" ]]; then
    read -r -a SKILL_CKPTS <<< "$SKILL_CKPTS"
else
    SKILL_CKPTS=()
    for d in /nas/ucb/ishirgarg/ogbench/impls/ckpts/skill_dt/*/*/; do
        [[ -f "$d/flags.json" ]] && SKILL_CKPTS+=("${d%/}")
    done
    if [[ ${#SKILL_CKPTS[@]} -eq 0 ]]; then
        echo "no skill_dt runs under impls/ckpts/skill_dt/*/*/; pass SKILL_CKPTS=\"<run dir> ...\"" >&2
        exit 1
    fi
fi

BASE_AGENT=${BASE_AGENT:-gciql}
LABEL_MODE=${LABEL_MODE:-window_mode}
CHUNK_HORIZON=${CHUNK_HORIZON:-20}
TRAIN_STEPS=${TRAIN_STEPS:-1000000}
SKILL_HORIZON_FLAG=()
if [[ -n "${SKILL_HORIZON:-}" ]]; then
    SKILL_HORIZON_FLAG=(--agent.skill_horizon="$SKILL_HORIZON")
fi
read -r -a EXTRA_FLAGS <<< "${EXTRA_FLAGS:-}"

LOG_DIR=logs/skill_dt_controller
mkdir -p "$LOG_DIR"

pids=()
logs=()
fail=0
i=0
for SKILL_CKPT in "${SKILL_CKPTS[@]}"; do
    ENV_NAME=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    AGENT_NAME=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['agent']['agent_name'])" "$SKILL_CKPT")
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
    if [[ "$AGENT_NAME" != "skill_dt" ]]; then
        echo "$SKILL_CKPT is a '$AGENT_NAME' run, not skill_dt" >&2
        exit 1
    fi

    mkdir -p "$SAVE_DIR"
    GPU=${GPUS[$((i % NGPU))]}
    LOG="$LOG_DIR/$(basename "$(dirname "$SKILL_CKPT")")_$(basename "$SKILL_CKPT").log"

    echo "ckpt=$SKILL_CKPT"
    echo "  epoch=$SKILL_EPOCH env=$ENV_NAME base=$BASE_AGENT label_mode=$LABEL_MODE H=$CHUNK_HORIZON save_dir=$SAVE_DIR gpu=$GPU log=$LOG"

    # In-training eval runs the 20-step Transformer context every env step, so it is
    # kept on the GPU (eval_on_cpu=0) rather than main.py's CPU default.
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main.py \
        --env_name="$ENV_NAME" \
        --save_dir="$SAVE_DIR" \
        --agent=agents/skill_dt_controller.py:"$BASE_AGENT" \
        --agent.skill_checkpoint_path="$SKILL_CKPT" \
        --agent.skill_restore_epoch="$SKILL_EPOCH" \
        --agent.chunk_horizon="$CHUNK_HORIZON" \
        --agent.label_mode="$LABEL_MODE" \
        "${SKILL_HORIZON_FLAG[@]}" \
        "${EXTRA_FLAGS[@]}" \
        --train_steps="$TRAIN_STEPS" \
        --log_interval=5000 \
        --eval_interval=100000 \
        --save_interval=100000 \
        --eval_episodes=50 \
        --eval_on_cpu=0 \
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
