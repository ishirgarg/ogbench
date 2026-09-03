#!/usr/bin/env bash
# Online CRL (contrastive) high-level skill controller over a FROZEN OGBench
# skill checkpoint (empowerment_skill or dds), learning from its own rollouts.
# Mirrors JaxGCRL's `go-explore-simple --agent_type crl_skill`; see
# agents/online_crl_skill_controller.py and main_online.py.
#
# SKILL_CKPT is the pretrained run dir (flags.json + params_*.pkl). ENV_NAME
# defaults to the checkpoint's own env; override it to train on a custom online
# task set registered under a different env name. RLPD is on by default:
# OFFLINE_DATASET (default: the checkpoint's own dataset, i.e. the data the skills
# were trained on; "none" disables RLPD) is labelled with the frozen skill agent
# and mixed into every batch. Checkpoints go to
# <SKILL_CKPT>/online_controller/{rlpd,norlpd}/OGBench/<run_group>/sd000_<timestamp>/, so the
# pretrained params_*.pkl in SKILL_CKPT itself is never touched.
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=${MUJOCO_GL:-egl}

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

SKILL_CKPT=${SKILL_CKPT:?set SKILL_CKPT=<skill-agent run dir>}
ENV_NAME=${ENV_NAME:-$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")}
# Resolve the pretrained epoch here and pass it explicitly so flags.json records it
# (main_online.py writes flags.json before constructing the agent).
SKILL_EPOCH=${SKILL_EPOCH:-$($PYTHON -c "
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl\$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
" "$SKILL_CKPT")}
SKILL_DATASET=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
OFFLINE_DATASET=${OFFLINE_DATASET:-$SKILL_DATASET}   # "none" -> no RLPD
TAG=rlpd; [[ "$OFFLINE_DATASET" == "none" ]] && TAG=norlpd
SAVE_DIR="$SKILL_CKPT/online_controller/$TAG"

GPU=${GPU:-0}
SEED=${SEED:-0}
K=${K:-10}                        # skill_commitment_k
ENT=${ENT:-0.5}                   # target_entropy_multiplier (same formula as online_crl)
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
EPISODE_LENGTH=${EPISODE_LENGTH:-}   # empty -> env's registered horizon (must be divisible by K)

if [[ -z "$ENV_NAME" || -z "$SKILL_EPOCH" ]]; then
    echo "could not read env_name / latest epoch from $SKILL_CKPT" >&2
    exit 1
fi

LOG_DIR=logs/online_crl_skill_controller
mkdir -p "$LOG_DIR" "$SAVE_DIR"
LOG="$LOG_DIR/$(basename "$SKILL_CKPT")_${ENV_NAME}_${TAG}_k${K}_ent${ENT}_s${SEED}.log"
EP_FLAG=()
if [[ -n "$EPISODE_LENGTH" ]]; then EP_FLAG=(--episode_length="$EPISODE_LENGTH"); fi
RLPD_FLAG=()
if [[ "$OFFLINE_DATASET" != "none" ]]; then RLPD_FLAG=(--offline_dataset="$OFFLINE_DATASET"); fi

echo "ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH env=$ENV_NAME offline=$OFFLINE_DATASET k=$K ent=$ENT gpu=$GPU -> $LOG"
CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main_online.py \
    --env_name="$ENV_NAME" \
    --seed="$SEED" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/online_crl_skill_controller.py \
    --agent.skill_checkpoint_path="$SKILL_CKPT" \
    --agent.skill_restore_epoch="$SKILL_EPOCH" \
    --agent.skill_commitment_k="$K" \
    --agent.target_entropy_multiplier="$ENT" \
    --total_env_steps="$TOTAL_ENV_STEPS" \
    "${EP_FLAG[@]}" \
    "${RLPD_FLAG[@]}" \
    --log_interval=5000 \
    --eval_interval=20000 \
    --save_interval=100000 \
    --eval_episodes=20 \
    --video_episodes=0 \
    > "$LOG" 2>&1 &
wait $!
