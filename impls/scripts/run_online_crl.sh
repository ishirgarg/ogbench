#!/usr/bin/env bash
# Online CRL baseline (flat goal-conditioned agent, no skill controller) on an
# OGBench env, learning from its own rollouts. Mirrors JaxGCRL's flat `crl`
# agent; see agents/online_crl.py and main_online.py.
#
# Launches one run per entry of ENVS on the matching GPU. Task pairs come from
# the env registration, so a custom online task set is just another env name.
#
# RLPD is on by default: every batch mixes rows from the OGBench dataset named
# by OFFLINE_DATASET (a single name for all ENVS, or per-env via the map below;
# set OFFLINE_DATASET=none to train from online data only).
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=${MUJOCO_GL:-egl}

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

read -r -a ENVS <<< "${ENVS:-antmaze-medium-navigate-v0}"
read -r -a GPUS <<< "${GPUS:-0}"
SEED=${SEED:-0}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
EPISODE_LENGTH=${EPISODE_LENGTH:-}   # empty -> per-env default below (antsoccer: 500, else env's registered horizon)
OFFLINE_DATASET=${OFFLINE_DATASET:-}  # empty -> per-env default below; "none" -> no RLPD

# Default offline dataset per online env (the env family's OGBench dataset).
declare -A OFFLINE_DEFAULTS=(
    [antmaze-medium-center-online-v0]=antmaze-medium-navigate-v0
    [antsoccer-arena-center-online-v0]=antsoccer-arena-navigate-v0
)
offline_dataset_for() {
    if [[ "$OFFLINE_DATASET" == "none" ]]; then echo ""; return; fi
    if [[ -n "$OFFLINE_DATASET" ]]; then echo "$OFFLINE_DATASET"; return; fi
    echo "${OFFLINE_DEFAULTS[$1]:-}"
}

# Default episode horizon per online env (antsoccer is shorter than the antmaze's registered 1000).
declare -A EPISODE_LENGTH_DEFAULTS=(
    [antsoccer-arena-center-online-v0]=500
)
episode_length_for() {
    if [[ -n "$EPISODE_LENGTH" ]]; then echo "$EPISODE_LENGTH"; return; fi
    echo "${EPISODE_LENGTH_DEFAULTS[$1]:-}"
}

LOG_DIR=logs/online_crl
mkdir -p "$LOG_DIR"

pids=()
for i in "${!ENVS[@]}"; do
    ENV_NAME=${ENVS[$i]}
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    EP_LEN=$(episode_length_for "$ENV_NAME")
    EP_FLAG=()
    if [[ -n "$EP_LEN" ]]; then EP_FLAG=(--episode_length="$EP_LEN"); fi
    OFFLINE=$(offline_dataset_for "$ENV_NAME")
    RLPD_FLAG=()
    TAG=norlpd
    if [[ -n "$OFFLINE" ]]; then RLPD_FLAG=(--offline_dataset="$OFFLINE"); TAG=rlpd; fi
    LOG="$LOG_DIR/${ENV_NAME}_${TAG}_s${SEED}.log"
    echo "launching online_crl env=${ENV_NAME} offline=${OFFLINE:-none} on GPU ${GPU} -> ${LOG}"
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main_online.py \
        --env_name="$ENV_NAME" \
        --seed="$SEED" \
        --agent=agents/online_crl.py \
        --total_env_steps="$TOTAL_ENV_STEPS" \
        "${EP_FLAG[@]}" \
        "${RLPD_FLAG[@]}" \
        --log_interval=5000 \
        --eval_interval=20000 \
        --save_interval=100000 \
        --eval_episodes=20 \
        --video_episodes=0 \
        > "$LOG" 2>&1 &
    pids+=($!)
    # Stagger: exp_name has 1-second resolution, so same-second launches would share a run dir.
    sleep 5
done

echo "waiting on ${#pids[@]} jobs..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit $fail
