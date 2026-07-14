#!/usr/bin/env bash
# Local (non-SLURM) version of empowerment_antmaze_iql_sweep.sh.
# Runs at most one job per GPU at a time (GPU-token semaphore); queued jobs
# start as soon as a GPU frees up.
#
# For each of two pretrained empowerment estimators we generate an
# antmaze-medium-navigate dataset whose *initial cell* is sampled proportionally
# to empowerment (everything else identical to the original dataset's policy and
# budget), sweeping 2 weighting configs: linear and softmax@1.0
#   -> 2 ckpts x 2 configs = 4 empowerment datasets.
# We also recollect 1 OGBench-style dataset from scratch (uniform starts, no
# empowerment) using the same policy/budget -> 5 generated datasets total.
#
# We then train GCIQL, swept over the actor-loss alpha [0.03 0.1 0.3 1], on:
#   - each of the 4 empowerment datasets          (16 runs)
#   - the recollected dataset                     ( 4 runs)
#   - the original built-in OGBench dataset        ( 4 runs, no --dataset_path)
#   -> 24 GCIQL runs total.
#
# Generated datasets are created first (distinctly named, so parallel generation
# never writes the same file); the alpha training jobs run afterwards and only
# read the already-generated datasets.

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export WANDB_API_KEY='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'
export MUJOCO_GL=egl

# -----------------------------
# Machine configuration
# -----------------------------
# GPU ids to use. At most one job runs on a GPU at a time; extra jobs wait in a
# queue until a GPU frees up (no oversubscription).
GPUS=(0 1 2 3 4 6 7)

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GEN_DIR="${REPO}/data_gen_scripts"
IMPLS_DIR="${REPO}/impls"

DATA_DIR="${HOME}/.ogbench/data"          # ogbench's default dataset location
SAVE_DIR=/nas/ucb/ishirgarg/ogbench
EXPERTS="${REPO}/data/experts/ant"         # ant SAC expert (absolute path)
EXPERT_EPOCH=400000
LOG_DIR="${IMPLS_DIR}/logs/emp_antmaze_iql_sweep"
mkdir -p "${DATA_DIR}" "${LOG_DIR}"

# ── Sweep definitions ───────────────────────────────────────────────────────
CKPT_PATHS=(
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/antmaze-medium-navigate/sd000_s_34594838.0.20260527_234324
    /nas/ucb/ishirgarg/ogbench/impls/ckpts/antmaze-medium-navigate/sd000_s_34594763.0.20260527_234148
)
CKPT_LABELS=(ckptA ckptB)

# Empowerment-weighting configs: linear and softmax @ temperature 1.0.
CFG_TAGS=(linear softmax_t1)
CFG_ARGS=(
    "--empowerment_sampling=linear"
    "--empowerment_sampling=softmax --empowerment_temp=1.0"
)

# GCIQL actor-loss alpha values to sweep per dataset.
ALPHAS=(0.03 0.1 0.3 1)

# Original dataset's policy/budget (kept identical for the empowerment variants).
ENV_GEN=antmaze-medium-v0
ENV_TRAIN=antmaze-medium-navigate-v0
NUM_EPISODES=1000
MAX_EPISODE_STEPS=1001

# Recollected OGBench-style dataset (uniform starts, no empowerment sampling).
RECOLLECTED_TAG="antmaze-medium-navigate-recollected"
RECOLLECTED_PATH="${DATA_DIR}/${RECOLLECTED_TAG}-v0.npz"

# Dataset path for a given (ckpt, config) index pair (shared across the alpha sweep).
ds_path_for() {
    local CKPT_LABEL=${CKPT_LABELS[$1]}
    local CFG_TAG=${CFG_TAGS[$2]}
    echo "${DATA_DIR}/antmaze-medium-navigate-emp-${CKPT_LABEL}-${CFG_TAG}-v0.npz"
}

# -----------------------------
# Generate the empowerment-weighted dataset for one (ckpt, config) pair.
# Args: GPU CKPT_IDX CFG_IDX
# -----------------------------
generate_dataset() {
    set -euo pipefail
    local GPU=$1 CKPT_IDX=$2 CFG_IDX=$3
    export CUDA_VISIBLE_DEVICES=${GPU}
    local CKPT_PATH=${CKPT_PATHS[${CKPT_IDX}]}
    local CKPT_LABEL=${CKPT_LABELS[${CKPT_IDX}]}
    local CFG_TAG=${CFG_TAGS[${CFG_IDX}]}
    local CFG_ARG=${CFG_ARGS[${CFG_IDX}]}
    local DS_PATH; DS_PATH=$(ds_path_for "${CKPT_IDX}" "${CFG_IDX}")
    local DS_TAG="antmaze-medium-navigate-emp-${CKPT_LABEL}-${CFG_TAG}"

    echo "ckpt=${CKPT_LABEL} (${CKPT_PATH}) config=${CFG_TAG} dataset=${DS_PATH}"

    # Always generate from scratch.
    cd "${GEN_DIR}"
    PYTHONPATH="${IMPLS_DIR}:${PYTHONPATH:-}" python generate_locomaze.py \
        --env_name=${ENV_GEN} \
        --save_path="${DS_PATH}" \
        --dataset_type=navigate \
        --num_episodes=${NUM_EPISODES} \
        --max_episode_steps=${MAX_EPISODE_STEPS} \
        --restore_path=${EXPERTS} \
        --restore_epoch=${EXPERT_EPOCH} \
        --empowerment_ckpt="${CKPT_PATH}" \
        ${CFG_ARG} \
        --log_wandb \
        --wandb_project=ogbench_datagen \
        --wandb_name="${DS_TAG}"
}

# -----------------------------
# Recollect an OGBench-style dataset from scratch (uniform starts, no empowerment
# sampling) using the same policy/budget as the empowerment variants.
# Args: GPU
# -----------------------------
generate_recollected() {
    set -euo pipefail
    local GPU=$1
    export CUDA_VISIBLE_DEVICES=${GPU}

    echo "Recollecting uniform dataset (no empowerment): ${RECOLLECTED_PATH}"

    # Always generate from scratch.
    cd "${GEN_DIR}"
    PYTHONPATH="${IMPLS_DIR}:${PYTHONPATH:-}" python generate_locomaze.py \
        --env_name=${ENV_GEN} \
        --save_path="${RECOLLECTED_PATH}" \
        --dataset_type=navigate \
        --num_episodes=${NUM_EPISODES} \
        --max_episode_steps=${MAX_EPISODE_STEPS} \
        --restore_path=${EXPERTS} \
        --restore_epoch=${EXPERT_EPOCH} \
        --log_wandb \
        --wandb_project=ogbench_datagen \
        --wandb_name="${RECOLLECTED_TAG}"
}

# -----------------------------
# Train GCIQL (IQL with DDPG+BC actor loss) on a dataset.
# Args: GPU RUN_LABEL ALPHA [DS_PATH]
# A non-empty DS_PATH trains on that dataset; an empty/omitted DS_PATH trains on
# the built-in OGBench dataset for ENV_TRAIN.
# -----------------------------
train_gciql() {
    set -euo pipefail
    local GPU=$1 RUN_LABEL=$2 ALPHA=$3 DS_PATH=${4:-}
    export CUDA_VISIBLE_DEVICES=${GPU}

    local DS_ARG=()
    if [[ -n "${DS_PATH}" ]]; then
        DS_ARG=(--dataset_path="${DS_PATH}")
        echo "Training GCIQL ${RUN_LABEL} alpha=${ALPHA} on ${DS_PATH}"
    else
        echo "Training GCIQL ${RUN_LABEL} alpha=${ALPHA} on built-in ${ENV_TRAIN} dataset"
    fi

    cd "${IMPLS_DIR}"
    python main.py \
        --env_name=${ENV_TRAIN} \
        ${DS_ARG[@]+"${DS_ARG[@]}"} \
        --eval_episodes=50 \
        --video_episodes=0 \
        --agent=agents/gciql.py \
        --agent.alpha=${ALPHA} \
        --save_dir=${SAVE_DIR} \
        --run_group=emp_antmaze_iql_${RUN_LABEL}_alpha${ALPHA}
}

# -----------------------------
# Dispatch: at most one job per GPU at a time (GPU-token semaphore).
# -----------------------------
NUM_GPUS=${#GPUS[@]}
JOB=0

# FIFO used as a semaphore: it holds one token (a GPU id) per free GPU. A job
# blocks on `read -u 9` until a token is available, then releases it on exit
# (via the EXIT trap, so a failed/`set -e` job still frees its GPU).
FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 9<>"$FIFO"
rm -f "$FIFO"
for GPU in "${GPUS[@]}"; do echo "$GPU"; done >&9

# Phase 1: generate the 5 datasets (4 empowerment: 2 ckpts x 2 configs, plus 1
# recollected; distinct names, safe in parallel). Wait for all before training
# so the alpha sweep only reads them.
for CKPT_IDX in 0 1; do
    for CFG_IDX in 0 1; do
        read -u 9 GPU      # block until a GPU is free
        TAG="gen_${CKPT_LABELS[${CKPT_IDX}]}_${CFG_TAGS[${CFG_IDX}]}"
        echo "GEN GPU=${GPU} ${TAG}"
        (
            trap 'echo "${GPU}" >&9' EXIT
            generate_dataset "${GPU}" "${CKPT_IDX}" "${CFG_IDX}" \
                > "${LOG_DIR}/${TAG}.log" 2>&1
        ) &
        sleep 2   # stagger near-simultaneous mujoco-EGL inits
    done
done
# Recollected dataset (uniform starts, no empowerment).
read -u 9 GPU
echo "GEN GPU=${GPU} gen_recollected"
(
    trap 'echo "${GPU}" >&9' EXIT
    generate_recollected "${GPU}" > "${LOG_DIR}/gen_recollected.log" 2>&1
) &
sleep 2
echo "Generating datasets; waiting for all before the alpha sweep..."
wait
echo "Datasets ready."

# Phase 2: 24 GCIQL training jobs (each dataset x 4 alphas).
# 2a) 4 empowerment datasets x 4 alphas = 16 runs.
for CKPT_IDX in 0 1; do
    for CFG_IDX in 0 1; do
        for ALPHA in "${ALPHAS[@]}"; do
            read -u 9 GPU      # block until a GPU is free
            RUN_LABEL="${CKPT_LABELS[${CKPT_IDX}]}_${CFG_TAGS[${CFG_IDX}]}"
            TAG="emp_${RUN_LABEL}_alpha${ALPHA}"
            DS_PATH=$(ds_path_for "${CKPT_IDX}" "${CFG_IDX}")
            echo "JOB=${JOB} GPU=${GPU} ${TAG}"
            (
                trap 'echo "${GPU}" >&9' EXIT
                train_gciql "${GPU}" "${RUN_LABEL}" "${ALPHA}" "${DS_PATH}" \
                    > "${LOG_DIR}/${TAG}.log" 2>&1
            ) &
            JOB=$((JOB + 1))
            sleep 2   # stagger near-simultaneous mujoco-EGL inits
        done
    done
done

# 2b) Recollected dataset x 4 alphas = 4 runs.
for ALPHA in "${ALPHAS[@]}"; do
    read -u 9 GPU
    TAG="recollected_alpha${ALPHA}"
    echo "JOB=${JOB} GPU=${GPU} ${TAG}"
    (
        trap 'echo "${GPU}" >&9' EXIT
        train_gciql "${GPU}" "recollected" "${ALPHA}" "${RECOLLECTED_PATH}" \
            > "${LOG_DIR}/${TAG}.log" 2>&1
    ) &
    JOB=$((JOB + 1))
    sleep 2
done

# 2c) Original built-in OGBench dataset x 4 alphas = 4 runs (no --dataset_path).
for ALPHA in "${ALPHAS[@]}"; do
    read -u 9 GPU
    TAG="original_alpha${ALPHA}"
    echo "JOB=${JOB} GPU=${GPU} ${TAG}"
    (
        trap 'echo "${GPU}" >&9' EXIT
        train_gciql "${GPU}" "original" "${ALPHA}" \
            > "${LOG_DIR}/${TAG}.log" 2>&1
    ) &
    JOB=$((JOB + 1))
    sleep 2
done

echo "Launched ${JOB} jobs across ${NUM_GPUS} GPUs (<=1 per GPU at a time). Waiting..."
wait
exec 9>&-
echo "All ${JOB} jobs finished."
