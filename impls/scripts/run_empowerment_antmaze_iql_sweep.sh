#!/usr/bin/env bash
# Local (non-SLURM) version of empowerment_antmaze_iql_sweep.sh.
# Runs at most one job per GPU at a time (GPU-token semaphore); queued jobs
# start as soon as a GPU frees up.
#
# For each of two pretrained empowerment estimators we generate an
# antmaze-medium-navigate dataset whose *initial cell* is sampled proportionally
# to empowerment (everything else identical to the original dataset's policy and
# budget), sweeping 2 weighting configs: linear and softmax@1.0
#   -> 2 ckpts x 2 configs = 4 datasets. On each dataset we then train GCIQL
# swept over the actor-loss alpha [0.03 0.1 0.3 1]
#   -> 4 datasets x 4 alphas = 16 GCIQL runs.
#
# Datasets are generated first (distinctly named, so parallel generation never
# writes the same file); the 16 alpha training jobs run afterwards and only read
# the already-generated datasets.

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
SAVE_DIR=/home/ishirgarg/ogbench
EXPERTS="${REPO}/data/experts/ant"         # ant SAC expert (absolute path)
EXPERT_EPOCH=400000
LOG_DIR="${IMPLS_DIR}/logs/emp_antmaze_iql_sweep"
mkdir -p "${DATA_DIR}" "${LOG_DIR}"

# ── Sweep definitions ───────────────────────────────────────────────────────
CKPT_PATHS=(
    /home/ishirgarg/ogbench/impls/ckpts/antmaze-medium-navigate/sd000_s_34594838.0.20260527_234324
    /home/ishirgarg/ogbench/impls/ckpts/antmaze-medium-navigate/sd000_s_34594763.0.20260527_234148
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
# Train GCIQL (IQL with DDPG+BC actor loss) on an already-generated dataset.
# Args: GPU CKPT_IDX CFG_IDX ALPHA
# -----------------------------
train_gciql() {
    set -euo pipefail
    local GPU=$1 CKPT_IDX=$2 CFG_IDX=$3 ALPHA=$4
    export CUDA_VISIBLE_DEVICES=${GPU}
    local CKPT_LABEL=${CKPT_LABELS[${CKPT_IDX}]}
    local CFG_TAG=${CFG_TAGS[${CFG_IDX}]}
    local DS_PATH; DS_PATH=$(ds_path_for "${CKPT_IDX}" "${CFG_IDX}")

    echo "Training GCIQL ckpt=${CKPT_LABEL} config=${CFG_TAG} alpha=${ALPHA} on ${DS_PATH}"

    cd "${IMPLS_DIR}"
    python main.py \
        --env_name=${ENV_TRAIN} \
        --dataset_path="${DS_PATH}" \
        --eval_episodes=50 \
        --agent=agents/gciql.py \
        --agent.alpha=${ALPHA} \
        --save_dir=${SAVE_DIR} \
        --run_group=emp_antmaze_iql_${CKPT_LABEL}_${CFG_TAG}_alpha${ALPHA}
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

# Phase 1: generate the 4 empowerment-weighted datasets (2 ckpts x 2 configs;
# distinct names, safe in parallel). Wait for all before training so the alpha
# sweep only reads them.
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
echo "Generating datasets; waiting for all before the alpha sweep..."
wait
echo "Datasets ready."

# Phase 2: 16 GCIQL training jobs: 4 datasets x 4 alphas.
for CKPT_IDX in 0 1; do
    for CFG_IDX in 0 1; do
        for ALPHA in "${ALPHAS[@]}"; do
            read -u 9 GPU      # block until a GPU is free
            TAG="emp_${CKPT_LABELS[${CKPT_IDX}]}_${CFG_TAGS[${CFG_IDX}]}_alpha${ALPHA}"
            echo "JOB=${JOB} GPU=${GPU} ${TAG}"
            (
                trap 'echo "${GPU}" >&9' EXIT
                train_gciql "${GPU}" "${CKPT_IDX}" "${CFG_IDX}" "${ALPHA}" \
                    > "${LOG_DIR}/${TAG}.log" 2>&1
            ) &
            JOB=$((JOB + 1))
            sleep 2   # stagger near-simultaneous mujoco-EGL inits
        done
    done
done

echo "Launched ${JOB} jobs across ${NUM_GPUS} GPUs (<=1 per GPU at a time). Waiting..."
wait
exec 9>&-
echo "All ${JOB} jobs finished."
