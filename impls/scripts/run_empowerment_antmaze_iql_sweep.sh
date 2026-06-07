#!/usr/bin/env bash
# Local (non-SLURM) version of empowerment_antmaze_iql_sweep.sh.
# Runs at most one job per GPU at a time (GPU-token semaphore); queued jobs
# start as soon as a GPU frees up.
#
# For each of two pretrained empowerment estimators we generate an
# antmaze-medium-navigate dataset whose *initial cell* is sampled proportionally
# to empowerment (everything else identical to the original dataset's policy and
# budget), sweeping 4 weighting configs: linear, softmax@0.5/1.0/2.0
#   -> 2 ckpts x 4 configs = 8 datasets. Plus the original dataset = 9 GCIQL runs.
#
# Each job generates only its own (distinctly named) dataset, so parallel jobs
# never write the same file.

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

CFG_TAGS=(linear softmax_t0.5 softmax_t1 softmax_t2)
CFG_ARGS=(
    "--empowerment_sampling=linear"
    "--empowerment_sampling=softmax --empowerment_temp=0.5"
    "--empowerment_sampling=softmax --empowerment_temp=1.0"
    "--empowerment_sampling=softmax --empowerment_temp=2.0"
)

# Original dataset's policy/budget (kept identical for the empowerment variants).
ENV_GEN=antmaze-medium-v0
ENV_TRAIN=antmaze-medium-navigate-v0
NUM_EPISODES=1000
MAX_EPISODE_STEPS=1001

# -----------------------------
# Per-job body: generate (if missing) then train GCIQL.
# Args: CKPT_IDX CFG_IDX   (CKPT_IDX<0 => original-dataset run)
# -----------------------------
run_emp_variant() {
    set -euo pipefail
    local GPU=$1 CKPT_IDX=$2 CFG_IDX=$3
    export CUDA_VISIBLE_DEVICES=${GPU}
    local CKPT_PATH=${CKPT_PATHS[${CKPT_IDX}]}
    local CKPT_LABEL=${CKPT_LABELS[${CKPT_IDX}]}
    local CFG_TAG=${CFG_TAGS[${CFG_IDX}]}
    local CFG_ARG=${CFG_ARGS[${CFG_IDX}]}

    local DS_TAG="antmaze-medium-navigate-emp-${CKPT_LABEL}-${CFG_TAG}"
    local DS_PATH="${DATA_DIR}/${DS_TAG}-v0.npz"

    echo "ckpt=${CKPT_LABEL} (${CKPT_PATH}) config=${CFG_TAG} dataset=${DS_PATH}"

    # 1) Generate the empowerment-weighted dataset (skip if it already exists).
    if [[ -f "${DS_PATH}" ]]; then
        echo "Dataset already exists, skipping generation: ${DS_PATH}"
    else
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
    fi

    # 2) Train GCIQL (IQL with DDPG+BC actor loss) on the generated dataset.
    cd "${IMPLS_DIR}"
    python main.py \
        --env_name=${ENV_TRAIN} \
        --dataset_path="${DS_PATH}" \
        --eval_episodes=50 \
        --agent=agents/gciql.py \
        --agent.alpha=0.1 \
        --save_dir=${SAVE_DIR} \
        --run_group=emp_antmaze_iql_${CKPT_LABEL}_${CFG_TAG}
}

run_original() {
    set -euo pipefail
    local GPU=$1
    export CUDA_VISIBLE_DEVICES=${GPU}
    echo "Training GCIQL on the original ${ENV_TRAIN} (uniform starts)."
    cd "${IMPLS_DIR}"
    python main.py \
        --env_name=${ENV_TRAIN} \
        --eval_episodes=50 \
        --agent=agents/gciql.py \
        --agent.alpha=0.1 \
        --save_dir=${SAVE_DIR} \
        --run_group=emp_antmaze_iql_original
}

# Recollect an OGBench-style dataset from scratch (uniform starts, no empowerment
# sampling) using the same policy/budget as the empowerment variants, then train
# GCIQL on it. This is the apples-to-apples baseline: same generation pipeline,
# only the start-cell distribution differs from the empowerment runs.
run_recollected() {
    set -euo pipefail
    local GPU=$1
    export CUDA_VISIBLE_DEVICES=${GPU}

    local DS_TAG="antmaze-medium-navigate-recollected"
    local DS_PATH="${DATA_DIR}/${DS_TAG}-v0.npz"

    echo "Recollecting uniform dataset (no empowerment): ${DS_PATH}"

    # 1) Generate the uniform dataset (skip if it already exists).
    if [[ -f "${DS_PATH}" ]]; then
        echo "Dataset already exists, skipping generation: ${DS_PATH}"
    else
        cd "${GEN_DIR}"
        PYTHONPATH="${IMPLS_DIR}:${PYTHONPATH:-}" python generate_locomaze.py \
            --env_name=${ENV_GEN} \
            --save_path="${DS_PATH}" \
            --dataset_type=navigate \
            --num_episodes=${NUM_EPISODES} \
            --max_episode_steps=${MAX_EPISODE_STEPS} \
            --restore_path=${EXPERTS} \
            --restore_epoch=${EXPERT_EPOCH} \
            --log_wandb \
            --wandb_project=ogbench_datagen \
            --wandb_name="${DS_TAG}"
    fi

    # 2) Train GCIQL on the recollected dataset.
    cd "${IMPLS_DIR}"
    python main.py \
        --env_name=${ENV_TRAIN} \
        --dataset_path="${DS_PATH}" \
        --eval_episodes=50 \
        --agent=agents/gciql.py \
        --agent.alpha=0.1 \
        --save_dir=${SAVE_DIR} \
        --run_group=emp_antmaze_iql_recollected
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

# 8 empowerment variants: 2 ckpts x 4 configs.
for CKPT_IDX in 0 1; do
    for CFG_IDX in 0 1 2 3; do
        read -u 9 GPU      # block until a GPU is free
        TAG="emp_${CKPT_LABELS[${CKPT_IDX}]}_${CFG_TAGS[${CFG_IDX}]}"
        echo "JOB=${JOB} GPU=${GPU} ${TAG}"
        (
            trap 'echo "${GPU}" >&9' EXIT
            run_emp_variant "${GPU}" "${CKPT_IDX}" "${CFG_IDX}" \
                > "${LOG_DIR}/${TAG}.log" 2>&1
        ) &
        JOB=$((JOB + 1))
        sleep 2   # stagger near-simultaneous mujoco-EGL inits
    done
done

# Original dataset (uniform starts).
read -u 9 GPU
echo "JOB=${JOB} GPU=${GPU} original"
(
    trap 'echo "${GPU}" >&9' EXIT
    run_original "${GPU}" > "${LOG_DIR}/original.log" 2>&1
) &
JOB=$((JOB + 1))

# Recollected OGBench-style dataset (uniform starts, no empowerment sampling).
read -u 9 GPU
echo "JOB=${JOB} GPU=${GPU} recollected"
(
    trap 'echo "${GPU}" >&9' EXIT
    run_recollected "${GPU}" > "${LOG_DIR}/recollected.log" 2>&1
) &
JOB=$((JOB + 1))

echo "Launched ${JOB} jobs across ${NUM_GPUS} GPUs (<=1 per GPU at a time). Waiting..."
wait
exec 9>&-
echo "All ${JOB} jobs finished."
