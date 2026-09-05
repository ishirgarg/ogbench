#!/usr/bin/env bash
# Single online_crl_skill_controller run on top of a FROZEN **DDS** skill
# checkpoint, with RLPD, for the rnn.ist.berkeley.edu Slurm cluster.
#
# Identical in structure and hyperparameters to
# run_online_crl_skill_controller_seed.sbatch (same K=10 commitment,
# target_entropy_multiplier=0.5, 1M env steps, same intervals) -- the only
# difference is that SKILL_CKPT points at a `dds` run instead of an
# `empowerment_skill` run, so the online controller's seeds are directly
# comparable across the two skill families.
#
# Why K=10 specifically: utils/skill_checkpoint.py loads the DDS agent and the
# controller's DDS offline-window labeller (online_crl_skill_controller.py)
# REQUIRES skill_commitment_k == the checkpoint's `sequence_length`, which is 10
# for all five ckpts/final/dds runs. main_online.py also requires
# episode_horizon % k == 0, which holds for 1000 and 500.
#
# See run_online_crl_seed.sbatch for the cluster-specific notes this mirrors
# (wandb key file, NAS dataset cache, JAX compile cache) -- read those before
# submitting if you haven't already.
#
# Usage (normally invoked by scripts/slurm/submit_dds_online_controller_seeds.sh):
#   sbatch --job-name=<name> --output=<log> \
#       scripts/slurm/run_dds_online_controller_seed.sbatch \
#       <SKILL_CKPT> <ENV_NAME> <SEED> [OFFLINE_DATASET] [EPISODE_LENGTH]
#
# OFFLINE_DATASET is optional; omit (or pass "") to use SKILL_CKPT's own dataset
# (its flags.json env_name) -- i.e. exactly the data the DDS skills were trained
# on, which is what we want for RLPD. Pass "none" to disable RLPD.
# EPISODE_LENGTH is optional; omit to use the env's registered horizon
# (antsoccer-arena-center-online-v0 needs 500 -- the submit script passes it).
#SBATCH --job-name=dds_online_controller
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --qos=default
set -euo pipefail

SKILL_CKPT=${1:?usage: run_dds_online_controller_seed.sbatch SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET] [EPISODE_LENGTH]}
ENV_NAME=${2:?usage: run_dds_online_controller_seed.sbatch SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET] [EPISODE_LENGTH]}
SEED=${3:?usage: run_dds_online_controller_seed.sbatch SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET] [EPISODE_LENGTH]}
OFFLINE_DATASET=${4:-}
EPISODE_LENGTH=${5:-}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_dds_online_controller_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
fi
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp
mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$REPO_DIR/impls/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# See run_online_crl_seed.sbatch: avoids a per-compute-node $HOME cache miss
# triggering a download that can time out with no internet egress.
export OGBENCH_DATASET_DIR=/nas/ucb/ishirgarg/.ogbench/data

cd "$IMPLS_DIR"

SKILL_EPOCH=$($PYTHON -c "
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl\$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
" "$SKILL_CKPT")
SKILL_DATASET=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
if [[ -z "$OFFLINE_DATASET" ]]; then OFFLINE_DATASET="$SKILL_DATASET"; fi
TAG=rlpd
RLPD_FLAG=(--offline_dataset="$OFFLINE_DATASET")
if [[ "$OFFLINE_DATASET" == "none" ]]; then TAG=norlpd; RLPD_FLAG=(); fi
EP_FLAG=()
if [[ -n "$EPISODE_LENGTH" ]]; then EP_FLAG=(--episode_length="$EPISODE_LENGTH"); fi
SAVE_DIR="$SKILL_CKPT/online_controller/$TAG"
mkdir -p "$SAVE_DIR"

echo "[run_dds_online_controller_seed] ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH env=$ENV_NAME offline=$OFFLINE_DATASET seed=$SEED episode_length=${EPISODE_LENGTH:-<registered>} job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

srun "$PYTHON" -u main_online.py \
    --env_name="$ENV_NAME" \
    --seed="$SEED" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/online_crl_skill_controller.py \
    --agent.skill_checkpoint_path="$SKILL_CKPT" \
    --agent.skill_restore_epoch="$SKILL_EPOCH" \
    --agent.skill_commitment_k=10 \
    --agent.target_entropy_multiplier=0.5 \
    --total_env_steps=1000000 \
    "${EP_FLAG[@]}" \
    "${RLPD_FLAG[@]}" \
    --log_interval=5000 \
    --eval_interval=20000 \
    --save_interval=100000 \
    --eval_episodes=20 \
    --video_episodes=0
