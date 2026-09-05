#!/usr/bin/env bash
# Single online_crl_skill_controller (high-level skill controller over a FROZEN
# skill checkpoint) training run with RLPD, for the rnn.ist.berkeley.edu Slurm
# cluster. See run_online_crl_seed.sbatch for the cluster-specific notes
# (wandb key file, NAS dataset cache) this mirrors -- read those before
# submitting if you haven't already.
#
# One run = one seed of agents/online_crl_skill_controller.py on top of a
# frozen SKILL_CKPT, using its own dataset for RLPD (same default as
# scripts/run_online_crl_skill_controller.sh: SKILL_CKPT's flags.json env_name).
# Flags match that script's body (K=10, target_entropy_multiplier=0.5 -- the
# clamped formula from agents/online_crl_skill_controller.py, verified fixed
# 2026-09-02) so seeds are directly comparable to the already-finished seed-0
# runs under <SKILL_CKPT>/online_controller/rlpd/.
#
# Usage (normally invoked by
# scripts/slurm/submit_online_crl_skill_controller_rlpd_seeds.sh):
#   sbatch --job-name=<name> --output=<log> \
#       scripts/slurm/run_online_crl_skill_controller_seed.sbatch \
#       <SKILL_CKPT> <ENV_NAME> <SEED> [OFFLINE_DATASET]
#
# OFFLINE_DATASET is optional; omit (or pass "") to use SKILL_CKPT's own
# dataset (its flags.json env_name), matching the local script's default.
# Pass "none" to disable RLPD.
#SBATCH --job-name=online_crl_skill_controller
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --qos=default
set -euo pipefail

SKILL_CKPT=${1:?usage: run_online_crl_skill_controller_seed.sbatch SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET]}
ENV_NAME=${2:?usage: run_online_crl_skill_controller_seed.sbatch SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET]}
SEED=${3:?usage: run_online_crl_skill_controller_seed.sbatch SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET]}
OFFLINE_DATASET=${4:-}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_online_crl_skill_controller_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
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
SAVE_DIR="$SKILL_CKPT/online_controller/$TAG"
mkdir -p "$SAVE_DIR"

echo "[run_online_crl_skill_controller_seed] ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH env=$ENV_NAME offline=$OFFLINE_DATASET seed=$SEED job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

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
    "${RLPD_FLAG[@]}" \
    --log_interval=5000 \
    --eval_interval=20000 \
    --save_interval=100000 \
    --eval_episodes=20 \
    --video_episodes=0
