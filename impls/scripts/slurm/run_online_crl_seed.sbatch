#!/usr/bin/env bash
# Single online_crl (flat CRL, no skill controller) training run, for the
# rnn.ist.berkeley.edu / BAIR "wild west" Slurm cluster (NOT Savio -- see
# scripts/rail_slurm_offline_crl_gciql_navigate_normal.sh for that cluster's
# conventions, they don't apply here: no --account/--partition, paths are
# under /nas/ucb, GPUs are requested generically via --gpus).
#
# One run = one seed of agents/online_crl.py on --env_name, optionally mixing
# in --offline_dataset via RLPD (see utils/rlpd.py, agents/online_crl.py). This
# mirrors the per-run body of scripts/run_online_crl.sh (same flags/intervals),
# but sized for a single Slurm job instead of stacking several local runs.
#
# Usage (normally invoked by scripts/slurm/submit_online_crl_rlpd_seeds.sh, not
# called directly):
#   sbatch --job-name=<name> --output=<log> scripts/slurm/run_online_crl_seed.sbatch \
#       <ENV_NAME> <OFFLINE_DATASET|none> <SEED> [EPISODE_LENGTH]
#
# EPISODE_LENGTH is optional; omit (or pass "") to use the env's registered
# horizon (antsoccer-arena-center-online-v0 needs 500 -- the submit script
# already passes this; see EPISODE_LENGTH_DEFAULTS in run_online_crl.sh).
#
# IMPORTANT -- read before submitting:
#   * wandb credentials: Slurm jobs do NOT have access to your rnn home
#     directory (only /nas/ucb/<you> and /nas/ttl=60d/<you> are shared), so a
#     ~/.netrc-based wandb login on rnn will NOT be visible inside the job.
#     WANDB_API_KEY is loaded below from /nas/ucb/ishirgarg/.wandb_api_key, a
#     private (chmod 600, owner-only) file OUTSIDE this git repo -- never put
#     the raw key in this tracked script, the NAS is readable by every user on
#     the cluster and this file would also land in git history.
#   * This has not been tested on the actual rnn/Slurm cluster -- the conda
#     env path and CUDA driver compatibility on the assigned compute node are
#     unverified from here.
#SBATCH --job-name=online_crl
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --qos=default
set -euo pipefail

ENV_NAME=${1:?usage: run_online_crl_seed.sbatch ENV_NAME OFFLINE_DATASET SEED [EPISODE_LENGTH]}
OFFLINE_DATASET=${2:?usage: run_online_crl_seed.sbatch ENV_NAME OFFLINE_DATASET SEED [EPISODE_LENGTH]}
SEED=${3:?usage: run_online_crl_seed.sbatch ENV_NAME OFFLINE_DATASET SEED [EPISODE_LENGTH]}
EPISODE_LENGTH=${4:-}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_online_crl_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
fi
# Disposable wandb staging data -> the 60-day scratch area, not the permanent NAS quota.
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp
mkdir -p "$TMPDIR"
# Shared, persistent compile cache on the NAS so the 18 seeds (likely running on
# different nodes/GPU types) don't each pay a cold-start XLA compile. JAX's
# persistent cache writes are keyed/atomic per compiled program, so concurrent
# writers are expected to be safe, but this hasn't been stress-tested by us.
export JAX_COMPILATION_CACHE_DIR="$REPO_DIR/impls/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
# Grow GPU memory on demand rather than preallocating a fixed fraction: with
# --gpus=1 we get a whole, exclusive GPU, but its size varies across the
# cluster's A4000/A6000/A100 mix, so a fixed preallocation fraction sized for
# one card could be wrong for another.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$IMPLS_DIR"

RLPD_FLAG=()
TAG=norlpd
if [[ "$OFFLINE_DATASET" != "none" ]]; then
    RLPD_FLAG=(--offline_dataset="$OFFLINE_DATASET")
    TAG=rlpd
fi
EP_FLAG=()
if [[ -n "$EPISODE_LENGTH" ]]; then EP_FLAG=(--episode_length="$EPISODE_LENGTH"); fi

echo "[run_online_crl_seed] env=$ENV_NAME offline=$OFFLINE_DATASET seed=$SEED episode_length=${EPISODE_LENGTH:-<registered>} job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

srun "$PYTHON" -u main_online.py \
    --env_name="$ENV_NAME" \
    --seed="$SEED" \
    --agent=agents/online_crl.py \
    --total_env_steps=1000000 \
    "${EP_FLAG[@]}" \
    "${RLPD_FLAG[@]}" \
    --log_interval=5000 \
    --eval_interval=20000 \
    --save_interval=100000 \
    --eval_episodes=20 \
    --video_episodes=0
