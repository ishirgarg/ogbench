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
# The per-fusion AUTOTUNE sub-cache that XLA puts *inside* the compilation cache dir is
# NOT safe for concurrent writers: with many seeds sharing the NAS cache, a job can read a
# temp .textproto another job has already renamed away and die with
# `JaxRuntimeError: NOT_FOUND: .../xla_gpu_per_fusion_autotune_cache_dir/tmp/...`.
# (Observed on jobs 1194122 / 1194127, 2026-09-04.) Give each job its own autotune dir; the
# module cache above stays shared, which is where the real cold-start saving is.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${SLURM_JOB_ID:-$$}"
# Grow GPU memory on demand rather than preallocating a fixed fraction: with
# --gpus=1 we get a whole, exclusive GPU, but its size varies across the
# cluster's A4000/A6000/A100 mix, so a fixed preallocation fraction sized for
# one card could be wrong for another.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# OGBench's dataset cache defaults to ~/.ogbench/data, which resolves per-machine.
# A Slurm job's $HOME is the *compute node's own* (unsynced) home directory, not
# rnn's, so a cache miss there triggers a download -- and a compute node may have
# no internet egress, which is exactly what timed out here. Point at a
# NAS-resident cache instead so every job shares it and none of them need network
# access. Keep this in sync with the datasets actually pre-downloaded there.
export OGBENCH_DATASET_DIR=/nas/ucb/ishirgarg/.ogbench/data

cd "$IMPLS_DIR"

RLPD_FLAG=()
TAG=norlpd
if [[ "$OFFLINE_DATASET" != "none" ]]; then
    RLPD_FLAG=(--offline_dataset="$OFFLINE_DATASET")
    TAG=rlpd
fi
# RLPD_FRAC_TIME (env var, optional): --rlpd_frac_time, the fraction of total_env_steps during
# which offline data is mixed in; batches are online-only afterwards (main_online.py, default 1.0).
# Only meaningful with RLPD on; the tag gets a "_ft<frac>" suffix so save dirs / logs do not collide.
if [[ -n "${RLPD_FRAC_TIME:-}" && ${#RLPD_FLAG[@]} -gt 0 ]]; then
    RLPD_FLAG+=(--rlpd_frac_time="$RLPD_FRAC_TIME")
    TAG="${TAG}_ft${RLPD_FRAC_TIME}"
fi
EP_FLAG=()
if [[ -n "$EPISODE_LENGTH" ]]; then EP_FLAG=(--episode_length="$EPISODE_LENGTH"); fi
# EMP_CKPT_DIR (env var, optional; sbatch exports the submitter's environment): a frozen
# empowerment_skill run dir for agents/online_crl.py's empowerment-modulated entropy target,
# with EMP_LAMBDA (1.0), EMP_NUM_BINS (8), EMP_NUM_SPLUS_SAMPLES (64). Tag suffix "_emp<lambda>".
# EMP_ENTROPY_TARGET=False keeps the estimator but drops the per-state target (tag "_noent").
# ADD_EXPLORE=reward | reward-to-rlpd (env var, optional) turns on the exploration reward bonus
# (agents/online_crl.py `add_explore`, r_x = E(s') - E_mean into its own critic Q_x) with
# BONUS_SCALE (1.0) the actor weight on Q_x and EXPLORE_REWARD (empowerment | max_episodic_empowerment,
# the running max of E over the episode; tag "max"); needs EMP_CKPT_DIR. BONUS_TIME_FRAC (empty ->
# agent default, no annealing) linearly decays bonus_scale to 0 by that fraction of total_env_steps (tag "_ann<frac>").
# Tag "_rb[max]<scale>[_ann<frac>]" / "_rbrlpd[max]<scale>[_ann<frac>]".
# ADD_EXPLORE=distill | distill-to-rlpd: the alternative bonus, E'(s, a) regressed onto the max of E over the whole trajectory (DISTILL_TARGET=future_max: states after s_t only, tag "_fut")
# along the trajectory and added to the actor loss (online rows only | online + RLPD rows); same BONUS_SCALE /
# BONUS_TIME_FRAC, EXPLORE_REWARD unused. Tag "_ed<scale>" / "_edrlpd<scale>".
EMP_FLAG=()
if [[ -n "${EMP_CKPT_DIR:-}" ]]; then
    EMP_FLAG=(
        --agent.emp_checkpoint_path="$EMP_CKPT_DIR"
        --agent.emp_lambda="${EMP_LAMBDA:-1.0}"
        --agent.emp_num_bins="${EMP_NUM_BINS:-8}"
        --agent.emp_num_splus_samples="${EMP_NUM_SPLUS_SAMPLES:-64}"
    )
    case "${EMP_ENTROPY_TARGET:-}" in
        False|false|0) EMP_FLAG+=(--agent.emp_entropy_target=False); TAG="${TAG}_noent" ;;
        *) TAG="${TAG}_emp${EMP_LAMBDA:-1.0}" ;;
    esac
fi
if [[ -n "${ADD_EXPLORE:-}" ]]; then
    if [[ -z "${EMP_CKPT_DIR:-}" ]]; then
        echo "[run_online_crl_seed] ADD_EXPLORE=$ADD_EXPLORE needs EMP_CKPT_DIR (the empowerment reward's estimator)" >&2
        exit 1
    fi
    EXPLORE_REWARD=${EXPLORE_REWARD:-empowerment}
    EMP_FLAG+=(
        --agent.add_explore="$ADD_EXPLORE"
        --agent.bonus_scale="${BONUS_SCALE:-1.0}"
        --agent.explore_reward="$EXPLORE_REWARD"
    )
    if [[ -n "${BONUS_TIME_FRAC:-}" ]]; then
        EMP_FLAG+=(--agent.explore_reward_time_frac="$BONUS_TIME_FRAC")
        TAG="${TAG}_ann${BONUS_TIME_FRAC}"
    fi
    RB_KIND=""
    if [[ "$EXPLORE_REWARD" == "max_episodic_empowerment" ]]; then RB_KIND=max; fi
    case "$ADD_EXPLORE" in
        reward) TAG="${TAG}_rb${RB_KIND}${BONUS_SCALE:-1.0}" ;;
        reward-to-rlpd) TAG="${TAG}_rbrlpd${RB_KIND}${BONUS_SCALE:-1.0}" ;;
        # Distilled trajectory-max empowerment E'(s, a) added straight to the actor loss (EXPLORE_REWARD unused).
        distill) TAG="${TAG}_ed${BONUS_SCALE:-1.0}" ;;&
        distill-to-rlpd) TAG="${TAG}_edrlpd${BONUS_SCALE:-1.0}" ;;&
        distill|distill-to-rlpd)
            if [[ -n "${DISTILL_TARGET:-}" ]]; then
                EMP_FLAG+=(--agent.distill_target="$DISTILL_TARGET")
                if [[ "$DISTILL_TARGET" == "future_max" ]]; then TAG="${TAG}_fut"; fi
            fi ;;
        *) echo "[run_online_crl_seed] unknown ADD_EXPLORE=$ADD_EXPLORE (reward | reward-to-rlpd | distill | distill-to-rlpd)" >&2; exit 1 ;;
    esac
fi

echo "[run_online_crl_seed] env=$ENV_NAME offline=$OFFLINE_DATASET seed=$SEED episode_length=${EPISODE_LENGTH:-<registered>} emp=${EMP_CKPT_DIR:-off} add_explore=${ADD_EXPLORE:-off} tag=$TAG job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

srun "$PYTHON" -u main_online.py \
    --env_name="$ENV_NAME" \
    --seed="$SEED" \
    --agent=agents/online_crl.py \
    --total_env_steps=1000000 \
    "${EP_FLAG[@]}" \
    "${RLPD_FLAG[@]}" \
    "${EMP_FLAG[@]}" \
    --log_interval=5000 \
    --eval_interval=20000 \
    --save_interval=100000 \
    --eval_episodes=20 \
    --video_episodes=0
