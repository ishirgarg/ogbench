#!/usr/bin/env bash
# Single online_crl_skill_controller run on top of a FROZEN **Skill-DT** skill
# checkpoint (agents/skill_dt.py, arXiv:2301.13573), with RLPD, for the
# rnn.ist.berkeley.edu Slurm cluster.
#
# Identical in structure and hyperparameters to run_dds_online_controller_seed.sbatch
# (same K=10 commitment, target_entropy_frac=0.9, 1M env steps, same intervals) --
# the only difference is that SKILL_CKPT points at a `skill_dt` run, so the online
# controller's seeds are directly comparable across the three skill families.
#
# What differs inside for Skill-DT (agents/online_crl_skill_controller.py):
#   * the low level is STATEFUL: the frozen Transformer keeps its K-step context
#     (paper Sec. A.5) across env steps and across the controller's skill switches;
#     a skill is executed by filling the unvisited tail of the future-skill
#     histogram with it over the remaining episode horizon;
#   * RLPD labels come from the frozen VQ encoder: every offline state gets its
#     codebook index, each k-window the most frequent one (--agent.skill_dt_label_mode).
# Unlike DDS, nothing ties K to the checkpoint (the encoder is per-state), so K=10
# is the shared convention rather than a requirement; main_online.py still needs
# episode_horizon % K == 0.
#
# See run_online_crl_seed.sbatch for the cluster-specific notes this mirrors
# (wandb key file, NAS dataset cache, JAX compile cache).
#
# Usage:
#   sbatch --job-name=<name> --output=<log> \
#       scripts/slurm/run_skill_dt_online_controller_seed.sbatch \
#       <SKILL_CKPT> <ENV_NAME> <SEED> [OFFLINE_DATASET] [EPISODE_LENGTH] [SAVE_SUBDIR] [TARGET_ENTROPY_FRAC]
#
# OFFLINE_DATASET is optional; omit (or pass "") to use SKILL_CKPT's own dataset
# (its flags.json env_name), i.e. exactly the data the Skill-DT skills were
# discovered on. Pass "none" to disable RLPD.
# EPISODE_LENGTH is optional; omit to use the env's registered horizon
# (antsoccer-arena-center-online-v0 needs 500).
# SAVE_SUBDIR (default online_controller) and TARGET_ENTROPY_FRAC (default 0.9)
# behave as in the DDS launcher.
#SBATCH --job-name=skill_dt_online_controller
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --qos=default
set -euo pipefail

SKILL_CKPT=${1:?usage: run_skill_dt_online_controller_seed.sbatch SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET] [EPISODE_LENGTH]}
ENV_NAME=${2:?usage: run_skill_dt_online_controller_seed.sbatch SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET] [EPISODE_LENGTH]}
SEED=${3:?usage: run_skill_dt_online_controller_seed.sbatch SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET] [EPISODE_LENGTH]}
OFFLINE_DATASET=${4:-}
EPISODE_LENGTH=${5:-}
# Optional 6th arg: the per-checkpoint subdirectory results are written to. Defaults to
# `online_controller` so every existing caller is unchanged; pass a different name (e.g.
# `online_controller_corner`) to keep a sweep on a different online env in its own tree
# instead of mixing it into the finished one.
SAVE_SUBDIR=${6:-online_controller}
TARGET_ENTROPY_FRAC=${7:-0.9}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_skill_dt_online_controller_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
fi
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp
mkdir -p "$TMPDIR"
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
# RLPD_FRAC_TIME (env var, optional): --rlpd_frac_time, the fraction of total_env_steps during
# which offline data is mixed in; batches are online-only afterwards (main_online.py, default 1.0).
# Only meaningful with RLPD on; the tag gets a "_ft<frac>" suffix so save dirs / logs do not collide.
if [[ -n "${RLPD_FRAC_TIME:-}" && ${#RLPD_FLAG[@]} -gt 0 ]]; then
    RLPD_FLAG+=(--rlpd_frac_time="$RLPD_FRAC_TIME")
    TAG="${TAG}_ft${RLPD_FRAC_TIME}"
fi
# USE_TES (env var, default 0): 1 -> --agent.use_tes=True, TES-SAC target entropy annealing
# (Xu et al. 2021, arXiv:2112.02852; agents/online_crl_skill_controller.py). The target then
# starts at TARGET_ENTROPY_FRAC * log(num_skills) (paper: frac 1.0) and is multiplied by 0.9
# whenever the batch policy entropy stabilises at it. TES_PATIENCE (env var, optional)
# overrides agent.tes_patience, the paper's unspecified T (default 500 gradient steps).
USE_TES=${USE_TES:-0}
TES_FLAG=()
if [[ "$USE_TES" == "1" ]]; then
    TES_FLAG=(--agent.use_tes=True)
    if [[ -n "${TES_PATIENCE:-}" ]]; then TES_FLAG+=(--agent.tes_patience="$TES_PATIENCE"); fi
fi
EP_FLAG=()
if [[ -n "$EPISODE_LENGTH" ]]; then EP_FLAG=(--episode_length="$EPISODE_LENGTH"); fi
SAVE_DIR="$SKILL_CKPT/$SAVE_SUBDIR/$TAG"
mkdir -p "$SAVE_DIR"

echo "[run_skill_dt_online_controller_seed] ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH env=$ENV_NAME offline=$OFFLINE_DATASET seed=$SEED episode_length=${EPISODE_LENGTH:-<registered>} target_entropy_frac=$TARGET_ENTROPY_FRAC use_tes=$USE_TES job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

srun "$PYTHON" -u main_online.py \
    --env_name="$ENV_NAME" \
    --seed="$SEED" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/online_crl_skill_controller.py \
    --agent.skill_checkpoint_path="$SKILL_CKPT" \
    --agent.skill_restore_epoch="$SKILL_EPOCH" \
    --agent.skill_commitment_k=10 \
    --agent.target_entropy_frac="$TARGET_ENTROPY_FRAC" \
    --total_env_steps=1000000 \
    "${EP_FLAG[@]}" \
    "${RLPD_FLAG[@]}" \
    "${TES_FLAG[@]}" \
    --log_interval=5000 \
    --eval_interval=20000 \
    --save_interval=100000 \
    --eval_episodes=20 \
    --video_episodes=0
