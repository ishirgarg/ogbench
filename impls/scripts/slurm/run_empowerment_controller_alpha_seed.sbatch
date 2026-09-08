#!/usr/bin/env bash
# Single OFFLINE empowerment_skill high-level controller run
# (agents/skill_bc_relabel_controller.py:gciql) at ONE AWR alpha, for the
# rnn.ist.berkeley.edu / BAIR "wild west" Slurm cluster (NOT Savio -- no
# --account/--partition, paths under /nas/ucb, GPUs requested generically).
#
# This is the same training stage as the Savio sweep
# scripts/run_skill_bc_relabel_awr_sweep_antsoccer.sh: the frozen empowerment_skill
# checkpoint is loaded read-only, its dataset is relabelled into chunk_horizon=10
# windows with the argmax skill, and only the gciql high level over the K=50 skill
# indices is trained. This script is that script ported to rnn -- one job per
# (checkpoint, alpha) instead of a Savio array task, reading checkpoints from this
# NAS checkout instead of /global/scratch.
#
# Why alpha and not a BC coefficient: the high-level action is a discrete skill
# index, so gciql pins actor_loss='awr' (its ddpgbc branch asserts a continuous
# action space -- agents/gciql.py:93). alpha multiplies the advantage inside exp(),
# so small values flatten every weight to 1 and reduce the actor to plain BC on the
# chunk labels. Same grid {0.3, 1, 3, 10} and same expectile 0.9 as the antsoccer
# Savio sweep, so the numbers are directly comparable across the two clusters and
# against the DDS controller sweep (scripts/slurm/run_dds_controller_alpha_seed.sbatch).
#
# IQL expectile is pinned at 0.9 (gciql's default, restated explicitly so the run's
# flags.json records the intent).
#
# Usage (normally invoked by scripts/slurm/submit_empowerment_controller_alpha_sweep.sh):
#   sbatch --job-name=<name> --output=<log> \
#       scripts/slurm/run_empowerment_controller_alpha_seed.sbatch \
#       <SKILL_CKPT> <ALPHA> [SEED]
#
# Checkpoints go to
#   <SKILL_CKPT>/controller_awr_sweep/alpha<ALPHA>/OGBench/Debug/sd000_s_<jobid>.<ts>/
# so the pretrained params_*.pkl in each SKILL_CKPT is never touched and each
# (checkpoint, alpha) gets its own directory (get_exp_name folds SLURM_JOB_ID in).
#
# See run_online_crl_seed.sbatch for the cluster-specific notes this mirrors (wandb
# key file, NAS dataset cache, JAX compile cache, per-job autotune dir) -- read those
# before submitting if you haven't already.
#SBATCH --job-name=emp_controller_alpha
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --qos=default
set -euo pipefail

SKILL_CKPT=${1:?usage: run_empowerment_controller_alpha_seed.sbatch SKILL_CKPT ALPHA [SEED]}
ALPHA=${2:?usage: run_empowerment_controller_alpha_seed.sbatch SKILL_CKPT ALPHA [SEED]}
SEED=${3:-0}
EXPECTILE=${EXPECTILE:-0.9}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_empowerment_controller_alpha_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
fi
# video_episodes=0, so nothing is rendered; egl is only needed for env construction.
# (MUJOCO_GL=osmesa is the fallback if a node's EGL cannot open a display.)
export MUJOCO_GL=${MUJOCO_GL:-egl}
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp
mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$REPO_DIR/impls/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
# The per-fusion AUTOTUNE sub-cache that XLA puts *inside* the compilation cache dir is
# NOT safe for concurrent writers: with many jobs sharing the NAS cache, a job can read a
# temp .textproto another job has already renamed away and die with
# `JaxRuntimeError: NOT_FOUND: .../xla_gpu_per_fusion_autotune_cache_dir/tmp/...`.
# (Observed on jobs 1194122 / 1194127, 2026-09-04.) Give each job its own autotune dir; the
# module cache above stays shared, which is where the real cold-start saving is.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${SLURM_JOB_ID:-$$}"
# A whole, exclusive GPU per job, but the cluster mixes A4000/A6000/A100 -- grow memory on
# demand instead of preallocating a fraction sized for one particular card. (The peak is
# the relabelling pass over the whole dataset, not training.)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# OGBench's dataset cache defaults to ~/.ogbench/data, which on a Slurm job resolves to the
# COMPUTE NODE's own unsynced home; a miss there triggers a download, and compute nodes may
# have no internet egress. Point at the NAS-resident cache every job shares.
export OGBENCH_DATASET_DIR=/nas/ucb/ishirgarg/.ogbench/data

cd "$IMPLS_DIR"

ENV_NAME=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
SKILL_EPOCH=$($PYTHON -c "
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl\$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
" "$SKILL_CKPT")
if [[ -z "$ENV_NAME" || -z "$SKILL_EPOCH" ]]; then
    echo "could not read env_name / latest epoch from $SKILL_CKPT" >&2; exit 1
fi

SAVE_DIR="$SKILL_CKPT/controller_awr_sweep/alpha${ALPHA}"
mkdir -p "$SAVE_DIR"

echo "[run_empowerment_controller_alpha_seed] ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH env=$ENV_NAME alpha=$ALPHA expectile=$EXPECTILE seed=$SEED job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

# `--agent.base.actor_loss` is deliberately not passed: _base_config pins it to 'awr'
# because the option MDP's action space is discrete. agent.chunk_horizon is left at its
# default of 10 (it drives SequenceDataset's sequence_length through a shared
# FieldReference), matching the Savio sweep and the DDS controller's chunk_horizon.
srun "$PYTHON" -u main.py \
    --env_name="$ENV_NAME" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/skill_bc_relabel_controller.py:gciql \
    --agent.skill_checkpoint_path="$SKILL_CKPT" \
    --agent.skill_restore_epoch="$SKILL_EPOCH" \
    --agent.base.expectile="$EXPECTILE" \
    --agent.base.alpha="$ALPHA" \
    --seed="$SEED" \
    --train_steps=1000000 \
    --log_interval=5000 \
    --eval_interval=100000 \
    --save_interval=100000 \
    --eval_episodes=50 \
    --video_episodes=0
