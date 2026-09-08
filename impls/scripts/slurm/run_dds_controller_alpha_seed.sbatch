#!/usr/bin/env bash
# Single OFFLINE DDS high-level controller run (agents/dds_controller.py:gciql) at ONE
# AWR alpha, for the rnn.ist.berkeley.edu / BAIR "wild west" Slurm cluster (NOT Savio --
# no --account/--partition, paths under /nas/ucb, GPUs requested generically).
#
# This is the same training stage as scripts/run_dds_controller_final5_local.sh (the DDS
# paper's "relabel with the encoder, then IQL + AWR over the codebook", arXiv:2503.20176
# Sec. 4.2-4.4): the frozen dds checkpoint is loaded read-only (encoder / codebook /
# diffusion decoder only) and only the gciql high level over the K=50 codes is trained.
# The ONLY departure from that script is that --agent.base.alpha is set explicitly instead
# of being left at agents/dds_controller.py's paper default of 3.0, so the AWR inverse
# temperature can be swept; every other flag (1M steps, intervals, 50 eval episodes,
# expectile 0.7 / lr 1e-4 / tau 0.005 from _PAPER_IQL_CONFIG, chunk_horizon 10) is
# unchanged, and no --run_group is passed.
#
# Why alpha and not a BC coefficient: the high-level action is a discrete codebook index,
# so gciql pins actor_loss='awr' (its ddpgbc branch asserts a continuous action space).
# alpha multiplies the advantage inside exp(), so small values flatten every weight to 1
# and reduce the actor to plain BC on the chunk labels. Same reasoning and same grid as
# the empowerment_skill controller sweep, scripts/run_skill_bc_relabel_awr_sweep_antsoccer.sh.
#
# Usage (normally invoked by scripts/slurm/submit_dds_controller_alpha_sweep.sh):
#   sbatch --job-name=<name> --output=<log> \
#       scripts/slurm/run_dds_controller_alpha_seed.sbatch \
#       <SKILL_CKPT> <ALPHA> [SEED]
#
# Checkpoints go to
#   <SKILL_CKPT>/controller_awr_sweep/alpha<ALPHA>/OGBench/Debug/sd000_s_<jobid>.<ts>/
# -- a sibling of the alpha=3.0 run's <SKILL_CKPT>/controller/, so neither the pretrained
# params_*.pkl nor the existing alpha=3.0 controller is ever touched.
#
# Runtime: the five local alpha=3.0 runs took 3.0-4.0 h wall clock with TWO runs sharing
# one A6000 (2026-09-05); a whole GPU per job here, so --time=12:00:00 is a wide margin.
#
# See run_online_crl_seed.sbatch for the cluster-specific notes this mirrors (wandb key
# file, NAS dataset cache, JAX compile cache, per-job autotune dir) -- read those before
# submitting if you haven't already.
#SBATCH --job-name=dds_controller_alpha
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --qos=default
# ppo/vae are the cluster's A4000 nodes (16 GB). The prepare_datasets relabelling pass
# peaks at ~17 GB, so a job landing there dies with
# `JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 5.73GiB`
# minutes in -- that is exactly how jobs 1194500 / 1194501 (pointmaze-teleport-navigate,
# alpha 0.3 and 1) failed on 2026-09-05, while the same sweep's alpha=10 run completed on
# an A100 node. Keep those two nodes out; A6000 (48 GB) and A100 (80 GB) both fit.
#SBATCH --exclude=ppo.ist.berkeley.edu,vae.ist.berkeley.edu
set -euo pipefail

SKILL_CKPT=${1:?usage: run_dds_controller_alpha_seed.sbatch SKILL_CKPT ALPHA [SEED]}
ALPHA=${2:?usage: run_dds_controller_alpha_seed.sbatch SKILL_CKPT ALPHA [SEED]}
SEED=${3:-0}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_dds_controller_alpha_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
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
# the prepare_datasets relabelling pass, ~17 GB measured locally, not training.)
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

echo "[run_dds_controller_alpha_seed] ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH env=$ENV_NAME alpha=$ALPHA seed=$SEED job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

# agent.chunk_horizon defaults to 10 and dds_controller.create asserts it equals the
# checkpoint's sequence_length (10 for all five), so it is left at the default.
srun "$PYTHON" -u main.py \
    --env_name="$ENV_NAME" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/dds_controller.py:gciql \
    --agent.skill_checkpoint_path="$SKILL_CKPT" \
    --agent.skill_restore_epoch="$SKILL_EPOCH" \
    --agent.base.alpha="$ALPHA" \
    --seed="$SEED" \
    --train_steps=1000000 \
    --log_interval=5000 \
    --eval_interval=100000 \
    --save_interval=100000 \
    --eval_episodes=50 \
    --video_episodes=0
