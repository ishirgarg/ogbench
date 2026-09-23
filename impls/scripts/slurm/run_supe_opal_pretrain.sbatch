#!/usr/bin/env bash
# SUPE stage 1, run ONCE per dataset: the continuous OPAL VAE (agents/opal.py, latent_type=continuous)
# that every SUPE online run on that dataset's envs then loads FROZEN (agents/supe.py,
# --agent.skill_checkpoint_path). This is SUPE's `run_opal.py` recipe on our main.py:
#   skill_dim=8, kl_coef 0.1 (0.2 on cube, the paper's --config.kl_coef=0.2), chunk_size = the
#   high-level horizon H (SUPE hpolicy_horizon = horizon_length = 4), batch 256, 1M steps,
#   OGBench widths (512, 512, 512) / 512 -- all agents/opal.py defaults except chunk and batch.
# rnn.ist.berkeley.edu Slurm conventions (see run_online_crl_seed.sbatch for the why of each export).
#
# Usage (normally via scripts/slurm/submit_supe_opal_pretrain.sh):
#   sbatch --job-name=<name> --output=<log> scripts/slurm/run_supe_opal_pretrain.sbatch \
#       <DATASET> [SEED=0] [CHUNK=4] [KL_COEF=0.1]
# Env vars: CKPT_ROOT (default /nas/ucb/ishirgarg/ogbench/impls/ckpts/final/supe_opal; the run lands in
#           $CKPT_ROOT/<DATASET>/OGBench/Debug/sd<seed>_...), REPO_DIR (this worktree), TRAIN_STEPS (1000000).
#SBATCH --job-name=supe_opal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --qos=default
set -euo pipefail

DATASET=${1:?usage: run_supe_opal_pretrain.sbatch DATASET [SEED] [CHUNK] [KL_COEF]}
SEED=${2:-0}
CHUNK=${3:-4}
KL_COEF=${4:-0.1}

REPO_DIR=${REPO_DIR:-/nas/ucb/ishirgarg/ogbench/.claude/worktrees/supe-baseline-integration-4d17b9}
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
CKPT_ROOT=${CKPT_ROOT:-/nas/ucb/ishirgarg/ogbench/impls/ckpts/final/supe_opal}
TRAIN_STEPS=${TRAIN_STEPS:-1000000}

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=${WANDB_KEY_FILE:-/nas/ucb/ishirgarg/.wandb_api_key}
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[supe_opal] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
fi
export TMPDIR=${TMPDIR:-/nas/ttl=60d/ishirgarg/tmp}
mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/nas/ucb/ishirgarg/ogbench/impls/.jax_cache}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${SLURM_JOB_ID:-$$}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}

cd "$IMPLS_DIR"
[[ -f "$OGBENCH_DATASET_DIR/$DATASET.npz" ]] || { echo "FATAL: $OGBENCH_DATASET_DIR/$DATASET.npz missing (compute nodes have no egress)" >&2; exit 1; }
SAVE_DIR="$CKPT_ROOT/$DATASET"
mkdir -p "$SAVE_DIR"

echo "[supe_opal] dataset=$DATASET seed=$SEED chunk=$CHUNK kl_coef=$KL_COEF steps=$TRAIN_STEPS save_dir=$SAVE_DIR job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

# Eval here only rolls out prior-sampled skills (agents/opal.py: no high level), so keep it cheap.
srun "$PYTHON" -u main.py \
    --env_name="$DATASET" \
    --seed="$SEED" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/opal.py \
    --agent.latent_type=continuous \
    --agent.skill_dim=8 \
    --agent.kl_coef="$KL_COEF" \
    --agent.chunk_size="$CHUNK" \
    --agent.sequence_length="$CHUNK" \
    --agent.batch_size=256 \
    --train_steps="$TRAIN_STEPS" \
    --log_interval=5000 \
    --eval_interval=250000 \
    --eval_episodes=5 \
    --save_interval=100000 \
    --video_episodes=0
