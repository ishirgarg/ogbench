#!/usr/bin/env bash
# Offline OFFLINE-RL training of the COMPOSED policy (agents/composed_skill_policy.py) on top of
# a pretrained skill checkpoint (empowerment_skill / opal-discrete / dds).
#
# Unlike every *_controller run, the low level is NOT frozen: pi_hi(k|s,g) and pi_lo(a|s,z_k) are
# trained jointly as one flat policy pi(a|s,g) = sum_k pi_hi(k|s,g) pi_lo(a|s,z_k), with a separate
# (smaller) learning rate for the low level. There is no skill horizon: the skill is redrawn every
# environment step. LOW_LR=0 reproduces the frozen-low-level baseline exactly and is the ablation
# this run is meant to beat.
#
# rnn.ist.berkeley.edu cluster conventions as in run_gciql_flat_seed.sbatch.
#
# Usage (normally via scripts/slurm/submit_composed_skill_policy_seeds.sh):
#   sbatch --job-name=<name> --output=<log> run_composed_skill_policy_seed.sbatch <SKILL_CKPT> [SEED]
# Env-var knobs: LOW_LR, ALPHA, COMPOSED_ACTOR_LOSS, BASE_AGENT, SAVE_SUBDIR, TRAIN_STEPS,
#                COMPOSED_BATCH_SIZE, DIFFUSION_LOGPROB_SCALE
# Checkpoints go to <SKILL_CKPT>/<SAVE_SUBDIR>/OGBench/Debug/sd<seed>_s_<jobid>.<ts>/
#SBATCH --job-name=composed_skill
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --qos=default
set -euo pipefail

SKILL_CKPT=${1:?usage: run_composed_skill_policy_seed.sbatch SKILL_CKPT [SEED]}
SEED=${2:-0}

LOW_LR=${LOW_LR:-3e-5}
ALPHA=${ALPHA:-3.0}
COMPOSED_ACTOR_LOSS=${COMPOSED_ACTOR_LOSS:-awr}
BASE_AGENT=${BASE_AGENT:-gciql}
TRAIN_STEPS=${TRAIN_STEPS:-1000000}
SAVE_SUBDIR=${SAVE_SUBDIR:-composed_skill_policy}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_composed_skill_policy_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
fi
export MUJOCO_GL=${MUJOCO_GL:-egl}
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp
mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$REPO_DIR/impls/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${SLURM_JOB_ID:-$$}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OGBENCH_DATASET_DIR=/nas/ucb/ishirgarg/.ogbench/data

cd "$IMPLS_DIR"

ENV_NAME=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
[[ -n "$ENV_NAME" ]] || { echo "could not read env_name from $SKILL_CKPT" >&2; exit 1; }
SKILL_AGENT=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['agent']['agent_name'])" "$SKILL_CKPT")

SAVE_DIR="$SKILL_CKPT/$SAVE_SUBDIR"
mkdir -p "$SAVE_DIR"

EXTRA=()
# DDS's continuous decoder is a diffusion epsilon-network with no closed-form density, so the
# mixture log-likelihood uses the scaled negative-ELBO surrogate. `create` selects this
# automatically; the scale is the knob that sets how peaked the skill responsibilities are.
if [[ "$SKILL_AGENT" == "dds" ]]; then
    EXTRA+=(--agent.diffusion_logprob_scale="${DIFFUSION_LOGPROB_SCALE:-1.0}")
fi
# Each row of the composed actor loss costs K low-level forward passes; subsample when K is large.
if [[ -n "${COMPOSED_BATCH_SIZE:-}" ]]; then
    EXTRA+=(--agent.composed_batch_size="$COMPOSED_BATCH_SIZE")
fi

echo "[run_composed_skill_policy_seed] skill_ckpt=$SKILL_CKPT low=$SKILL_AGENT env=$ENV_NAME seed=$SEED" \
     "base=$BASE_AGENT loss=$COMPOSED_ACTOR_LOSS low_lr=$LOW_LR alpha=$ALPHA" \
     "job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

srun "$PYTHON" -u main.py \
    --env_name="$ENV_NAME" \
    --save_dir="$SAVE_DIR" \
    --agent="agents/composed_skill_policy.py:$BASE_AGENT" \
    --agent.skill_checkpoint_path="$SKILL_CKPT" \
    --agent.low_lr="$LOW_LR" \
    --agent.alpha="$ALPHA" \
    --agent.composed_actor_loss="$COMPOSED_ACTOR_LOSS" \
    "${EXTRA[@]}" \
    --seed="$SEED" \
    --train_steps="$TRAIN_STEPS" \
    --log_interval=5000 \
    --eval_interval=100000 \
    --save_interval=100000 \
    --eval_episodes=50 \
    --video_episodes=0
