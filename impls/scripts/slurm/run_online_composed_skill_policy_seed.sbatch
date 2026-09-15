#!/usr/bin/env bash
# Single agents/online_composed_skill_policy.py training run (one seed) for the
# rnn.ist.berkeley.edu Slurm cluster. See run_online_crl_seed.sbatch for the
# cluster-specific notes (wandb key file, NAS dataset cache, per-job autotune dir)
# this mirrors -- read those before submitting if you haven't already.
#
# Unlike every *_controller run, the low level is NOT frozen: pi_hi(k|s,g) and
# pi_lo(a|s,z_k) are trained jointly as one flat policy
# pi(a|s,g) = sum_k pi_hi(k|s,g) pi_lo(a|s,z_k), with a separate (smaller) learning
# rate for the low level. There is no skill horizon -- the skill is redrawn every env
# step -- so rows are env steps (rollout_type='flat') and RLPD needs no window
# labelling. LOW_LR=0 reproduces the frozen-low-level baseline at k=1 exactly and is
# the ablation this run is meant to beat.
#
# Usage (normally via scripts/slurm/submit_composed_online_lowlr_sweep.sh):
#   sbatch --job-name=<name> --output=<log> \
#       scripts/slurm/run_online_composed_skill_policy_seed.sbatch \
#       <SKILL_CKPT> <ENV_NAME> <SEED> [OFFLINE_DATASET] [EPISODE_LENGTH] [SAVE_SUBDIR] [TARGET_ENTROPY_FRAC] [LOW_LR]
#
# OFFLINE_DATASET is optional; omit (or pass "") to use SKILL_CKPT's own dataset (its
# flags.json env_name), matching every other online launcher here. Pass "none" to
# disable RLPD. EPISODE_LENGTH is optional; omit to use the env's registered horizon
# (there is no divisibility constraint here -- that one belongs to the macro rollout).
# TARGET_ENTROPY_FRAC (7th) sets H_target = frac * log(num_skills) for the categorical
# high level; it is the ONLY entropy term (see the agent's docstring). LOW_LR (8th) is
# the low level's own Adam learning rate and is part of the save path, so a sweep over
# it never collides.
#
# Checkpoints go to <SKILL_CKPT>/<SAVE_SUBDIR>/<rlpd|norlpd>_lowlr<LOW_LR>/OGBench/Debug/sd<seed>_<ts>/,
# so the pretrained params_*.pkl in SKILL_CKPT itself is never touched.
#SBATCH --job-name=online_composed_skill_policy
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --qos=default
set -euo pipefail

USAGE="usage: run_online_composed_skill_policy_seed.sbatch SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET] [EPISODE_LENGTH] [SAVE_SUBDIR] [TARGET_ENTROPY_FRAC] [LOW_LR]"
SKILL_CKPT=${1:?$USAGE}
ENV_NAME=${2:?$USAGE}
SEED=${3:?$USAGE}
OFFLINE_DATASET=${4:-}
EPISODE_LENGTH=${5:-}
SAVE_SUBDIR=${6:-online_composed}
TARGET_ENTROPY_FRAC=${7:-0.5}
LOW_LR=${8:-3e-5}

TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
# ACTOR_BATCH_SIZE (env var, optional): rows the K-term actor loss runs on (unset -> the whole
# batch, i.e. the exact loss). Each row costs K low-level forward passes and K critic
# evaluations, so this is the knob to turn if a cell is wall-clock bound; the critic loss
# always sees the full batch either way.
ACTOR_BATCH_SIZE=${ACTOR_BATCH_SIZE:-}
# UTD_RATIO (env var, optional): gradient steps per collected row. Rows here are ENV STEPS
# (this agent is flat), so the default 1 means 1M updates over 1M env steps -- ten times the
# gradient budget a k=10 macro controller gets on the same env steps. Set 0.1 to match that
# controller's budget instead (main_online.py accepts a fractional utd_ratio).
UTD_RATIO=${UTD_RATIO:-}
# GRAD_METHOD (env var, default 'enumerate'): how the Q signal reaches the high level.
#   enumerate -- exact sum over all K skills (zero variance, K low-level + K critic passes/row)
#   reinforce -- one sampled skill; the low level takes its pathwise gradient, the high level a
#                score-function gradient rewarded by the low level's loss (1 pass/row, variance)
# REINFORCE_BASELINE (env var, 'batch' | 'none') applies only to the latter.
# A non-default method is appended to the save-path tag so the two never collide.
GRAD_METHOD=${GRAD_METHOD:-enumerate}
REINFORCE_BASELINE=${REINFORCE_BASELINE:-}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_online_composed_skill_policy_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
fi
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp
mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$REPO_DIR/impls/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
# Per-job autotune dir: the shared one is not safe for concurrent writers (see
# run_online_crl_skill_controller_seed.sbatch for the failure this avoids).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${SLURM_JOB_ID:-$$}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OGBENCH_DATASET_DIR=/nas/ucb/ishirgarg/.ogbench/data

cd "$IMPLS_DIR"

# Resolve the pretrained epoch here and pass it explicitly so flags.json records it
# (main_online.py writes flags.json before constructing the agent).
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
if [[ -n "${RLPD_FRAC_TIME:-}" && ${#RLPD_FLAG[@]} -gt 0 ]]; then
    RLPD_FLAG+=(--rlpd_frac_time="$RLPD_FRAC_TIME")
    TAG="${TAG}_ft${RLPD_FRAC_TIME}"
fi
EP_FLAG=()
if [[ -n "$EPISODE_LENGTH" ]]; then EP_FLAG=(--episode_length="$EPISODE_LENGTH"); fi
AB_FLAG=()
if [[ -n "$ACTOR_BATCH_SIZE" ]]; then AB_FLAG=(--agent.actor_batch_size="$ACTOR_BATCH_SIZE"); fi
UTD_FLAG=()
if [[ -n "$UTD_RATIO" ]]; then UTD_FLAG=(--agent.utd_ratio="$UTD_RATIO"); fi
METHOD_FLAG=(--agent.composed_grad_method="$GRAD_METHOD")
METHOD_TAG=""
if [[ "$GRAD_METHOD" != "enumerate" ]]; then METHOD_TAG="_$GRAD_METHOD"; fi
if [[ -n "$REINFORCE_BASELINE" ]]; then
    METHOD_FLAG+=(--agent.reinforce_baseline="$REINFORCE_BASELINE")
    if [[ "$GRAD_METHOD" == "reinforce" && "$REINFORCE_BASELINE" != "batch" ]]; then
        METHOD_TAG="${METHOD_TAG}_b${REINFORCE_BASELINE}"
    fi
fi

# The low-level learning rate is the swept axis, so it is part of the save path.
SAVE_DIR="$SKILL_CKPT/$SAVE_SUBDIR/${TAG}_lowlr${LOW_LR}${METHOD_TAG}"
mkdir -p "$SAVE_DIR"

echo "[run_online_composed_skill_policy_seed] ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH env=$ENV_NAME offline=$OFFLINE_DATASET seed=$SEED episode_length=${EPISODE_LENGTH:-<registered>} target_entropy_frac=$TARGET_ENTROPY_FRAC low_lr=$LOW_LR actor_batch_size=${ACTOR_BATCH_SIZE:-<full batch>} utd_ratio=${UTD_RATIO:-<config default>} grad_method=$GRAD_METHOD job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

srun "$PYTHON" -u main_online.py \
    --env_name="$ENV_NAME" \
    --seed="$SEED" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/online_composed_skill_policy.py \
    --agent.skill_checkpoint_path="$SKILL_CKPT" \
    --agent.skill_restore_epoch="$SKILL_EPOCH" \
    --agent.low_lr="$LOW_LR" \
    --agent.target_entropy_frac="$TARGET_ENTROPY_FRAC" \
    "${AB_FLAG[@]}" \
    "${UTD_FLAG[@]}" \
    "${METHOD_FLAG[@]}" \
    --total_env_steps="$TOTAL_ENV_STEPS" \
    "${EP_FLAG[@]}" \
    "${RLPD_FLAG[@]}" \
    --log_interval=5000 \
    --eval_interval=20000 \
    --save_interval=100000 \
    --eval_episodes=20 \
    --video_episodes=0
