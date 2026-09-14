#!/usr/bin/env bash
# E0 for SR-IQL (impls/docs/grounded_option_qi_proposal.md, v3): train a FLAT goal-conditioned
# gciql baseline on the dataset of an empowerment_skill checkpoint, saving it INSIDE that
# checkpoint's run folder, then score the zero-training skill selector
#     z* = argmax_z Q_gciql(s, mu_z(s), g)
# (eval_skill_qsel.py) at skill_horizon 1 and 10 using the same checkpoint's frozen skills.
# The flat gciql number itself (evaluation/overall_success in the run's eval.csv) is the
# upper reference every skill selector is compared against.
#
# rnn.ist.berkeley.edu cluster conventions as in run_empowerment_controller_alpha_seed.sbatch.
#
# Usage (normally via scripts/slurm/submit_sriql_e0_e1.sh):
#   sbatch --job-name=<name> --output=<log> scripts/slurm/run_gciql_flat_seed.sbatch <SKILL_CKPT> [SEED]
# Checkpoints go to <SKILL_CKPT>/gciql_flat/OGBench/Debug/sd000_s_<jobid>.<ts>/
#SBATCH --job-name=gciql_flat
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --qos=default
set -euo pipefail

SKILL_CKPT=${1:?usage: run_gciql_flat_seed.sbatch SKILL_CKPT [SEED]}
SEED=${2:-0}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_gciql_flat_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
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

SAVE_DIR="$SKILL_CKPT/gciql_flat"
mkdir -p "$SAVE_DIR"

echo "[run_gciql_flat_seed] skill_ckpt=$SKILL_CKPT env=$ENV_NAME seed=$SEED job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

srun "$PYTHON" -u main.py \
    --env_name="$ENV_NAME" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/gciql.py \
    --seed="$SEED" \
    --train_steps=1000000 \
    --log_interval=5000 \
    --eval_interval=100000 \
    --save_interval=100000 \
    --eval_episodes=50 \
    --video_episodes=0

# The run dir main.py created for THIS job (exp name folds SLURM_JOB_ID in).
RUN_DIR=$(ls -d "$SAVE_DIR"/OGBench/Debug/sd$(printf '%03d' "$SEED")_s_${SLURM_JOB_ID}.*/ | head -n 1)
RUN_DIR=${RUN_DIR%/}
[[ -f "$RUN_DIR/params_1000000.pkl" ]] || { echo "no final checkpoint in $RUN_DIR" >&2; exit 1; }

# E0: the Q-selector over the frozen skills, at both horizons (CPU eval, ~minutes each).
for H in 1 10; do
    srun "$PYTHON" -u eval_skill_qsel.py \
        --gciql_dir="$RUN_DIR" --skill_dir="$SKILL_CKPT" \
        --skill_horizon="$H" --eval_episodes=50 --seed="$SEED"
done
