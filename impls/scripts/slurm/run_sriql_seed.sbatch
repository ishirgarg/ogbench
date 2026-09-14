#!/usr/bin/env bash
# E1 for SR-IQL (impls/docs/grounded_option_qi_proposal.md, v3): train one
# agents/skill_restricted_iql.py high level over the frozen K=50 skills of an
# empowerment_skill checkpoint, fully offline, at one (kernel, coverage_alpha, expectile)
# cell. Saved INSIDE the skill checkpoint's run folder; the frozen params_*.pkl is never
# touched. Training-time eval re-picks the skill every step (skill_horizon=1, the choice
# consistent with the value function). Afterwards the final checkpoint is re-scored
# (eval_sriql.py) at skill_horizon=10 and with the off-support selector 'qmu'
# (Q(s, mu_z(s), g) instead of the in-sample Qz) -- ablation E2(ii).
#
# rnn.ist.berkeley.edu cluster conventions as in run_empowerment_controller_alpha_seed.sbatch.
#
# Usage (normally via scripts/slurm/submit_sriql_e0_e1.sh):
#   sbatch --job-name=<name> --output=<log> scripts/slurm/run_sriql_seed.sbatch \
#       <SKILL_CKPT> <KERNEL gauss|hard> <ALPHA> <EXPECTILE> [SEED]
# Checkpoints go to <SKILL_CKPT>/sriql_sweep/<KERNEL>_a<ALPHA>_e<EXPECTILE>/OGBench/Debug/sd000_s_<jobid>.<ts>/
#SBATCH --job-name=sriql
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --qos=default
set -euo pipefail

SKILL_CKPT=${1:?usage: run_sriql_seed.sbatch SKILL_CKPT KERNEL ALPHA EXPECTILE [SEED]}
KERNEL=${2:?usage: run_sriql_seed.sbatch SKILL_CKPT KERNEL ALPHA EXPECTILE [SEED]}
ALPHA=${3:?usage: run_sriql_seed.sbatch SKILL_CKPT KERNEL ALPHA EXPECTILE [SEED]}
EXPECTILE=${4:?usage: run_sriql_seed.sbatch SKILL_CKPT KERNEL ALPHA EXPECTILE [SEED]}
SEED=${5:-0}
# Optional fixed kernel width (default: data-estimated sqrt(E_D[min_z ||a-mu_z||^2]), which
# measured ESS ~40/50 on both envs, i.e. near-uniform -- 0.2 gives ESS ~6-10).
KERNEL_SIGMA=${KERNEL_SIGMA:-}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_sriql_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
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
SKILL_EPOCH=$($PYTHON -c "
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl\$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
" "$SKILL_CKPT")
if [[ -z "$ENV_NAME" || -z "$SKILL_EPOCH" ]]; then
    echo "could not read env_name / latest epoch from $SKILL_CKPT" >&2; exit 1
fi

KTAG=$KERNEL; SIGMA_FLAG=()
if [[ -n "$KERNEL_SIGMA" ]]; then KTAG="${KERNEL}${KERNEL_SIGMA}"; SIGMA_FLAG=(--agent.kernel_sigma="$KERNEL_SIGMA"); fi
SAVE_DIR="$SKILL_CKPT/sriql_sweep/${KTAG}_a${ALPHA}_e${EXPECTILE}"
mkdir -p "$SAVE_DIR"

echo "[run_sriql_seed] ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH env=$ENV_NAME kernel=$KERNEL sigma=${KERNEL_SIGMA:-data} alpha=$ALPHA expectile=$EXPECTILE seed=$SEED job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

srun "$PYTHON" -u main.py \
    --env_name="$ENV_NAME" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/skill_restricted_iql.py \
    --agent.skill_checkpoint_path="$SKILL_CKPT" \
    --agent.skill_restore_epoch="$SKILL_EPOCH" \
    --agent.kernel="$KERNEL" "${SIGMA_FLAG[@]}" \
    --agent.coverage_alpha="$ALPHA" \
    --agent.expectile="$EXPECTILE" \
    --agent.skill_horizon=1 \
    --seed="$SEED" \
    --train_steps=1000000 \
    --log_interval=5000 \
    --eval_interval=100000 \
    --save_interval=100000 \
    --eval_episodes=50 \
    --video_episodes=0

RUN_DIR=$(ls -d "$SAVE_DIR"/OGBench/Debug/sd$(printf '%03d' "$SEED")_s_${SLURM_JOB_ID}.*/ | head -n 1)
RUN_DIR=${RUN_DIR%/}
[[ -f "$RUN_DIR/params_1000000.pkl" ]] || { echo "no final checkpoint in $RUN_DIR" >&2; exit 1; }

# Post-hoc: held skill (H=10) and the off-support selector ablation, on the final checkpoint.
srun "$PYTHON" -u eval_sriql.py --run_dir="$RUN_DIR" --skill_horizon=10 --eval_episodes=50 --seed="$SEED"
srun "$PYTHON" -u eval_sriql.py --run_dir="$RUN_DIR" --selector=qmu --skill_horizon=1 --eval_episodes=50 --seed="$SEED"
