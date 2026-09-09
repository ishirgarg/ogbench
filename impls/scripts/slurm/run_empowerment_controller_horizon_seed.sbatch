#!/usr/bin/env bash
# Single OFFLINE empowerment_skill high-level controller run
# (agents/skill_bc_relabel_controller.py:gciql) at ONE (chunk_horizon, AWR alpha) pair,
# for the rnn.ist.berkeley.edu / BAIR "wild west" Slurm cluster.
#
# Identical to run_empowerment_controller_alpha_seed.sbatch except that H is swept
# instead of pinned at 10. H is ONE number with three roles, all moved by
# --agent.chunk_horizon:
#   * the window each chunk label is fitted over
#     (z*(t) = argmax_z sum_{i<H} log pi(a_{t+i}|s_{t+i},z), utils/datasets.py:527),
#   * the option length of the semi-MDP the high level is trained on
#     (s_t, z*(t)) -> s_{t+H}, and
#   * the number of env steps the frozen low-level policy is executed for before the
#     high level re-selects (`skill_horizon`, which defaults to chunk_horizon).
# `sequence_length` is linked to `chunk_horizon` through a shared FieldReference in
# get_config, and `create` re-asserts the equality, so passing chunk_horizon alone is
# both necessary and sufficient. Do NOT also pass --agent.skill_horizon: setting it
# apart from chunk_horizon deliberately introduces the train/execute mismatch this
# sweep exists to remove.
#
# Why sweep H at all: plot_relabel_diagnosis.py measured, on
# antsoccer-arena-navigate at H=10, that (a) executing the assigned skill reproduces
# its window no better than a randomly chosen skill (0.307 m vs 0.338 m endpoint
# error, oracle 0.126 m) and (b) executing the assigned skill covers 4-6x less ground
# than the data window credited to it (0.10-0.21 m vs 0.71-1.21 m per option). Both
# are H-dependent: 10 ant steps is ~0.5 s, inside which inertia dominates whichever z
# was commanded, so the K=50 skills are neither distinguishable in action space nor
# able to move the ant. A longer H gives the skill room to express itself (better
# labels, more progress per option) at the cost of a coarser high level and shorter
# windows surviving near trajectory ends; a shorter H does the reverse. That trade-off
# is what the {5, 10, 15, 20, 50} grid measures.
#
# Usage (normally invoked by
# scripts/slurm/submit_empowerment_controller_horizon_sweep.sh):
#   sbatch --job-name=<name> --output=<log> \
#       scripts/slurm/run_empowerment_controller_horizon_seed.sbatch \
#       <SKILL_CKPT> <HORIZON> <ALPHA> [SEED]
#
# Checkpoints go to
#   <SKILL_CKPT>/controller_horizon_sweep/h<H>_alpha<ALPHA>/OGBench/Debug/sd000_s_<jobid>.<ts>/
# i.e. INSIDE the frozen run's own folder under ckpts/final/empowerment_final/<env>/,
# alongside the existing controller_awr_sweep/ (which holds H=10 at the same four
# alphas -- that arm is not re-run here). The pretrained params_*.pkl in SKILL_CKPT is
# never touched, and get_exp_name folds SLURM_JOB_ID in so each job gets its own dir.
#
# See run_online_crl_seed.sbatch for the cluster-specific notes this mirrors (wandb
# key file, NAS dataset cache, JAX compile cache, per-job autotune dir).
#SBATCH --job-name=emp_controller_horizon
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=1
# 24h rather than the alpha sweep's 16h: at H=50 every batch carries a 50-step window
# (SequenceDataset emits [B, H, obs] / [B, H, act]), so the per-step cost is well above
# the H=10 runs these limits were set from.
#SBATCH --time=24:00:00
#SBATCH --qos=default
set -euo pipefail

SKILL_CKPT=${1:?usage: run_empowerment_controller_horizon_seed.sbatch SKILL_CKPT HORIZON ALPHA [SEED]}
HORIZON=${2:?usage: run_empowerment_controller_horizon_seed.sbatch SKILL_CKPT HORIZON ALPHA [SEED]}
ALPHA=${3:?usage: run_empowerment_controller_horizon_seed.sbatch SKILL_CKPT HORIZON ALPHA [SEED]}
SEED=${4:-0}
EXPECTILE=${EXPECTILE:-0.9}

REPO_DIR=/nas/ucb/ishirgarg/ogbench
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[run_empowerment_controller_horizon_seed] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
fi
# video_episodes=0, so nothing is rendered; egl is only needed for env construction.
# (MUJOCO_GL=osmesa is the fallback if a node's EGL cannot open a display.)
export MUJOCO_GL=${MUJOCO_GL:-egl}
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp
mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$REPO_DIR/impls/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
# The per-fusion AUTOTUNE sub-cache XLA puts inside the compilation cache dir is NOT safe
# for concurrent writers; give each job its own. (Same rationale as the alpha sweep.)
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${SLURM_JOB_ID:-$$}"
# The cluster mixes A4000/A6000/A100 -- grow memory on demand instead of preallocating a
# fraction sized for one card. (The peak is the relabelling pass, not training.)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Compute nodes have no internet egress; point at the NAS-resident dataset cache.
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

SAVE_DIR="$SKILL_CKPT/controller_horizon_sweep/h${HORIZON}_alpha${ALPHA}"
mkdir -p "$SAVE_DIR"

echo "[run_empowerment_controller_horizon_seed] ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH env=$ENV_NAME horizon=$HORIZON alpha=$ALPHA expectile=$EXPECTILE seed=$SEED job=${SLURM_JOB_ID:-<none>} node=$(hostname)"
echo "[run_empowerment_controller_horizon_seed] save_dir=$SAVE_DIR"

# `--agent.base.actor_loss` is deliberately not passed: _base_config pins it to 'awr'
# because the option MDP's action space is discrete. `--agent.chunk_horizon` moves
# `sequence_length` (shared FieldReference) and `skill_horizon` (None -> chunk_horizon)
# with it, so labelling, training and execution all use the same H.
srun "$PYTHON" -u main.py \
    --env_name="$ENV_NAME" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/skill_bc_relabel_controller.py:gciql \
    --agent.skill_checkpoint_path="$SKILL_CKPT" \
    --agent.skill_restore_epoch="$SKILL_EPOCH" \
    --agent.chunk_horizon="$HORIZON" \
    --agent.base.expectile="$EXPECTILE" \
    --agent.base.alpha="$ALPHA" \
    --seed="$SEED" \
    --train_steps=1000000 \
    --log_interval=5000 \
    --eval_interval=100000 \
    --save_interval=100000 \
    --eval_episodes=50 \
    --video_episodes=0
