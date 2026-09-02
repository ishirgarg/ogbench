#!/bin/bash
#SBATCH --job-name=skill_bc_relabel_awr_antsoccer
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-7

# Normal-priority (rail_gpu4_normal) AWR-alpha sweep for the goal-conditioned skill
# controller (skill_bc_relabel_controller:gciql) on top of the two frozen k=50
# antsoccer checkpoints from the empowerment_final set.
#
# NOT DDPG+BC: the high-level action is a discrete skill index, and gciql's ddpgbc
# branch asserts a continuous action space (agents/gciql.py:93). The controller
# therefore pins base.discrete=True / base.actor_loss='awr' in
# skill_bc_relabel_controller._base_config, and this sweep moves AWR's inverse
# temperature instead. That is why the grid is {1, 3, 10} rather than a BC-coefficient
# grid: alpha multiplies the advantage inside exp(), so values ~0.1 flatten every
# weight to 1 and reduce the actor to plain BC on the chunk labels. gcivl uses 10.0
# and hiql 3.0 in hyperparameters.sh; dds_controller's paper config uses 3.0 for the
# same kind of discrete option-level high level. 0.3 is swept here (and NOT in the
# antmaze script) as the deliberate near-BC end of the range: it is where the
# advantage weighting is expected to wash out, which bounds the sweep from below.
#
# IQL expectile is pinned at 0.9 (gciql's default, restated explicitly so the run's
# flags.json records the intent).
#
# Sweep: 2 checkpoints x 4 alphas = 8 runs -> array 0..7.
#   IDX 0-3 : antsoccer-medium-navigate-v0  k=50  alpha in {0.3, 1, 3, 10}
#   IDX 4-7 : antsoccer-medium-stitch-v0    k=50  alpha in {0.3, 1, 3, 10}
#
# SKILL_CKPTS are the runs' ORIGINAL Savio save_dirs (read out of each local
# ckpts/empowerment_final/*/flags.json); the k50_s0.01_bc0.001 suffixes exist only on
# the rsynced copies under impls/ckpts. Override the root with EMP_ROOT=... if the
# runs have been moved off scratch.
#
# The pretrained agent is loaded read-only; only the controller is trained.
# Checkpoints go to
#   <SKILL_CKPT>/controller_awr_sweep/alpha<ALPHA>/OGBench/Debug/sd000_s_<jobid>.<ts>/
# so the pretrained params_*.pkl in each SKILL_CKPT is never touched, and each array
# task gets its own directory (get_exp_name folds SLURM_JOB_ID in).
#
# Submit from impls/:  sbatch scripts/run_skill_bc_relabel_awr_sweep_antsoccer.sh

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

IMPLS_DIR=/global/home/users/ishirgarg/ogbench/impls
EMP_ROOT=${EMP_ROOT:-/global/scratch/users/ishirgarg/ogbench/OGBench/Debug}

# ── Sweep definitions (parallel arrays indexed by IDX) ───────────────────────
CKPT_A="$EMP_ROOT/sd000_s_38005166.0.20260825_023027"   # antsoccer-medium-navigate-v0
CKPT_B="$EMP_ROOT/sd000_s_38006052.0.20260825_035401"   # antsoccer-medium-stitch-v0
SKILL_CKPTS=("$CKPT_A" "$CKPT_A" "$CKPT_A" "$CKPT_A" "$CKPT_B" "$CKPT_B" "$CKPT_B" "$CKPT_B")
ALPHAS=(0.3 1 3 10 0.3 1 3 10)
EXPECTILE=0.9
SEED=0

if [ -z "${IDX:-}" ] || [ "$IDX" -ge ${#SKILL_CKPTS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='${IDX:-}' out of range for ${#SKILL_CKPTS[@]} runs; use --array=0-$((${#SKILL_CKPTS[@]} - 1))." >&2
    exit 1
fi
SKILL_CKPT=${SKILL_CKPTS[$IDX]}
ALPHA=${ALPHAS[$IDX]}

# ── Environment ─────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

cd "$IMPLS_DIR"

# ── Read env_name / latest epoch off the pretrained run ─────────────────────
# -e above means a missing or moved SKILL_CKPT fails here rather than launching a
# job with an empty --env_name / --skill_restore_epoch.
ENV_NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
SKILL_EPOCH=$(python -c "
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
" "$SKILL_CKPT")

if [[ -z "$ENV_NAME" || -z "$SKILL_EPOCH" ]]; then
    echo "could not read env_name / latest epoch from $SKILL_CKPT" >&2
    exit 1
fi

SAVE_DIR="$SKILL_CKPT/controller_awr_sweep/alpha${ALPHA}"
mkdir -p "$SAVE_DIR"

echo "IDX=$IDX  ENV=$ENV_NAME  ALPHA=$ALPHA  EXPECTILE=$EXPECTILE  SEED=$SEED"
echo "  ckpt=$SKILL_CKPT"
echo "  epoch=$SKILL_EPOCH  save_dir=$SAVE_DIR"

# ── Run ─────────────────────────────────────────────────────────────────────
# `--agent.base.actor_loss` is deliberately not passed: _base_config pins it to
# 'awr' because the option MDP's action space is discrete.
python main.py \
    --env_name="$ENV_NAME" \
    --save_dir="$SAVE_DIR" \
    --agent=agents/skill_bc_relabel_controller.py:gciql \
    --agent.skill_checkpoint_path="$SKILL_CKPT" \
    --agent.skill_restore_epoch="$SKILL_EPOCH" \
    --agent.base.expectile=$EXPECTILE \
    --agent.base.alpha=$ALPHA \
    --seed=$SEED \
    --train_steps=1000000 \
    --log_interval=5000 \
    --eval_interval=100000 \
    --save_interval=100000 \
    --eval_episodes=50 \
    --video_episodes=0
