#!/bin/bash
#SBATCH --job-name=emp_comparison_navigate
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-5

# Offline empowerment comparison on the two navigate envs, matched configs
# (all three agents share lr 3e-4, batch 1024, (512,512,512) trunks,
# layer_norm, discount 0.99; 15 skills where the agent has skills):
#     empowerment_crl         — dual-CRL critics + distilled E(s)
#     empowerment_crl_flowbc  — InfoNCE with flow-matching BC negatives
#     empowerment_opal_dads   — discrete-skill stochastic EM + BA bracket
#
#   Full sweep = 2 envs x 3 agents = 6 runs  ->  --array=0-5
#
# Index decoding (ENV outer, AGENT inner):
#   IDX       = SLURM_ARRAY_TASK_ID   (0..5)
#   AGENT_IDX = IDX % 3               (0..2)
#   ENV_IDX   = IDX / 3               (0..1)

IDX=${SLURM_ARRAY_TASK_ID}

# ── Sweep definitions ─────────────────────────────────────────────────────────
ENVS=(
    antmaze-medium-navigate-v0
    antsoccer-arena-navigate-v0
)
AGENTS=(
    empowerment_crl
    empowerment_crl_flowbc
    empowerment_opal_dads
)
SEED=0

AGENT_IDX=$((IDX % 3))
ENV_IDX=$((IDX / 3))

ENV=${ENVS[$ENV_IDX]}
AGENT=${AGENTS[$AGENT_IDX]}

# 15 skills for the skill-based agent (empowerment_crl / _flowbc are
# action-level and have no num_skills key).
EXTRA_FLAGS=""
if [ "$AGENT" = "empowerment_opal_dads" ]; then
    EXTRA_FLAGS="--agent.num_skills=15"
fi

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench
RUN_GROUP="emp_cmp_${AGENT}_${ENV}"

echo "IDX=$IDX  ENV=$ENV  AGENT=$AGENT  SEED=$SEED  RUN_GROUP=$RUN_GROUP  EXTRA=$EXTRA_FLAGS"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --env_name=$ENV \
    --agent=agents/${AGENT}.py \
    $EXTRA_FLAGS \
    --seed=$SEED \
    --train_steps=1000000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR \
    --run_group=$RUN_GROUP
