#!/bin/bash
#SBATCH --job-name=emp_comparison_navigate
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-7

# Offline empowerment comparison on the two navigate envs, matched configs
# (all agents share lr 3e-4, batch 1024, (512,512,512) trunks, layer_norm,
# discount 0.99; 15 skills where the agent has skills):
#     empowerment_crl_flowbc  — InfoNCE with flow-matching BC negatives
#     empowerment_opal_dads   — discrete-skill stochastic EM + BA bracket
#     empowerment_dv          — Donsker-Varadhan critic with flow-matching negatives
#     empowerment_crl         — dual-CRL critics + distilled E(s)  (launched last)
#
#   Full sweep = 4 agents x 2 envs = 8 runs  ->  --array=0-7
#
# Launch order (explicit per-index schedule):
#   IDX 0-2 : the three comparison agents on antsoccer
#   IDX 3-5 : the three comparison agents on antmaze
#   IDX 6-7 : the normal empowerment_crl on each env (launched last)
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..7)

IDX=${SLURM_ARRAY_TASK_ID}

# ── Sweep definitions ─────────────────────────────────────────────────────────
# Parallel arrays: RUN_ENVS[$IDX] / RUN_AGENTS[$IDX] give the (env, agent) pair.
RUN_ENVS=(
    antsoccer-arena-navigate-v0   # 0
    antsoccer-arena-navigate-v0   # 1
    antsoccer-arena-navigate-v0   # 2
    antmaze-medium-navigate-v0    # 3
    antmaze-medium-navigate-v0    # 4
    antmaze-medium-navigate-v0    # 5
    antsoccer-arena-navigate-v0   # 6
    antmaze-medium-navigate-v0    # 7
)
RUN_AGENTS=(
    empowerment_crl_flowbc        # 0
    empowerment_opal_dads         # 1
    empowerment_dv                # 2
    empowerment_crl_flowbc        # 3
    empowerment_opal_dads         # 4
    empowerment_dv                # 5
    empowerment_crl               # 6
    empowerment_crl               # 7
)
SEED=0

ENV=${RUN_ENVS[$IDX]}
AGENT=${RUN_AGENTS[$IDX]}

# 15 skills for the skill-based agent (empowerment_crl / _flowbc / _dv are
# action-level and have no num_skills key).
EXTRA_FLAGS=""
if [ "$AGENT" = "empowerment_opal_dads" ]; then
    EXTRA_FLAGS="--agent.num_skills=15"
fi

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  AGENT=$AGENT  SEED=$SEED  EXTRA=$EXTRA_FLAGS"

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
    --save_dir=$SAVE_DIR
