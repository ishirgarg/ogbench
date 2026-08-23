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
# (all agents share lr 3e-4, batch 1024, (512,512,512) trunks, layer_norm,
# discount 0.99):
#     empowerment_crl_flowbc  — InfoNCE with flow-matching BC negatives
#     empowerment_dv          — Donsker-Varadhan critic with flow-matching negatives
#     empowerment_crl         — dual-CRL critics + distilled E(s)  (launched last)
#
#   Full sweep = 3 agents x 2 envs = 6 runs  ->  --array=0-5
#
# Launch order (explicit per-index schedule):
#   IDX 0-1 : the two comparison agents on antsoccer
#   IDX 2-3 : the two comparison agents on antmaze
#   IDX 4-5 : the normal empowerment_crl on each env (launched last)
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..5)

IDX=${SLURM_ARRAY_TASK_ID}

# ── Sweep definitions ─────────────────────────────────────────────────────────
# Parallel arrays: RUN_ENVS[$IDX] / RUN_AGENTS[$IDX] give the (env, agent) pair.
RUN_ENVS=(
    antsoccer-arena-navigate-v0   # 0
    antsoccer-arena-navigate-v0   # 1
    antmaze-medium-navigate-v0    # 2
    antmaze-medium-navigate-v0    # 3
    antsoccer-arena-navigate-v0   # 4
    antmaze-medium-navigate-v0    # 5
)
RUN_AGENTS=(
    empowerment_crl_flowbc        # 0
    empowerment_dv                # 1
    empowerment_crl_flowbc        # 2
    empowerment_dv                # 3
    empowerment_crl               # 4
    empowerment_crl               # 5
)
SEED=0

ENV=${RUN_ENVS[$IDX]}
AGENT=${RUN_AGENTS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  AGENT=$AGENT  SEED=$SEED"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --env_name=$ENV \
    --agent=agents/${AGENT}.py \
    --seed=$SEED \
    --train_steps=1000000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR
