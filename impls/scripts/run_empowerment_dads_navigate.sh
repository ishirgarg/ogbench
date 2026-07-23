#!/bin/bash
#SBATCH --job-name=emp_dads_navigate
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

# empowerment_dads (MI-maximizing label assignment, Barber-Agakov bracket) on
# the two navigate envs, matched to run_empowerment_comparison_navigate.sh
# (lr 3e-4, batch 1024, (512,512,512) trunks, layer_norm, discount 0.99,
# 15 skills).
#
#   IDX 0 : antsoccer-arena-navigate-v0
#   IDX 1 : antmaze-medium-navigate-v0
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..1)

IDX=${SLURM_ARRAY_TASK_ID}

# ── Sweep definitions ─────────────────────────────────────────────────────────
RUN_ENVS=(
    antsoccer-arena-navigate-v0   # 0
    antmaze-medium-navigate-v0    # 1
)
SEED=0

ENV=${RUN_ENVS[$IDX]}
AGENT=empowerment_dads

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench
RUN_GROUP="emp_cmp_${AGENT}_${ENV}"

echo "IDX=$IDX  ENV=$ENV  AGENT=$AGENT  SEED=$SEED  RUN_GROUP=$RUN_GROUP"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --env_name=$ENV \
    --agent=agents/${AGENT}.py \
    --agent.num_skills=15 \
    --seed=$SEED \
    --train_steps=1000000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR \
    --run_group=$RUN_GROUP
