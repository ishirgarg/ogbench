#!/bin/bash
#SBATCH --job-name=emp_skill_bc_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-7

IDX=${SLURM_ARRAY_TASK_ID}

# -----------------------------
# Sweep definitions
# -----------------------------
ENVS=(
    antmaze-medium-stitch-v0
    antsoccer-medium-stitch-v0
    antsoccer-arena-stitch-v0
    pointmaze-teleport-stitch-v0
)
BC_ALPHAS=(0.01 0.03)

# Decode index: ENV outer, BC inner.
ENV_INDEX=$((IDX / 2))
BC_INDEX=$((IDX % 2))

ENV=${ENVS[$ENV_INDEX]}
BC_ALPHA=${BC_ALPHAS[$BC_INDEX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

# -----------------------------
# Run
# -----------------------------
export MUJOCO_GL=egl

python main.py \
    --env_name=$ENV \
    --save_dir=$SAVE_DIR \
    --agent=agents/empowerment_skill.py \
    --agent.num_skills=15 \
    --agent.bc_alpha=$BC_ALPHA \
    --train_steps=1500000 \
    --video_episodes=0
