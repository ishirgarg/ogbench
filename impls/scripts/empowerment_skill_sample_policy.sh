#!/bin/bash
#SBATCH --job-name=emp_skill_sample_policy
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
    antmaze-medium-navigate-v0
    antmaze-medium-stitch-v0
    antsoccer-arena-navigate-v0
    antsoccer-arena-stitch-v0
)
NUM_SKILLS=(15 50)
BC_ALPHA=0.01

# Decode index: ENV outer, NUM_SKILLS inner.
ENV_INDEX=$((IDX / 2))
SKILL_INDEX=$((IDX % 2))

ENV=${ENVS[$ENV_INDEX]}
SKILLS=${NUM_SKILLS[$SKILL_INDEX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

# -----------------------------
# Run
# -----------------------------
export MUJOCO_GL=egl

python main.py \
    --env_name=$ENV \
    --save_dir=$SAVE_DIR \
    --agent=agents/empowerment_skill.py \
    --agent.num_skills=$SKILLS \
    --agent.bc_alpha=$BC_ALPHA \
    --agent.stochastic_policy_actions=True \
    --train_steps=1500000 \
    --video_episodes=0
