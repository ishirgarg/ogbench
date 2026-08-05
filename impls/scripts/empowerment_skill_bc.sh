#!/bin/bash
#SBATCH --job-name=emp_skill_only_bc
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

IDX=${SLURM_ARRAY_TASK_ID}

# -----------------------------
# Sweep definitions
# -----------------------------
ENVS=(
    antmaze-medium-stitch-v0
    antsoccer-medium-stitch-v0
)

ENV=${ENVS[$IDX]}

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
    --agent.stochastic_policy_actions=True \
    --agent.only_bc=True \
    --train_steps=1000000 \
    --video_episodes=0
