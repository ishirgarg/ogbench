#!/bin/bash
#SBATCH --job-name=emp_skill_sample_policy_bc
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
    antmaze-medium-navigate-v0
)
ACTION_NOISE_STD=(0.1 0.2)
BC_ALPHAS=(0.001 0.003)
SKILLS=15

# Decode index: ENV outer, ACTION_NOISE_STD middle, BC_ALPHA inner.
ENV_INDEX=$((IDX / 4))
REMAINDER=$((IDX % 4))
NOISE_INDEX=$((REMAINDER / 2))
BC_INDEX=$((REMAINDER % 2))

ENV=${ENVS[$ENV_INDEX]}
NOISE=${ACTION_NOISE_STD[$NOISE_INDEX]}
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
    --agent.num_skills=$SKILLS \
    --agent.bc_alpha=$BC_ALPHA \
    --agent.action_noise_std=$NOISE \
    --agent.stochastic_policy_actions=True \
    --train_steps=1000000 \
    --video_episodes=0
