#!/bin/bash
#SBATCH --job-name=emp_skill_sample_policy_antsoccer_50
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00

ENV=antsoccer-arena-navigate-v0
NOISE=0.1
BC_ALPHA=0.01
SKILLS=50

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
