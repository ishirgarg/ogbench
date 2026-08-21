#!/bin/bash
#SBATCH --job-name=emp_skill_4env_noise_bc_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-63

IDX=${SLURM_ARRAY_TASK_ID}

# -----------------------------
# Sweep definitions
# -----------------------------
SKILLS=(50 15)
ENVS=(
    antmaze-medium-navigate-v0
    antmaze-medium-stitch-v0
    antsoccer-medium-navigate-v0
    antsoccer-medium-stitch-v0
)
BC_ALPHAS=(0.001 0.003)
ACTION_NOISE_STD=(0.0001 0.001 0.01 0.1)

# Decode index: SKILLS outer (50 first, then 15), then ENV, then BC_ALPHA,
# then ACTION_NOISE_STD inner.
SKILLS_INDEX=$((IDX / 32))
REMAINDER=$((IDX % 32))
ENV_INDEX=$((REMAINDER / 8))
REMAINDER=$((REMAINDER % 8))
BC_INDEX=$((REMAINDER / 4))
NOISE_INDEX=$((REMAINDER % 4))

NUM_SKILLS=${SKILLS[$SKILLS_INDEX]}
ENV=${ENVS[$ENV_INDEX]}
BC_ALPHA=${BC_ALPHAS[$BC_INDEX]}
NOISE=${ACTION_NOISE_STD[$NOISE_INDEX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

# -----------------------------
# Run
# -----------------------------
export MUJOCO_GL=egl

python main.py \
    --env_name=$ENV \
    --save_dir=$SAVE_DIR \
    --agent=agents/empowerment_skill.py \
    --agent.num_skills=$NUM_SKILLS \
    --agent.bc_alpha=$BC_ALPHA \
    --agent.stochastic_policy_actions=True \
    --agent.action_noise_std=$NOISE \
    --agent.perturb_q_loss_actions=True \
    --agent.log_interval=8000 \
    --log_interval=8000 \
    --train_steps=1000000 \
    --video_episodes=0
