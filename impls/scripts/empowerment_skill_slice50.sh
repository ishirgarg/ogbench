#!/bin/bash
#SBATCH --job-name=emp_skill_slice50
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-5

IDX=${SLURM_ARRAY_TASK_ID}

# -----------------------------
# Sweep definitions
# -----------------------------
# The `-slice50-` token splits every trajectory of the underlying stitch dataset
# into sub-trajectories of 50 transitions each (see impls/utils/dataset_slicing.py).
# The sliced .npz is generated into ~/.ogbench/data on first use and cached there.
ENVS=(
    antmaze-medium-stitch-slice50-v0
    antsoccer-arena-stitch-slice50-v0
)
NUM_SKILLS=(15 50 100)
BC_ALPHA=0.01

# Decode index: ENV outer, NUM_SKILLS inner.
ENV_INDEX=$((IDX / 3))
SKILL_INDEX=$((IDX % 3))

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
    --train_steps=1500000 \
    --video_episodes=0
