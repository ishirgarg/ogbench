#!/bin/bash
#SBATCH --job-name=cube_noisy_dds
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

# DDS (agents/dds.py) on the two cube manipulation envs with the noisy
# dataset variant, same launch shape as scripts/run_dds_50skills.sh /
# the dds case in scripts/run_cube_3agent_50skills.sh: codebook size
# K = num_skills = 50, single seed (0), every other agents/dds.py
# get_config() paper default left unchanged, log_interval 8000,
# train_steps 1e6, video_episodes 0.
#
#   IDX 0 : cube-single-noisy-v0
#   IDX 1 : cube-double-noisy-v0
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..1)
# Submit from impls/:  sbatch scripts/run_cube_noisy_dds.sh

IDX=${SLURM_ARRAY_TASK_ID}

ENVS=(
    cube-single-noisy-v0  # 0
    cube-double-noisy-v0  # 1
)
SEED=0
K=50   # codebook size (num_skills)

if [ -z "$IDX" ] || [ "$IDX" -ge ${#ENVS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#ENVS[@]} runs; use --array=0-$((${#ENVS[@]} - 1))." >&2
    exit 1
fi
ENV=${ENVS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  num_skills(K)=$K  SEED=$SEED"

export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e

python main.py \
    --env_name=$ENV \
    --agent=agents/dds.py \
    --agent.num_skills=$K \
    --seed=$SEED \
    --train_steps=1000000 \
    --log_interval=8000 \
    --video_episodes=0 \
    --save_interval=25000 \
    --save_dir=$SAVE_DIR
