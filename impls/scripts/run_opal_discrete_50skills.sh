#!/bin/bash
#SBATCH --job-name=opal_discrete_50skills
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_low
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-3

# OPAL discrete path (agents/opal.py latent_type=discrete: Appendix F offline-
# DADS EM clustering -> BC) on the four antmaze/antsoccer datasets, identical to
# the discrete branch of run_opal_sweep.sh except num_skills=50 instead of 15
# (chunk/sequence length 10, cluster_steps 500k, 1M train steps, seed 0).
# Lowest-priority queue. The high-level skill policy is excluded, so goal
# success is low by design; watch training/mutual_information and
# training/num_active_skills instead.
#
#   IDX 0 : antmaze-medium-navigate-v0
#   IDX 1 : antsoccer-arena-navigate-v0
#   IDX 2 : antmaze-medium-stitch-v0
#   IDX 3 : antsoccer-arena-stitch-v0
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..3)
# Submit from impls/:  sbatch scripts/run_opal_discrete_50skills.sh

IDX=${SLURM_ARRAY_TASK_ID}

ENVS=(
    antmaze-medium-navigate-v0    # 0
    antsoccer-arena-navigate-v0   # 1
    antmaze-medium-stitch-v0      # 2
    antsoccer-arena-stitch-v0     # 3
)
SEED=0

ENV=${ENVS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  CONFIG=discrete  NUM_SKILLS=50  SEED=$SEED"

AGENT_FLAGS=(
    --agent.latent_type=discrete
    --agent.num_skills=50
    --agent.chunk_size=10
    --agent.sequence_length=10
    --agent.cluster_steps=500000
)

export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e

python main.py \
    --env_name=$ENV \
    --agent=agents/opal.py \
    "${AGENT_FLAGS[@]}" \
    --seed=$SEED \
    --train_steps=1000000 \
    --video_episodes=0 \
    --save_interval=25000 \
    --save_dir=$SAVE_DIR
