#!/bin/bash
#SBATCH --job-name=opal_50skills_antsoccer_navigate_kl1
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00

# OPAL discrete path (agents/opal.py latent_type=discrete: Appendix F offline-
# DADS EM clustering -> BC), antsoccer-arena-navigate-v0 only, identical to
# IDX=1 of run_opal_discrete_50skills.sh (num_skills=50, chunk/sequence length
# 10, cluster_steps 500k, 1M train steps, seed 0) except kl_coef=1 instead of
# the default 0.1, and run on its own on the normal-priority queue.
#
# Submit from impls/:  sbatch scripts/run_opal_discrete_50skills_antsoccer_navigate_kl1.sh

ENV=antsoccer-arena-navigate-v0
SEED=0

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "ENV=$ENV  CONFIG=discrete  NUM_SKILLS=50  KL_COEF=1  SEED=$SEED"

AGENT_FLAGS=(
    --agent.latent_type=discrete
    --agent.num_skills=50
    --agent.chunk_size=10
    --agent.sequence_length=10
    --agent.cluster_steps=500000
    --agent.kl_coef=1
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
