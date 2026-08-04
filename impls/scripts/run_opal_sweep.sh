#!/bin/bash
#SBATCH --job-name=opal_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-7

# OPAL on the four antmaze/antsoccer datasets, both skill-extraction paths of
# agents/opal.py: the continuous VAE and the Appendix F offline-DADS clustering
# -> BC path (k=10). Both train on length-10 sequence windows. The high-level
# skill policy is excluded from both, so goal success is low by design; for the
# discrete runs watch training/mutual_information and training/num_active_skills
# instead.
#
#   4 envs x 2 configs, seed 0.  ENV = IDX / 2, CONFIG = IDX % 2, envs ordered
#   navigate first, so IDX 0-3 are navigate and IDX 4-7 are stitch. Submit
#   --array=0-3 then --array=4-7 to run them in separate batches.

IDX=${SLURM_ARRAY_TASK_ID}

ENVS=(
    # navigate first ...
    antmaze-medium-navigate-v0
    antsoccer-arena-navigate-v0
    # ... then stitch
    antmaze-medium-stitch-v0
    antsoccer-arena-stitch-v0
)
CONFIGS=(continuous discrete)
SEED=0

ENV=${ENVS[$((IDX / 2))]}
CONFIG=${CONFIGS[$((IDX % 2))]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  CONFIG=$CONFIG  SEED=$SEED"

if [ "$CONFIG" = "continuous" ]; then
    AGENT_FLAGS=(
        --agent.latent_type=continuous
        --agent.skill_dim=8
        --agent.kl_coef=0.1
        --agent.chunk_size=10
        --agent.sequence_length=10
    )
else
    AGENT_FLAGS=(
        --agent.latent_type=discrete
        --agent.num_skills=10
        --agent.chunk_size=10
        --agent.sequence_length=10
        --agent.cluster_steps=500000
    )
fi

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
    --save_dir=$SAVE_DIR
