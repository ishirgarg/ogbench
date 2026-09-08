#!/bin/bash
#SBATCH --job-name=stitch_100skills_normal
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-3

# Normal-priority (rail_gpu4_normal) 100-skill runs of empowerment_skill and
# DDS (agents/dds.py) on the two stitch datasets:
# antsoccer-arena-stitch-v0 and pointmaze-teleport-stitch-v0.
#
# Same agent flags as the 50-skill scripts, with num_skills / codebook size
# K raised to 100:
#   dds               : scripts/run_dds_50skills.sh
#   empowerment_skill : scripts/run_empowerment_skill_extra_envs_priority.sh
#                       (its stitch rows: noise=0.01, bc=0.001)
#
# dds keeps every agents/dds.py get_config() paper default (D_z=128, H=10,
# 5 diffusion steps, 500k VQ-VAE pretrain + 500k semi-MDP IQL/AWR) with only
# the codebook size overridden to K=100; save_interval 25000.
# empowerment_skill uses bc_alpha=0.001, noise=0.01,
# stochastic_policy_actions, perturb_q_loss_actions.
# Shared: seed 0, train_steps = 1e6, log_interval 8000, video_episodes 0.
#
#   IDX 0 : empowerment_skill  antsoccer-arena-stitch-v0     k=100 noise=0.01 bc=0.001
#   IDX 1 : empowerment_skill  pointmaze-teleport-stitch-v0  k=100 noise=0.01 bc=0.001
#   IDX 2 : dds                antsoccer-arena-stitch-v0     K=100
#   IDX 3 : dds                pointmaze-teleport-stitch-v0  K=100
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..3)
# Submit from impls/:  sbatch scripts/run_stitch_100skills_normal.sh

IDX=${SLURM_ARRAY_TASK_ID}

RUN_AGENTS=(
    empowerment_skill  # 0
    empowerment_skill  # 1
    dds                # 2
    dds                # 3
)
RUN_ENVS=(
    antsoccer-arena-stitch-v0     # 0
    pointmaze-teleport-stitch-v0  # 1
    antsoccer-arena-stitch-v0     # 2
    pointmaze-teleport-stitch-v0  # 3
)
SEED=0
SKILLS=100      # num_skills / DDS codebook size K
BC_ALPHA=0.001  # empowerment_skill only
NOISE=0.01      # empowerment_skill only

if [ -z "$IDX" ] || [ "$IDX" -ge ${#RUN_ENVS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RUN_ENVS[@]} runs; use --array=0-$((${#RUN_ENVS[@]} - 1))." >&2
    exit 1
fi
AGENT=${RUN_AGENTS[$IDX]}
ENV=${RUN_ENVS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  AGENT=$AGENT  ENV=$ENV  SKILLS=$SKILLS  SEED=$SEED"

export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e

case $AGENT in
    empowerment_skill)
        python main.py \
            --env_name=$ENV \
            --save_dir=$SAVE_DIR \
            --agent=agents/empowerment_skill.py \
            --agent.num_skills=$SKILLS \
            --agent.bc_alpha=$BC_ALPHA \
            --agent.stochastic_policy_actions=True \
            --agent.action_noise_std=$NOISE \
            --agent.perturb_q_loss_actions=True \
            --agent.log_interval=8000 \
            --seed=$SEED \
            --log_interval=8000 \
            --train_steps=1000000 \
            --video_episodes=0
        ;;
    dds)
        python main.py \
            --env_name=$ENV \
            --agent=agents/dds.py \
            --agent.num_skills=$SKILLS \
            --seed=$SEED \
            --train_steps=1000000 \
            --log_interval=8000 \
            --video_episodes=0 \
            --save_interval=25000 \
            --save_dir=$SAVE_DIR
        ;;
    *)
        echo "ERROR: unknown agent '$AGENT'" >&2
        exit 1
        ;;
esac
