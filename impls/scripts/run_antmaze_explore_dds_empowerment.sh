#!/bin/bash
#SBATCH --job-name=antmaze_explore_dds_emp
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-2

# DDS (agents/dds.py) and empowerment_skill on antmaze-medium-explore-v0 --
# the antmaze-only subset of scripts/run_explore_dds_empowerment.sh, which is
# the only `explore` dataset already present in ~/.ogbench/data.
#
# Flags are copied verbatim from the scripts that run the same two agents on
# the navigate/stitch datasets, so these numbers are directly comparable:
#   dds               : scripts/run_dds_50skills.sh
#   empowerment_skill : scripts/empowerment_skill_4env_noise_bc_sweep.sh
#
# dds keeps every agents/dds.py get_config() paper default (D_z=128, H=10,
# 5 diffusion steps, 500k VQ-VAE pretrain + 500k semi-MDP IQL/AWR) with only
# the codebook size overridden to K = num_skills = 50; save_interval 25000.
# empowerment_skill uses num_skills=50, stochastic_policy_actions,
# perturb_q_loss_actions, action_noise_std pinned to 0.01 (no noise sweep
# here) and bc_alpha swept over {0.001, 0.0001}.
# Shared: seed 0, train_steps = 1e6, log_interval 8000, video_episodes 0,
# savio4_gpu / rail_gpu4_high.
#
#   IDX 0 : dds                antmaze-medium-explore-v0  K=50
#   IDX 1 : empowerment_skill  antmaze-medium-explore-v0  k=50 noise=0.01 bc=0.001
#   IDX 2 : empowerment_skill  antmaze-medium-explore-v0  k=50 noise=0.01 bc=0.0001
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..2)
# Submit from impls/:  sbatch scripts/run_antmaze_explore_dds_empowerment.sh

IDX=${SLURM_ARRAY_TASK_ID}

RUN_AGENTS=(
    dds                # 0
    empowerment_skill  # 1
    empowerment_skill  # 2
)
RUN_BC_ALPHAS=(
    -       # 0 (dds: unused)
    0.001   # 1
    0.0001  # 2
)
ENV=antmaze-medium-explore-v0
SEED=0
SKILLS=50    # num_skills / DDS codebook size K
NOISE=0.01   # empowerment_skill only, pinned (no noise sweep)

if [ -z "$IDX" ] || [ "$IDX" -ge ${#RUN_AGENTS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RUN_AGENTS[@]} runs; use --array=0-$((${#RUN_AGENTS[@]} - 1))." >&2
    exit 1
fi
AGENT=${RUN_AGENTS[$IDX]}
BC_ALPHA=${RUN_BC_ALPHAS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  AGENT=$AGENT  ENV=$ENV  SKILLS=$SKILLS  BC_ALPHA=$BC_ALPHA  SEED=$SEED"

export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e

case $AGENT in
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
    *)
        echo "ERROR: unknown agent '$AGENT'" >&2
        exit 1
        ;;
esac
