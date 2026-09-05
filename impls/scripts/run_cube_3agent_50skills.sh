#!/bin/bash
#SBATCH --job-name=cube_3agent_50skills
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-5

# The three 50-skill agents -- empowerment_skill, DDS (agents/dds.py) and OPAL
# discrete (agents/opal.py latent_type=discrete) -- on the two manipulation
# datasets that the existing locomotion sweep leaves out: cube-single-play-v0
# and cube-double-play-v0.
#
# Flags per agent are copied verbatim from the scripts that run the same three
# agents over the other envs, so the cube numbers are directly comparable:
#   empowerment_skill : scripts/run_empowerment_skill_extra_envs_priority.sh
#   dds               : scripts/run_dds_50skills.sh
#   opal (discrete)   : scripts/run_opal_discrete_50skills.sh
# Shared across all three: num_skills = 50, seed 0, train_steps = 1e6,
# log_interval 8000, video_episodes 0, savio4_gpu / rail_gpu4_high.
#
# empowerment_skill uses bc_alpha=0.001 -- the value the priority script uses
# for every non-cube env (its cube rows are a bc_alpha grid; this run pins the
# comparable setting) -- plus noise=0.01, stochastic_policy_actions,
# perturb_q_loss_actions.
# dds keeps every agents/dds.py get_config() paper default (D_z=128, H=10,
# 5 diffusion steps, 500k VQ-VAE pretrain + 500k semi-MDP IQL/AWR) with only
# the codebook size overridden to K=50; save_interval 25000.
# opal keeps chunk_size=sequence_length=10, cluster_steps=500000,
# save_interval 25000. The high-level skill policy is excluded, so goal success
# is low by design -- watch training/mutual_information and
# training/num_active_skills.
#
#   IDX 0 : empowerment_skill  cube-single-play-v0
#   IDX 1 : empowerment_skill  cube-double-play-v0
#   IDX 2 : dds                cube-single-play-v0
#   IDX 3 : dds                cube-double-play-v0
#   IDX 4 : opal (discrete)    cube-single-play-v0
#   IDX 5 : opal (discrete)    cube-double-play-v0
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..5)
# Submit from impls/:  sbatch scripts/run_cube_3agent_50skills.sh

IDX=${SLURM_ARRAY_TASK_ID}

RUN_AGENTS=(
    empowerment_skill  # 0
    empowerment_skill  # 1
    dds                # 2
    dds                # 3
    opal               # 4
    opal               # 5
)
RUN_ENVS=(
    cube-single-play-v0  # 0
    cube-double-play-v0  # 1
    cube-single-play-v0  # 2
    cube-double-play-v0  # 3
    cube-single-play-v0  # 4
    cube-double-play-v0  # 5
)
SEED=0
SKILLS=50            # num_skills / DDS codebook size K
BC_ALPHA=0.001       # empowerment_skill only
NOISE=0.01           # empowerment_skill only

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
    opal)
        python main.py \
            --env_name=$ENV \
            --agent=agents/opal.py \
            --agent.latent_type=discrete \
            --agent.num_skills=$SKILLS \
            --agent.chunk_size=10 \
            --agent.sequence_length=10 \
            --agent.cluster_steps=500000 \
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
