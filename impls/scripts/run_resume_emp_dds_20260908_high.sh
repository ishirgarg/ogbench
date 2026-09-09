#!/bin/bash
#SBATCH --job-name=emp_stitch_100skills_high
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

# High-priority (rail_gpu4_high) 100-skill empowerment_skill runs on the two
# stitch datasets:
#
#   IDX 0 : empowerment_skill  antsoccer-arena-stitch-v0     k=100 noise=0.01 bc=0.001
#   IDX 1 : empowerment_skill  pointmaze-teleport-stitch-v0  k=100 noise=0.01 bc=0.001
#
# These are FRESH runs, not resumes -- each starts a new run folder and a new
# wandb run.
#
# Flags match the empowerment_skill rows of scripts/run_stitch_100skills_normal.sh
# (itself the stitch rows of scripts/run_empowerment_skill_extra_envs_priority.sh
# with num_skills raised to 100): bc_alpha=0.001, noise=0.01,
# stochastic_policy_actions, perturb_q_loss_actions, seed 0, train_steps = 1e6,
# log_interval 8000, video_episodes 0. No dds runs here.
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..1)
# Submit from impls/:  sbatch scripts/run_resume_emp_dds_20260908_high.sh
# One run only:        sbatch --array=1 scripts/run_resume_emp_dds_20260908_high.sh

IDX=${SLURM_ARRAY_TASK_ID}

RUN_ENVS=(
    antsoccer-arena-stitch-v0     # 0
    pointmaze-teleport-stitch-v0  # 1
)
SEED=0
SKILLS=100      # num_skills
BC_ALPHA=0.001
NOISE=0.01

if [ -z "$IDX" ] || [ "$IDX" -ge ${#RUN_ENVS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RUN_ENVS[@]} runs; use --array=0-$((${#RUN_ENVS[@]} - 1))." >&2
    exit 1
fi
ENV=${RUN_ENVS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  AGENT=empowerment_skill  ENV=$ENV  SKILLS=$SKILLS  SEED=$SEED"

export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e

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
