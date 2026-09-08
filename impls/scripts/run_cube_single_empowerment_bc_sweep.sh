#!/bin/bash
#SBATCH --job-name=cube_single_emp_bc_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-3

# High-priority (rail_gpu4_high) bc_alpha sweep for empowerment_skill on
# cube-single-play-v0. Same agent flags and structure as
# scripts/run_cube_3agent_50skills.sh (num_skills=50, seed 0, noise=0.01,
# stochastic_policy_actions, perturb_q_loss_actions, train_steps=1e6,
# log_interval 8000, video_episodes 0) -- the only thing that varies across
# the array is bc_alpha.
#
#   IDX 0 : cube-single-play-v0  k=50  noise=0.01  bc=0.01
#   IDX 1 : cube-single-play-v0  k=50  noise=0.01  bc=0.1
#   IDX 2 : cube-single-play-v0  k=50  noise=0.01  bc=1
#   IDX 3 : cube-single-play-v0  k=50  noise=0.01  bc=10
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..3)
# Submit from impls/:  sbatch scripts/run_cube_single_empowerment_bc_sweep.sh

IDX=${SLURM_ARRAY_TASK_ID}

RUN_BC_ALPHAS=(
    0.01  # 0
    0.1   # 1
    1     # 2
    10    # 3
)
ENV=cube-single-play-v0
SEED=0
SKILLS=50
NOISE=0.01

if [ -z "$IDX" ] || [ "$IDX" -ge ${#RUN_BC_ALPHAS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RUN_BC_ALPHAS[@]} runs; use --array=0-$((${#RUN_BC_ALPHAS[@]} - 1))." >&2
    exit 1
fi
BC_ALPHA=${RUN_BC_ALPHAS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  AGENT=empowerment_skill  ENV=$ENV  SKILLS=$SKILLS  BC_ALPHA=$BC_ALPHA  SEED=$SEED"

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
