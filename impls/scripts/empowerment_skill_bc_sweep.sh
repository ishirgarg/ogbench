#!/bin/bash
#SBATCH --job-name=emp_skill_bc_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-14

IDX=${SLURM_ARRAY_TASK_ID}

# -----------------------------
# Sweep definitions
# -----------------------------
ENVS=(
    antmaze-medium-navigate-v0
    antsoccer-medium-navigate-v0
    antsoccer-arena-navigate-v0
    pointmaze-teleport-navigate-v0
    visual-antmaze-medium-navigate-v0
)
BC_ALPHAS=(0.01 0.03 0.1)

# Decode index: ENV outer, BC inner.
ENV_INDEX=$((IDX / 3))
BC_INDEX=$((IDX % 3))

ENV=${ENVS[$ENV_INDEX]}
BC_ALPHA=${BC_ALPHAS[$BC_INDEX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

# Visual envs need a CNN encoder + smaller batch size to fit on GPU.
EXTRA_ARGS=""
if [[ "$ENV" == visual-* ]]; then
    EXTRA_ARGS="--agent.batch_size=256 --agent.encoder=impala_small"
fi

# -----------------------------
# Run
# -----------------------------
export MUJOCO_GL=egl

python main.py \
    --env_name=$ENV \
    --save_dir=$SAVE_DIR \
    --agent=agents/empowerment_skill.py \
    --agent.num_skills=15 \
    --agent.bc_alpha=$BC_ALPHA \
    --train_steps=1500000 \
    $EXTRA_ARGS
