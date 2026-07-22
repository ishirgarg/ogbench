#!/bin/bash
#SBATCH --job-name=emp_dv_navigate
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

# Offline empowerment_dv (Donsker-Varadhan critic with flow-matching negatives)
# on the two navigate envs, config matched to the empowerment comparison sweep
# (lr 3e-4, batch 1024, (512,512,512) trunks, layer_norm, discount 0.99).
#
#   Sweep = 2 envs x 1 agent = 2 runs  ->  --array=0-1
#
# Index decoding:
#   IDX     = SLURM_ARRAY_TASK_ID   (0..1)
#   ENV_IDX = IDX                   (0..1)

IDX=${SLURM_ARRAY_TASK_ID}

# ── Sweep definitions ─────────────────────────────────────────────────────────
ENVS=(
    antmaze-medium-navigate-v0
    antsoccer-arena-navigate-v0
)
AGENT=empowerment_dv
SEED=0

ENV=${ENVS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench
RUN_GROUP="emp_cmp_${AGENT}_${ENV}"

echo "IDX=$IDX  ENV=$ENV  AGENT=$AGENT  SEED=$SEED  RUN_GROUP=$RUN_GROUP"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --env_name=$ENV \
    --agent=agents/${AGENT}.py \
    --seed=$SEED \
    --train_steps=1000000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR \
    --run_group=$RUN_GROUP
