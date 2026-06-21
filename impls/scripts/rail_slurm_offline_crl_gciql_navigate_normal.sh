#!/bin/bash
#SBATCH --job-name=ogb_offline_normal
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-3

# Offline CRL + GCIQL on the two navigate datasets, normal-priority slice.
#
# Trains the exact runs from impls/hyperparameters.sh:
#   antmaze-medium-navigate-v0   (GCIQL)  --agent.alpha=0.3
#   antmaze-medium-navigate-v0   (CRL)    --agent.alpha=0.1
#   antsoccer-arena-navigate-v0  (GCIQL)  --agent.alpha=0.1
#   antsoccer-arena-navigate-v0  (CRL)    --agent.alpha=0.3
#
# Sweep: 4 (env,agent,alpha) configs x 1 seed = 4 runs -> array 0..3.
#
# Index decoding:
#   CFG_IDX = SLURM_ARRAY_TASK_ID                  (0..3)
#     CFG 0 -> antmaze   gciql alpha 0.3
#     CFG 1 -> antmaze   crl   alpha 0.1
#     CFG 2 -> antsoccer gciql alpha 0.1
#     CFG 3 -> antsoccer crl   alpha 0.3

export MUJOCO_GL=egl

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

# ogbench checkout on BRC.
IMPLS_DIR="/global/home/users/ishirgarg/ogbench/impls"
SAVE_DIR="/global/scratch/users/ishirgarg/ogbench/exp"
mkdir -p "$SAVE_DIR"

# -----------------------------
# Sweep definitions (parallel arrays indexed by CFG_IDX)
# -----------------------------
ENVS=(antmaze-medium-navigate-v0 antmaze-medium-navigate-v0 antsoccer-arena-navigate-v0 antsoccer-arena-navigate-v0)
AGENTS=(gciql crl gciql crl)
ALPHAS=(0.3 0.1 0.1 0.3)
SEED=0

# -----------------------------
# Decode index
# -----------------------------
CFG_IDX=${SLURM_ARRAY_TASK_ID}

ENV=${ENVS[$CFG_IDX]}
AGENT=${AGENTS[$CFG_IDX]}
ALPHA=${ALPHAS[$CFG_IDX]}

RUN_GROUP="${ENV}__${AGENT}_a${ALPHA}"

echo "IDX=$IDX  ENV=$ENV  AGENT=$AGENT  ALPHA=$ALPHA  SEED=$SEED  GROUP=$RUN_GROUP"

cd "$IMPLS_DIR"

python main.py \
        --env_name=$ENV \
        --eval_episodes=50 \
        --agent=agents/${AGENT}.py \
        --agent.alpha=$ALPHA \
        --seed=$SEED \
        --save_dir=$SAVE_DIR \
        --run_group=$RUN_GROUP
