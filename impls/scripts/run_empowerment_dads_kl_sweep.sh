#!/bin/bash
#SBATCH --job-name=emp_dads_kl_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_lowest
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-7

# empowerment_dads (MI-maximizing label assignment, Barber-Agakov bracket) on
# the two navigate envs and their stitch counterparts, sweeping the uniform-
# usage KL coefficient (kl_coef in {0.1, 1.0}). Otherwise matched to
# run_empowerment_dads_navigate.sh (lr 3e-4, batch 1024, (512,512,512) trunks,
# layer_norm, discount 0.99, 15 skills). Lowest-priority queue.
#
# Submit from impls/:  sbatch scripts/run_empowerment_dads_kl_sweep.sh
#
#   IDX 0 : antsoccer-arena-navigate-v0   kl_coef=0.1
#   IDX 1 : antmaze-medium-navigate-v0    kl_coef=0.1
#   IDX 2 : antsoccer-arena-stitch-v0     kl_coef=0.1
#   IDX 3 : antmaze-medium-stitch-v0      kl_coef=0.1
#   IDX 4 : antsoccer-arena-navigate-v0   kl_coef=1.0
#   IDX 5 : antmaze-medium-navigate-v0    kl_coef=1.0
#   IDX 6 : antsoccer-arena-stitch-v0     kl_coef=1.0
#   IDX 7 : antmaze-medium-stitch-v0      kl_coef=1.0
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..7)  ->  ENV = IDX % 4, KL = IDX / 4

IDX=${SLURM_ARRAY_TASK_ID}

# ── Sweep definitions ─────────────────────────────────────────────────────────
RUN_ENVS=(
    antsoccer-arena-navigate-v0   # 0
    antmaze-medium-navigate-v0    # 1
    antsoccer-arena-stitch-v0     # 2
    antmaze-medium-stitch-v0      # 3
)
KL_COEFS=(
    0.1   # 0
    1.0   # 1
)
SEED=0
NUM_SKILLS=15

NENV=${#RUN_ENVS[@]}
NKL=${#KL_COEFS[@]}
if [ -z "$IDX" ] || [ "$IDX" -ge $((NENV * NKL)) ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for $((NENV * NKL)) runs; use --array=0-$((NENV * NKL - 1))." >&2
    exit 1
fi

ENV=${RUN_ENVS[$((IDX % NENV))]}
KL=${KL_COEFS[$((IDX / NENV))]}
AGENT=empowerment_dads

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  AGENT=$AGENT  KL_COEF=$KL  NUM_SKILLS=$NUM_SKILLS  SEED=$SEED"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --env_name=$ENV \
    --agent=agents/${AGENT}.py \
    --agent.num_skills=$NUM_SKILLS \
    --agent.kl_coef=$KL \
    --seed=$SEED \
    --train_steps=1000000 \
    --save_interval=25000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR
