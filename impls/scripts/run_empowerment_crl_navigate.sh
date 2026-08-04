#!/bin/bash
#SBATCH --job-name=emp_crl_navigate
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

# Fresh (from-scratch, not resumed) empowerment_crl runs on the two navigate
# envs, high priority. Submit from impls/:  sbatch scripts/run_empowerment_crl_navigate.sh
#
# This does NOT touch the existing checkpoints under
# scripts/run_empowerment_crl_resume.sh's RUNS list — it starts new run
# folders from step 0 (e.g. if the prior distillation phase overfit and a
# clean restart is wanted instead of a resume).
#
#   IDX 0 : antmaze-medium-navigate-v0
#   IDX 1 : antsoccer-arena-navigate-v0

IDX=${SLURM_ARRAY_TASK_ID}

RUN_ENVS=(
    antmaze-medium-navigate-v0    # 0
    antsoccer-arena-navigate-v0   # 1
)
SEED=0

ENV=${RUN_ENVS[$IDX]}

if [ -z "$ENV" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RUN_ENVS[@]} envs; use --array=0-$((${#RUN_ENVS[@]} - 1))." >&2
    exit 1
fi

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  AGENT=empowerment_crl  SEED=$SEED"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --env_name=$ENV \
    --agent=agents/empowerment_crl.py \
    --seed=$SEED \
    --train_steps=1000000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR
