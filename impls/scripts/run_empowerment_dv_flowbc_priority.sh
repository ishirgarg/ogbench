#!/bin/bash
#SBATCH --job-name=emp_dv_flowbc_priority
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-3

# High-priority (rail_gpu4_high) FRESH (from-scratch, not resumed) launch of
# all four empowerment_dv / empowerment_crl_flowbc navigate runs (2 agents x
# 2 envs), each on its own GPU. Starts new run folders from step 0 instead of
# resuming the runs previously tracked by scripts/run_empowerment_dv_resume.sh
# and scripts/run_empowerment_crl_flowbc_resume.sh.
#
# Submit from impls/:  sbatch scripts/run_empowerment_dv_flowbc_priority.sh
#
# Flags match the original launch in scripts/run_empowerment_comparison_navigate.sh
# (sweep IDX 0,2,3,5): lr 3e-4, batch 1024, (512,512,512) trunks, layer_norm,
# discount 0.99, seed 0, train_steps 1e6.
#
#   IDX 0 : empowerment_dv          -- antsoccer-arena-navigate-v0
#   IDX 1 : empowerment_dv          -- antmaze-medium-navigate-v0
#   IDX 2 : empowerment_crl_flowbc  -- antsoccer-arena-navigate-v0
#   IDX 3 : empowerment_crl_flowbc  -- antmaze-medium-navigate-v0

IDX=${SLURM_ARRAY_TASK_ID}

RUN_AGENTS=(
    empowerment_dv           # 0 -- antsoccer-arena-navigate-v0
    empowerment_dv           # 1 -- antmaze-medium-navigate-v0
    empowerment_crl_flowbc   # 2 -- antsoccer-arena-navigate-v0
    empowerment_crl_flowbc   # 3 -- antmaze-medium-navigate-v0
)
RUN_ENVS=(
    antsoccer-arena-navigate-v0   # 0
    antmaze-medium-navigate-v0    # 1
    antsoccer-arena-navigate-v0   # 2
    antmaze-medium-navigate-v0    # 3
)
SEED=0

if [ -z "$IDX" ] || [ "$IDX" -ge ${#RUN_AGENTS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RUN_AGENTS[@]} runs; use --array=0-$((${#RUN_AGENTS[@]} - 1))." >&2
    exit 1
fi
AGENT_NAME=${RUN_AGENTS[$IDX]}
ENV=${RUN_ENVS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  AGENT=$AGENT_NAME  ENV=$ENV  SEED=$SEED"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --env_name=$ENV \
    --agent=agents/"$AGENT_NAME".py \
    --seed=$SEED \
    --train_steps=1000000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR
