#!/bin/bash
#SBATCH --job-name=emp_opal_dads_50skills
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_lowest
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-3

# empowerment_opal_dads (discrete-skill stochastic EM + BA bracket) on the two
# navigate envs and their stitch counterparts, matched to
# run_empowerment_dads_navigate.sh / _comparison (lr 3e-4, batch 1024,
# (512,512,512) trunks, layer_norm, discount 0.99) but with 50 skills instead
# of the usual 15. Lowest-priority queue.
#
#   IDX 0 : antsoccer-arena-navigate-v0
#   IDX 1 : antmaze-medium-navigate-v0
#   IDX 2 : antsoccer-arena-stitch-v0
#   IDX 3 : antmaze-medium-stitch-v0
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..3)

IDX=${SLURM_ARRAY_TASK_ID}

# ── Sweep definitions ─────────────────────────────────────────────────────────
RUN_ENVS=(
    antsoccer-arena-navigate-v0   # 0
    antmaze-medium-navigate-v0    # 1
    antsoccer-arena-stitch-v0     # 2
    antmaze-medium-stitch-v0      # 3
)
SEED=0

ENV=${RUN_ENVS[$IDX]}
AGENT=empowerment_opal_dads

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  AGENT=$AGENT  SEED=$SEED"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --env_name=$ENV \
    --agent=agents/${AGENT}.py \
    --agent.num_skills=50 \
    --seed=$SEED \
    --train_steps=1000000 \
    --save_interval=25000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR
