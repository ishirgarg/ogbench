#!/bin/bash
#SBATCH --job-name=emp_dads_antsoccer_navigate_kl1
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00

# empowerment_dads (MI-maximizing label assignment, Barber-Agakov bracket) on
# antsoccer-arena-navigate-v0 only, from scratch. Identical to IDX=4 of
# run_empowerment_dads_kl_sweep.sh (num_skills=15, kl_coef=1.0, seed 0, 1M
# train steps, save_interval 25k, and everything else at agents/
# empowerment_dads.py's defaults: lr 3e-4, batch 1024, (512,512,512) trunks,
# layer_norm, discount 0.99) except that it runs on its own on the
# normal-priority queue rather than as part of the lowest-priority array.
#
# This is a fresh run, NOT a resume: it starts at step 0 in a new run folder
# and opens a new wandb run.
#
# Submit from impls/:  sbatch scripts/run_empowerment_dads_antsoccer_navigate_kl1.sh

ENV=antsoccer-arena-navigate-v0
SEED=0
NUM_SKILLS=15
KL=1.0
AGENT=empowerment_dads

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "ENV=$ENV  AGENT=$AGENT  KL_COEF=$KL  NUM_SKILLS=$NUM_SKILLS  SEED=$SEED"

export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e

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
