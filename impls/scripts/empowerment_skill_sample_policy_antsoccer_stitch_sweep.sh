#!/bin/bash
#SBATCH --job-name=emp_skill_antsoccer_stitch_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-15

IDX=${SLURM_ARRAY_TASK_ID}

# -----------------------------
# Sweep definitions
# -----------------------------
ENV=antsoccer-arena-stitch-v0
BC_ALPHAS=(0.0003 0.001 0.003 0.01)
ACTION_NOISE_STD=(0.00003 0.0001)
PERTURB_Q_LOSS_ACTIONS=(True False)
SKILLS=15

# Decode index: BC_ALPHA outer, ACTION_NOISE_STD middle, PERTURB_Q_LOSS_ACTIONS inner.
BC_INDEX=$((IDX / 4))
REMAINDER=$((IDX % 4))
NOISE_INDEX=$((REMAINDER / 2))
PERTURB_INDEX=$((REMAINDER % 2))

BC_ALPHA=${BC_ALPHAS[$BC_INDEX]}
NOISE=${ACTION_NOISE_STD[$NOISE_INDEX]}
PERTURB_Q_LOSS_ACTIONS=${PERTURB_Q_LOSS_ACTIONS[$PERTURB_INDEX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

# -----------------------------
# Run
# -----------------------------
export MUJOCO_GL=egl

python main.py \
    --env_name=$ENV \
    --save_dir=$SAVE_DIR \
    --agent=agents/empowerment_skill.py \
    --agent.num_skills=$SKILLS \
    --agent.bc_alpha=$BC_ALPHA \
    --agent.action_noise_std=$NOISE \
    --agent.perturb_q_loss_actions=$PERTURB_Q_LOSS_ACTIONS \
    --agent.stochastic_policy_actions=True \
    --train_steps=1000000 \
    --video_episodes=0
