#!/bin/bash
#SBATCH --job-name=emp_skill_extra_envs_priority
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-13

# High-priority (rail_gpu4_high) empowerment_skill launch for envs not covered
# by scripts/empowerment_skill_4env_noise_bc_sweep.sh. Same agent flags/
# structure as that sweep (num_skills, bc_alpha, action_noise_std,
# stochastic_policy_actions, perturb_q_loss_actions, log_interval=8000,
# train_steps=1e6, video_episodes=0), but a curated list of runs instead of a
# full grid:
#
#   IDX 0  : antsoccer-arena-navigate-v0    k=50  noise=0.01  bc=0.001
#   IDX 1  : antsoccer-arena-stitch-v0      k=50  noise=0.01  bc=0.001
#   IDX 2  : pointmaze-teleport-navigate-v0 k=50  noise=0.01  bc=0.001
#   IDX 3  : pointmaze-teleport-stitch-v0   k=50  noise=0.01  bc=0.001
#   IDX 4-8 : cube-double-play-v0           k=50  noise=0.01  bc in [0.001, 0.01, 0.1, 1, 10]
#   IDX 9-13: cube-single-play-v0           k=50  noise=0.01  bc in [0.001, 0.01, 0.1, 1, 10]
#
# perturb_q_loss_actions=True for every run above.
#
# Submit from impls/:  sbatch scripts/run_empowerment_skill_extra_envs_priority.sh

IDX=${SLURM_ARRAY_TASK_ID}

RUN_ENVS=(
    antsoccer-arena-navigate-v0     # 0
    antsoccer-arena-stitch-v0       # 1
    pointmaze-teleport-navigate-v0  # 2
    pointmaze-teleport-stitch-v0    # 3
    cube-double-play-v0             # 4
    cube-double-play-v0             # 5
    cube-double-play-v0             # 6
    cube-double-play-v0             # 7
    cube-double-play-v0             # 8
    cube-single-play-v0             # 9
    cube-single-play-v0             # 10
    cube-single-play-v0             # 11
    cube-single-play-v0             # 12
    cube-single-play-v0             # 13
)
RUN_BC_ALPHAS=(
    0.001  # 0
    0.001  # 1
    0.001  # 2
    0.001  # 3
    0.001  # 4
    0.01   # 5
    0.1    # 6
    1      # 7
    10     # 8
    0.001  # 9
    0.01   # 10
    0.1    # 11
    1      # 12
    10     # 13
)
SKILLS=50
NOISE=0.01

if [ -z "$IDX" ] || [ "$IDX" -ge ${#RUN_ENVS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RUN_ENVS[@]} runs; use --array=0-$((${#RUN_ENVS[@]} - 1))." >&2
    exit 1
fi
ENV=${RUN_ENVS[$IDX]}
BC_ALPHA=${RUN_BC_ALPHAS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  SKILLS=$SKILLS  NOISE=$NOISE  BC_ALPHA=$BC_ALPHA"

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
    --agent.stochastic_policy_actions=True \
    --agent.action_noise_std=$NOISE \
    --agent.perturb_q_loss_actions=True \
    --agent.log_interval=8000 \
    --log_interval=8000 \
    --train_steps=1000000 \
    --video_episodes=0
