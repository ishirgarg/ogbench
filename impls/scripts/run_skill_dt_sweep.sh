#!/bin/bash
#SBATCH --job-name=skill_dt_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-11

# Skill Decision Transformer (agents/skill_dt.py, arXiv:2301.13573) on the
# six locomotion datasets of run_empowerment_skill_extra_envs_priority.sh
# (cube-* dropped) plus antmaze-medium-{navigate,stitch}, at 15 and 50 skills.
# High-priority (rail_gpu4_high) queue.
#
#   6 envs x 2 codebook sizes, seed 0.  ENV = IDX / 2, SKILLS = IDX % 2.
#
#     IDX 0  : antsoccer-arena-navigate-v0     15 skills
#     IDX 1  : antsoccer-arena-navigate-v0     50 skills
#     IDX 2  : antsoccer-arena-stitch-v0       15 skills
#     IDX 3  : antsoccer-arena-stitch-v0       50 skills
#     IDX 4  : pointmaze-teleport-navigate-v0  15 skills
#     IDX 5  : pointmaze-teleport-navigate-v0  50 skills
#     IDX 6  : pointmaze-teleport-stitch-v0    15 skills
#     IDX 7  : pointmaze-teleport-stitch-v0    50 skills
#     IDX 8  : antmaze-medium-navigate-v0      15 skills
#     IDX 9  : antmaze-medium-navigate-v0      50 skills
#     IDX 10 : antmaze-medium-stitch-v0        15 skills
#     IDX 11 : antmaze-medium-stitch-v0        50 skills
#
# Submit from impls/:  sbatch scripts/run_skill_dt_sweep.sh
#
# READING THE RESULTS -- Skill DT is reward-free and goal-agnostic, so the
# `evaluation/*_success` that main.py logs is an average over a RANDOMLY drawn
# skill per episode. It is near-zero by design and is NOT the paper's number:
# Sec. 5.2 reports the MAX over a sweep of every skill. Get that after training
# with
#     python eval_skill_policy.py --run_dir <run>
# which rolls out each codebook entry against each task. During training watch
# `training/policy/action_mse` (behaviour cloning fit) and
# `training/vq/codebook_perplexity` (how many of the num_skills codes are
# actually in use -- if it collapses toward 1, the codebook is dead).

IDX=${SLURM_ARRAY_TASK_ID}

ENVS=(
    antsoccer-arena-navigate-v0     # IDX 0-1
    antsoccer-arena-stitch-v0       # IDX 2-3
    pointmaze-teleport-navigate-v0  # IDX 4-5
    pointmaze-teleport-stitch-v0    # IDX 6-7
    antmaze-medium-navigate-v0      # IDX 8-9
    antmaze-medium-stitch-v0        # IDX 10-11
)
SKILLS=(15 50)
SEED=0

if [ -z "$IDX" ] || [ "$IDX" -ge $((${#ENVS[@]} * ${#SKILLS[@]})) ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for $((${#ENVS[@]} * ${#SKILLS[@]})) runs; use --array=0-$((${#ENVS[@]} * ${#SKILLS[@]} - 1))." >&2
    exit 1
fi
ENV=${ENVS[$((IDX / 2))]}
NUM_SKILLS=${SKILLS[$((IDX % 2))]}

# Skill DT's conditioning statistic Z_t is normalized over the steps remaining
# in the TRAJECTORY at training time, but over the steps remaining in the
# EPISODE at rollout time (paper Sec. A.5). Those coincide on -navigate, whose
# trajectories are 1001 steps against a 1000-step horizon, so the default (None
# -> follow the env) is right there. They do NOT coincide on -stitch, whose
# trajectories are 201 steps (antmaze, antsoccer-arena and pointmaze alike):
# leaving it at 1000 would feed the policy a statistic it never saw in
# training. Pin it to the trajectory length instead.
case "$ENV" in
    *-stitch-*) EVAL_MAX_STEPS_FLAG=(--agent.eval_max_steps=201) ;;
    *)          EVAL_MAX_STEPS_FLAG=() ;;
esac

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  NUM_SKILLS=$NUM_SKILLS  SEED=$SEED"

# Everything not listed here stays at the paper's Table 5 values (4 layers,
# 4 heads, embed 256, context K=20, dropout 0.0, batch 256, lr 1e-4, grad-norm
# 0.25) and at relabel_interval=50, Alg. 1's hindsight re-labelling period,
# which costs roughly 10% wall clock on a 1M-state dataset.
AGENT_FLAGS=(
    --agent.num_skills=$NUM_SKILLS
    "${EVAL_MAX_STEPS_FLAG[@]}"
)

export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e

# The in-training eval is only a smoke test of the rollout path (see above), so
# it is kept cheap: 5 episodes/task, on GPU. Skill DT rolls a 20-step
# Transformer context every env step, which is far heavier than the MLP policies
# main.py's defaults (20 episodes, on CPU) were sized for.
python main.py \
    --env_name=$ENV \
    --agent=agents/skill_dt.py \
    "${AGENT_FLAGS[@]}" \
    --seed=$SEED \
    --train_steps=1000000 \
    --video_episodes=0 \
    --eval_episodes=5 \
    --eval_on_cpu=0 \
    --save_interval=25000 \
    --save_dir=$SAVE_DIR
