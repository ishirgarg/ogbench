#!/bin/bash
#SBATCH --job-name=dds_50skills
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-5

# Discrete Diffusion Skills (DDS) -- faithful OGBench re-implementation of
# "Offline RL with Discrete Diffusion Skills" (arXiv:2503.20176), trained purely
# offline (agents/dds.py). Same six locomotion datasets and same launch shape as
# run_opal_discrete_50skills.sh, with codebook size K = num_skills = 50.
#
# -- Paper setup (already the agents/dds.py get_config() defaults; left unchanged) -
#   skill_dim D_z = 128, commitment_beta = 0.25, subgoal_steps H = 10
#   (sequence_length = H), transformer encoder 256/4-layer/8-head, diffusion
#   decoder 256/4-block/4x-expand, diffusion_steps = 5, time_dim = 16,
#   beta_min/max = 0.1/10, value/actor hidden = (256,256), discount = 0.99,
#   tau = 0.005, expectile (tau_IQL) = 0.7, AWR alpha = 3.0,
#   skill_pretrain_steps = 500000, batch_size = 256 (IQL, Table 7) and
#   skill_batch_size = 128 (VQ-VAE, Table 6) -- so no --batch_size override here.
#   train_steps = 1,000,000 = 500k skill VQ-VAE pretrain + hard freeze + 500k
#   high-level (semi-MDP IQL value/critic + AWR code policy) -- the paper's
#   relabel-then-train budget for a single OGBench run (see dds.py B4).
#   All six envs are continuous-action, so DDS uses its diffusion action
#   decoder (discrete=False, the default).
#
#   The one hyperparameter the DDS paper sweeps is the codebook size K
#   (num_skills): paper default 16, ablated over 4-32. We pin K = 50 to match
#   the 50-skill OPAL/empowerment runs; single seed (0), log_interval 8000.
#
#   IDX 0 : antsoccer-arena-navigate-v0
#   IDX 1 : antsoccer-arena-stitch-v0
#   IDX 2 : pointmaze-teleport-navigate-v0
#   IDX 3 : pointmaze-teleport-stitch-v0
#   IDX 4 : antmaze-medium-navigate-v0
#   IDX 5 : antmaze-medium-stitch-v0
#
#   IDX = SLURM_ARRAY_TASK_ID   (0..5)
# Submit from impls/:  sbatch scripts/run_dds_50skills.sh

IDX=${SLURM_ARRAY_TASK_ID}

ENVS=(
    antsoccer-arena-navigate-v0     # 0
    antsoccer-arena-stitch-v0       # 1
    pointmaze-teleport-navigate-v0  # 2
    pointmaze-teleport-stitch-v0    # 3
    antmaze-medium-navigate-v0      # 4
    antmaze-medium-stitch-v0        # 5
)
SEED=0
K=50   # codebook size (num_skills)

if [ -z "$IDX" ] || [ "$IDX" -ge ${#ENVS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#ENVS[@]} runs; use --array=0-$((${#ENVS[@]} - 1))." >&2
    exit 1
fi
ENV=${ENVS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench

echo "IDX=$IDX  ENV=$ENV  num_skills(K)=$K  SEED=$SEED"

export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e

python main.py \
    --env_name=$ENV \
    --agent=agents/dds.py \
    --agent.num_skills=$K \
    --seed=$SEED \
    --train_steps=1000000 \
    --log_interval=8000 \
    --video_episodes=0 \
    --save_interval=25000 \
    --save_dir=$SAVE_DIR
