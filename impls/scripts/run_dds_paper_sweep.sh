#!/bin/bash
#SBATCH --job-name=dds_paper_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-7

# Discrete Diffusion Skills (DDS) — faithful OGBench re-implementation of
# "Offline RL with Discrete Diffusion Skills" (arXiv:2503.20176), trained purely
# offline (agents/dds.py). This sweep runs DDS on the four antmaze/antsoccer datasets:
#     antmaze-medium-navigate-v0     (continuous action, diffusion decoder)
#     antsoccer-arena-navigate-v0    (continuous action, diffusion decoder)
#     antmaze-medium-stitch-v0       (continuous action, diffusion decoder)
#     antsoccer-arena-stitch-v0      (continuous action, diffusion decoder)
#
# ── Paper setup (already the agents/dds.py get_config() defaults; left unchanged) ─
#   skill_dim D_z = 128, commitment_beta = 0.25, subgoal_steps H = 10
#   (sequence_length = H), transformer encoder 256/4-layer/8-head, diffusion
#   decoder 256/4-block/4x-expand, diffusion_steps = 5, time_dim = 16,
#   beta_min/max = 0.1/10, value/actor hidden = (256,256), discount = 0.99,
#   tau = 0.005, expectile (tau_IQL) = 0.7, AWR alpha = 3.0,
#   skill_pretrain_steps = 500000.
#   train_steps = 1,000,000 (the default) = 500k skill VQ-VAE pretrain + hard
#   freeze + 500k high-level (semi-MDP IQL value/critic + AWR code policy) — the
#   paper's relabel-then-train budget for a single OGBench run (see dds.py B4).
#   All three envs are continuous-action, so DDS uses its diffusion action
#   decoder (discrete=False, the default).
#
# ── What is swept ───────────────────────────────────────────────────────────
#   The one hyperparameter the DDS paper sweeps is the codebook size K
#   (num_skills): paper default 16, ablated over 4-32 (dds.py: "swept 4-32").
#   We sweep K in {15, 50}. Everything else is held at the paper defaults.
#   Single seed (0); no seed sweep.
#
#   Full sweep = 4 envs x 2 K-values = 8 runs  ->  --array=0-7
#
# Index decoding (ENV outer, K inner):
#   IDX     = SLURM_ARRAY_TASK_ID              (0..7)
#   K_IDX   = IDX % 2                           (0..1)
#   ENV_IDX = IDX / 2                           (0..3)

IDX=${SLURM_ARRAY_TASK_ID}

# ── Sweep definitions ─────────────────────────────────────────────────────────
ENVS=(
    antmaze-medium-navigate-v0
    antsoccer-arena-navigate-v0
    antmaze-medium-stitch-v0
    antsoccer-arena-stitch-v0
)
NUM_SKILLS=(15 50)   # codebook size K (paper default 16; ablated 4-32)
SEED=0

K_IDX=$((IDX % 2))
ENV_IDX=$((IDX / 2))

ENV=${ENVS[$ENV_IDX]}
K=${NUM_SKILLS[$K_IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench
RUN_GROUP="dds_${ENV}_K${K}"

echo "IDX=$IDX  ENV=$ENV  num_skills(K)=$K  SEED=$SEED  RUN_GROUP=$RUN_GROUP"

# ── Run ───────────────────────────────────────────────────────────────────────
# mujoco rendering uses EGL; local wandb data goes to scratch (home quota is
# small). Eval videos are disabled (--video_episodes=0).
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

python main.py \
    --env_name=$ENV \
    --agent=agents/dds.py \
    --agent.num_skills=$K \
    --seed=$SEED \
    --train_steps=1000000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR \
    --run_group=$RUN_GROUP
