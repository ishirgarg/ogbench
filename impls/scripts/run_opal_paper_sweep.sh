#!/bin/bash
#SBATCH --job-name=opal_paper_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-4

# OPAL — faithful OGBench re-implementation of the OFFLINE VAE skill-controller
# pretraining used by SUPE ("Leveraging Skills from Unlabeled Prior Data for
# Efficient Online Exploration", arXiv:2410.18076; VAE objective from OPAL,
# Ajay et al. 2021). Ported from the SUPE repo's supe/pretraining/opal.py +
# configs/opal_config.py + run_opal.py (agents/opal.py). Trained purely offline.
# Only the low-level skill controller (the VAE) is learned; the SUPE/OPAL IQL
# high-level skill policy ("reward labeling") and online exploration are omitted.
#
# This sweep runs OPAL on the SAME five OGBench datasets as the QueST sweep:
#     antmaze-medium-navigate-v0     (continuous action)
#     antsoccer-arena-navigate-v0    (continuous action)
#     pointmaze-teleport-navigate-v0 (continuous action)
#     antmaze-medium-stitch-v0       (continuous action)
#     antsoccer-arena-stitch-v0      (continuous action)
# All are continuous-action, state-based tasks — OPAL's target setting
# (discrete=False, no visual encoder).
#
# ── Faithful hyperparameters (agents/opal.py get_config() defaults) ────────────
#   Skill latent dim skill_dim = 8; KL coefficient kl_coef = 0.1 (the value SUPE
#   uses for antmaze/antsoccer/pointmaze navigate; source opal_config.py + README);
#   chunk / option horizon c = 4 (source run_opal.py horizon_length). VAE nets use
#   the OGBench widths SUPE's run_opal.py sets for `is_ogbench`:
#   vae_encoder_hidden_size = 512, vae_hidden_dims = (512,512,512) (base D4RL run
#   uses 256 / (256,256)). Posterior q(z|tau): 2-layer bidirectional-GRU sequence
#   encoder over [MLP(s_i), a_i] tokens. Prior p(z|s_1) and decoder pi(a|s,z):
#   Gaussian MLP heads (log_std clipped [-20,2] for prior/decoder; posterior
#   log_std raw). Loss: -log pi(a|s,z) reconstruction + kl_coef * KL(q||p).
#   Optimizer: Adam(lr = 3e-4). batch_size = 256. Single-stage, 1M steps (source
#   run_opal.py max_steps), matching the QueST sweep's total 1M-step budget.
#
#   Unlike QueST (a two-stage AE-then-prior pipeline), OPAL trains the whole VAE
#   jointly in ONE stage, so this is a single python command per run.
#
# ── What is swept ──────────────────────────────────────────────────────────────
#   Just the five envs (single seed 0, all other hyperparameters at the faithful
#   OPAL defaults).  Full sweep = 5 envs  ->  --array=0-4
#
# Index decoding:
#   IDX = SLURM_ARRAY_TASK_ID   (0..4)  -> ENV index

IDX=${SLURM_ARRAY_TASK_ID}

# ── Sweep definitions ─────────────────────────────────────────────────────────
ENVS=(
    antmaze-medium-navigate-v0
    antsoccer-arena-navigate-v0
    pointmaze-teleport-navigate-v0
    antmaze-medium-stitch-v0
    antsoccer-arena-stitch-v0
)
SEED=0

ENV=${ENVS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench
RUN_GROUP="opal_${ENV}"

echo "IDX=$IDX  ENV=$ENV  SEED=$SEED  RUN_GROUP=$RUN_GROUP"

# ── Env ─────────────────────────────────────────────────────────────────────
# mujoco rendering uses EGL; local wandb data goes to scratch (home quota is
# small). Eval videos are disabled (--video_episodes=0).
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e

# ── Offline OPAL VAE skill-controller pretraining (single stage, 1M steps) ─────
python main.py \
    --env_name=$ENV \
    --agent=agents/opal.py \
    --seed=$SEED \
    --train_steps=1000000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR \
    --run_group=$RUN_GROUP
