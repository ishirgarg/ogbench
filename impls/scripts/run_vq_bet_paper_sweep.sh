#!/bin/bash
#SBATCH --job-name=vq_bet_paper_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-9

# VQ-BeT — faithful OGBench re-implementation of "Behavior Generation with Latent
# Actions" (Lee et al., ICML 2024, arXiv:2403.03181), ported 1:1 from the official
# repo (vq_bet_official/) and trained purely offline (agents/vq_bet.py). This sweep
# runs VQ-BeT on the SAME five OGBench datasets as the QueST sweep:
#     antmaze-medium-navigate-v0     (continuous action)
#     antsoccer-arena-navigate-v0    (continuous action)
#     pointmaze-teleport-navigate-v0 (continuous action)
#     antmaze-medium-stitch-v0       (continuous action)
#     antsoccer-arena-stitch-v0      (continuous action)
# All are continuous-action, state-based, goal-conditioned tasks — VQ-BeT's target
# setting (discrete=False, goal_conditioned=True, no visual encoder).
#
# ── Paper setup (agents/vq_bet.py get_config() defaults; from the official
#    examples/configs/*ant*.yaml — the closest analogue to OGBench navigation) ────
#   Tokenizer (VqVae, pretrain_ant.yaml): EncoderMLP (hidden 128, ReLU, orthogonal
#   init) encoder/decoder; ResidualVQ with vqvae_groups=2 EMA Euclidean codebooks
#   (decay 0.8), n_latent_dims=512, act_window_size=1; L1 reconstruction
#   (encoder_loss_multiplier=0.033) + 5x commitment; Adam(lr 1e-3, wd 1e-4).
#   Transformer (BehaviorTransformer, train_ant_goalcond.yaml): nanoGPT
#   6-layer/6-head/120-embd (GPT-2 init, dropout 0.1, output_dim 256); flat G*C
#   code head + per-(group,code) offset head (MLP [1024,1024,*], ReLU); FocalLoss
#   gamma=2, code loss = 5*primary + 3*secondary (secondary_code_multiplier),
#   offset_loss_multiplier=0.1; AdamW(lr 5.5e-5, wd 2e-4, betas 0.9/0.999).
#   batch_size 1024.
#
#   Each array task runs the two stages as TWO SEQUENTIAL python commands in the
#   SAME job — exactly like the official pipeline (pretrain_vqvae.py then train.py):
#     Stage I  (--agent.stage=vqvae): 500k tokenizer-only steps, checkpointed.
#     Stage II (--agent.stage=bet):   500k transformer steps, restoring ONLY the
#         frozen tokenizer. Within this stage the official 50/50 split is applied
#         via --agent.bet_stage1_steps=250000: the GPT + code-prediction head train
#         for the first 250k steps, then ONLY the offset head trains for the last
#         250k (train.py L177-191). The tokenizer stays fully frozen throughout.
#   Total compute = 1M steps, same budget as the QueST sweep.
#
# ── What is swept ──────────────────────────────────────────────────────────────
#   The "number of skills" knob is VQ-BeT's per-codebook code count
#   (vqvae_n_embed), exposed here as --agent.num_skills (a VQ-BeT skill is one
#   discrete RVQ code; with vqvae_groups=2 the representable-skill count is
#   num_skills^2). We sweep it over {15, 50} to mirror the QueST/DDS K sweep for a
#   like-for-like comparison. Everything else is held at the paper defaults
#   (num_skills defaults to 10, the ant value). Single seed (0).
#
#   Full sweep = 5 envs x 2 skill counts = 10 runs  ->  --array=0-9
#
# Index decoding (ENV outer, K inner), mirroring the QueST script:
#   IDX     = SLURM_ARRAY_TASK_ID              (0..9)
#   K_IDX   = IDX % 2                           (0..1)
#   ENV_IDX = IDX / 2                           (0..4)

IDX=${SLURM_ARRAY_TASK_ID}

# ── Sweep definitions ─────────────────────────────────────────────────────────
ENVS=(
    antmaze-medium-navigate-v0
    antsoccer-arena-navigate-v0
    pointmaze-teleport-navigate-v0
    antmaze-medium-stitch-v0
    antsoccer-arena-stitch-v0
)
NUM_SKILLS=(15 50)   # VQ-BeT per-codebook code count = "number of skills" (QueST-style K sweep)
SEED=0

K_IDX=$((IDX % 2))
ENV_IDX=$((IDX / 2))

ENV=${ENVS[$ENV_IDX]}
K=${NUM_SKILLS[$K_IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench
VQVAE_GROUP="vq_bet_vqvae_${ENV}_K${K}"   # Stage I run group (tokenizer)
RUN_GROUP="vq_bet_${ENV}_K${K}"           # Stage II run group (transformer; the eval run)

echo "IDX=$IDX  ENV=$ENV  num_skills(K)=$K  SEED=$SEED  RUN_GROUP=$RUN_GROUP"

# ── Env ─────────────────────────────────────────────────────────────────────
# mujoco rendering uses EGL; local wandb data goes to scratch (home quota is
# small). Eval videos are disabled (--video_episodes=0).
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e   # if Stage I fails, do not launch Stage II with no tokenizer to restore

# ── Stage I — VQ-VAE tokenizer (official pretrain_vqvae.py) ─────────────────────
# Trains ONLY the encoder/decoder (Adam lr=1e-3) with the EMA codebook; the
# transformer optimizer is frozen. Saves exactly one checkpoint (params_500000.pkl)
# under $SAVE_DIR/OGBench/$VQVAE_GROUP/<exp_name>/ (exp_name embeds SLURM_JOB_ID).
# Eval is skipped during this stage (the policy is untrained here).
python main.py \
    --env_name=$ENV \
    --agent=agents/vq_bet.py \
    --agent.num_skills=$K \
    --agent.stage=vqvae \
    --seed=$SEED \
    --train_steps=500000 \
    --eval_interval=1000000 \
    --save_interval=500000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR \
    --run_group=$VQVAE_GROUP

# ── Stage II — transformer (official train.py) ─────────────────────────────────
# Restore ONLY the trained tokenizer (fresh transformer + fresh optimizer). The
# restore path globs this job's Stage-I exp dir; quoting keeps the shell from
# expanding '*' so the agent's glob.glob resolves it (asserts a single match).
# SLURM_JOB_ID is shared by both commands in this job, so it uniquely pins the
# Stage-I run even across re-runs of the same array task.
python main.py \
    --env_name=$ENV \
    --agent=agents/vq_bet.py \
    --agent.num_skills=$K \
    --agent.stage=bet \
    --agent.bet_stage1_steps=250000 \
    --agent.restore_vqvae_path="$SAVE_DIR/OGBench/$VQVAE_GROUP/*${SLURM_JOB_ID}*" \
    --agent.restore_vqvae_epoch=500000 \
    --seed=$SEED \
    --train_steps=500000 \
    --eval_temperature=1.0 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR \
    --run_group=$RUN_GROUP
