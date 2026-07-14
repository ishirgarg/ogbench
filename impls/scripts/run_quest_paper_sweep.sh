#!/bin/bash
#SBATCH --job-name=quest_paper_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-9

# QueST — faithful OGBench re-implementation of "QueST: Self-Supervised Skill
# Abstractions for Learning Continuous Control" (Mete et al., NeurIPS 2024,
# arXiv:2407.15840), trained purely offline (agents/quest.py). This sweep runs
# QueST on the same five OGBench datasets as the DDS sweep:
#     antmaze-medium-navigate-v0     (continuous action)
#     antsoccer-arena-navigate-v0    (continuous action)
#     pointmaze-teleport-navigate-v0 (continuous action)
#     antmaze-medium-stitch-v0       (continuous action)
#     antsoccer-arena-stitch-v0      (continuous action)
# All are continuous-action, state-based, goal-conditioned tasks — QueST's
# target setting (discrete=False, goal_conditioned=True, no visual encoder).
#
# ── Paper setup (agents/quest.py get_config() defaults; left unchanged) ────────
#   Stage I (action autoencoder, Table 3): horizon_length T = 32,
#   downsample_factor F = 4 -> num_tokens n = 8; FSQ levels (8,5,5,5) ->
#   vocab 1000; ae_dim 256; causal conv kernels (5,3,3)/strides (2,2,1) +
#   GroupNorm(8)+Mish; encoder 2-layer/4-head, decoder 4-layer/4-head; masked
#   (unmasked-over-repeat-padded) L1 reconstruction only (FSQ needs no VQ loss).
#   Stage II (autoregressive skill prior, Table 4): prior_dim 384,
#   6-layer/6-head causal transformer, token cross-entropy; top_k 5.
#   Dropout 0.1 (attn + embd). Optimizer: AdamW(lr 1e-4, wd 1e-4, betas
#   0.9/0.999, no decay on norms/biases/embeddings), CosineAnnealingLR -> 1e-5,
#   global-norm grad clip 100. batch_size: 256 for Stage I autoencoder
#   (config/train_autoencoder.yaml), 128 for Stage II prior (config/train_base.yaml).
#
#   Each array task runs the two stages as TWO SEQUENTIAL python commands in the
#   SAME job — exactly like the official pipeline (autoencoder.sh then main.sh):
#     Stage I (--agent.stage=ae):   500k autoencoder-only steps, checkpointed.
#     Stage II (--agent.stage=prior): 500k prior-only steps, restoring ONLY the
#         trained autoencoder params and building a FRESH optimizer.
#   This gives a genuinely fresh Adam optimizer for the prior (no bias-correction
#   transient), reproducing the official separate stage-0/stage-1 jobs bit-exactly.
#   Total compute = 1M steps, same as a single gated run. (A single-command
#   equivalent is --agent.stage=both with --agent.stage1_steps=500000, which is
#   faithful up to one bounded ~1000-step boundary transient.)
#
# ── What is swept ──────────────────────────────────────────────────────────────
#   The DDS analog of "number of skills" is QueST's FSQ codebook size (the number
#   of discrete skill TOKENS; note a QueST skill is a length-n=8 SEQUENCE of these
#   tokens, so the representable-skill count is codebook_size^n). We sweep it over
#   {15, 50} to match the DDS K sweep, via the --agent.codebook_size knob
#   (agents/quest.py get_fsq_level: 15 -> FSQ levels [5,3]; 50 -> [10,5], the exact
#   product-50 factorization with all levels >= 3, since FSQ's bound is undefined
#   at level 2). Everything else is held at the paper defaults. Single seed (0).
#
#   Full sweep = 5 envs x 2 codebook sizes = 10 runs  ->  --array=0-9
#
# Index decoding (ENV outer, K inner), mirroring the DDS script:
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
CODEBOOK_SIZES=(15 50)   # FSQ codebook size = number of skill tokens (DDS-style K sweep)
SEED=0

K_IDX=$((IDX % 2))
ENV_IDX=$((IDX / 2))

ENV=${ENVS[$ENV_IDX]}
K=${CODEBOOK_SIZES[$K_IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench
AE_GROUP="quest_ae_${ENV}_K${K}"        # Stage I run group (autoencoder)
RUN_GROUP="quest_${ENV}_K${K}"          # Stage II run group (prior; the eval run)

echo "IDX=$IDX  ENV=$ENV  codebook_size(K)=$K  SEED=$SEED  RUN_GROUP=$RUN_GROUP"

# ── Env ─────────────────────────────────────────────────────────────────────
# mujoco rendering uses EGL; local wandb data goes to scratch (home quota is
# small). Eval videos are disabled (--video_episodes=0). total_steps must equal
# each stage's train_steps so the per-stage cosine schedule reaches eta_min.
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e   # if Stage I fails, do not launch Stage II with no checkpoint to restore

# ── Stage I — autoencoder (official stage 0) ───────────────────────────────────
# Saves exactly one checkpoint (params_500000.pkl) at the end of the run under
# $SAVE_DIR/OGBench/$AE_GROUP/<exp_name>/  (exp_name embeds this job's SLURM_JOB_ID).
python main.py \
    --env_name=$ENV \
    --agent=agents/quest.py \
    --agent.codebook_size=$K \
    --agent.stage=ae \
    --agent.batch_size=256 \
    --agent.total_steps=500000 \
    --seed=$SEED \
    --train_steps=500000 \
    --eval_interval=1000000 \
    --save_interval=500000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR \
    --run_group=$AE_GROUP

# ── Stage II — prior (official stage 1) ────────────────────────────────────────
# Restore ONLY the trained autoencoder params (fresh prior + fresh optimizer).
# The restore path globs this job's Stage-I exp dir; quoting keeps the shell from
# expanding '*' so the agent's glob.glob resolves it (asserts a single match).
# SLURM_JOB_ID is shared by both commands in this job, so it uniquely pins the
# Stage-I run even across re-runs of the same array task.
python main.py \
    --env_name=$ENV \
    --agent=agents/quest.py \
    --agent.codebook_size=$K \
    --agent.stage=prior \
    --agent.total_steps=500000 \
    --agent.restore_ae_path="$SAVE_DIR/OGBench/$AE_GROUP/*${SLURM_JOB_ID}*" \
    --agent.restore_ae_epoch=500000 \
    --seed=$SEED \
    --train_steps=500000 \
    --video_episodes=0 \
    --save_dir=$SAVE_DIR \
    --run_group=$RUN_GROUP
