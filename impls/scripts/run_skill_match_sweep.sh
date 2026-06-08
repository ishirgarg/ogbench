#!/usr/bin/env bash
# Local (non-SLURM) version of scripts/skill_match_navigate.sh.
# Runs at most one job per GPU at a time (GPU-token semaphore), up to one job
# per GPU in parallel; the next queued job starts as soon as a GPU frees up.
#
# Sweep: 4 (env, pretrained empowerment_skill checkpoint) configs x 3 AWR
# temperatures (alpha) = 12 jobs. num_skills is auto-read from each checkpoint's
# flags.json, so the K=15 and K=50 checkpoints can be mixed.

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export WANDB_API_KEY='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'
export MUJOCO_GL=egl

# -----------------------------
# Machine / sweep configuration
# -----------------------------
# GPU ids to use. At most one job runs on a GPU at a time; extra jobs wait in a
# queue until a GPU frees up (no oversubscription). With a single GPU listed,
# all jobs run sequentially on it.
GPUS=(0 1 2 3 4 6 7)

CKPT_PREFIX=/home/ishirgarg/ogbench/impls/ckpts
SAVE_DIR=/home/ishirgarg/ogbench/
LOG_DIR=logs/skill_match_sweep
mkdir -p "$LOG_DIR"

ENVS=(
    antmaze-medium-navigate-v0
    antmaze-medium-navigate-v0
    antsoccer-arena-navigate-v0
    antsoccer-arena-navigate-v0
)
# Pretrained empowerment_skill run dirs (match ENVS by index).
SKILL_CKPTS=(
    $CKPT_PREFIX/antmaze-medium-navigate/sd000_s_34594763.0.20260527_234148
    $CKPT_PREFIX/antmaze-medium-navigate/sd000_s_34594838.0.20260527_234324
    $CKPT_PREFIX/antsoccer-arena-navigate/sd000_s_34594769.0.20260527_234149
    $CKPT_PREFIX/antsoccer-arena-navigate/sd000_s_34739255.0.20260531_064819
)
ALPHAS=(1 3 10)

# -----------------------------
# Dispatch: at most one job per GPU at a time (GPU-token semaphore).
# -----------------------------
NUM_GPUS=${#GPUS[@]}
JOB=0

# FIFO used as a semaphore: it holds one token (a GPU id) per free GPU. A job
# blocks on `read -u 9` until a token is available, then releases it on exit.
FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 9<>"$FIFO"
rm -f "$FIFO"
for GPU in "${GPUS[@]}"; do echo "$GPU"; done >&9

for CFG in "${!ENVS[@]}"; do
    ENV=${ENVS[$CFG]}
    SKILL_CKPT=${SKILL_CKPTS[$CFG]}
    for ALPHA in "${ALPHAS[@]}"; do
        read -u 9 GPU      # block until a GPU is free
        TAG="${ENV}_$(basename "$SKILL_CKPT")_a${ALPHA}"
        LOG="${LOG_DIR}/${TAG}.log"

        echo "========================================"
        echo "JOB=${JOB}  GPU=${GPU}"
        echo "ENV=${ENV}"
        echo "CKPT=${SKILL_CKPT}"
        echo "ALPHA=${ALPHA}"
        echo "LOG=${LOG}"
        echo "========================================"

        (
            # Release the GPU back to the pool on exit (success or failure).
            trap 'echo "${GPU}" >&9' EXIT
            CUDA_VISIBLE_DEVICES=${GPU} python main.py \
                --env_name=$ENV \
                --save_dir=$SAVE_DIR \
                --agent=agents/skill_match.py \
                --agent.skill_checkpoint_path=$SKILL_CKPT \
                --agent.alpha=$ALPHA \
                --eval_episodes=50 \
                --video_episodes=0 \
                --train_steps=1000000 \
                > "$LOG" 2>&1
        ) &

        JOB=$((JOB + 1))
        # Small stagger so near-simultaneous mujoco-EGL inits don't race.
        sleep 2
    done
done

echo "Launched ${JOB} jobs across ${NUM_GPUS} GPUs (<=1 per GPU at a time). Waiting..."
wait
exec 9>&-
echo "All ${JOB} jobs finished."
