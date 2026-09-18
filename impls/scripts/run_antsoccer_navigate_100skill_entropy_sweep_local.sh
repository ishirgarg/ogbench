#!/usr/bin/env bash
# Local (non-Slurm) online CRL high-level skill controller sweep over the K=100 antsoccer
# empowerment checkpoint:
#   ckpts/final/empowerment_final/antsoccer-arena-navigate/sd000_s_38624729.0.20260908_014657
# (env_name antsoccer-arena-navigate-v0, agent.num_skills=100 -- NOT the K=50 checkpoint used
# by submit_empowerment_online_controller_seeds.sh / submit_antsoccer_corner_online_controller_seeds.sh).
#
# Sweep: target_entropy_frac in {0.25, 0.5} plus TES-SAC entropy annealing (Xu et al. 2021,
# arXiv:2112.02852; agent.use_tes=True, starting frac 1.0 per the 2026-09-14 TES seeds
# convention), x 4 seeds (0-3), x 2 online envs (center, corner) = 3 * 4 * 2 = 24 runs.
#
# Envs (ogbench/locomaze/__init__.py; deterministic, no init/goal noise):
#   antsoccer-arena-center-online-v0  -> saved under <SKILL_CKPT>/online_controller/rlpd/
#   antsoccer-arena-corner-online-v0  -> saved under <SKILL_CKPT>/online_controller_corner/rlpd/
# Both use --episode_length=500 (registered horizon is 1000, but every antsoccer online run in
# this repo overrides to 500 -- see scripts/run_online_crl.sh's per-env table); 500 % K(10) == 0.
#
# RLPD is on for every run, offline_dataset = the checkpoint's own dataset
# (antsoccer-arena-navigate-v0), matching every other online_crl_skill_controller sweep.
#
# GPU placement: all 12 CENTER runs on GPU 6, all 12 CORNER runs on GPU 7, launched
# concurrently (JOBS_PER_GPU defaults to 12, i.e. no queueing within a GPU). This is a shared
# lab box -- check `nvidia-smi` before launching; GPUs 6/7 were already near 100% compute
# utilization (though with only ~3.7GB/49GB memory used) from other users' jobs as of
# 2026-09-16. XLA_PYTHON_CLIENT_PREALLOCATE=false is set so 12 JAX processes can share a card
# without pre-grabbing all its memory. Lower JOBS_PER_GPU if runs OOM or thrash too much.
#
# Usage (from this NAS checkout, detached so it survives an ssh disconnect):
#   cd impls && setsid nohup bash scripts/run_antsoccer_navigate_100skill_entropy_sweep_local.sh \
#       > logs/antsoccer100_entropy_sweep_local.out 2>&1 &
# Overrides (env vars): DRY_RUN=1, SEEDS="0 1 2 3", ENT_CONDS="0.25 0.5 tes",
#   ENVS="center corner", GPU_CENTER=6, GPU_CORNER=7, JOBS_PER_GPU=12, TOTAL_ENV_STEPS=1000000.
set -uo pipefail   # no -e: one run's failure must not kill the other 23
cd "$(dirname "$0")/.."   # -> impls/

PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}   # EGL is broken headless on this box (see repo memory); video_episodes=0 anyway
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # mandatory for packing multiple jobs per GPU
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
export TMPDIR=${TMPDIR:-/nas/ttl=60d/ishirgarg/tmp}
mkdir -p "$TMPDIR"
export WANDB_DIR=${WANDB_DIR:-$TMPDIR/wandb_antsoccer100_entropy_sweep}
mkdir -p "$WANDB_DIR"
export JAX_COMPILATION_CACHE_DIR=/nas/ucb/ishirgarg/ogbench/impls/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
# 128 cores split across up to 24 concurrent jobs ~= 5 each; left unbounded, every process's
# BLAS/OMP pool would try to claim all 128 cores and thrash.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}

DRY_RUN=${DRY_RUN:-0}
SKILL_CKPT=${SKILL_CKPT:-ckpts/final/empowerment_final/antsoccer-arena-navigate/sd000_s_38624729.0.20260908_014657}
SEEDS=${SEEDS:-"0 1 2 3"}
ENT_CONDS=${ENT_CONDS:-"0.25 0.5 tes"}   # "tes" -> use_tes=True, target_entropy_frac=1.0 (paper init)
ENVS=${ENVS:-"center corner"}
GPU_CENTER=${GPU_CENTER:-6}
GPU_CORNER=${GPU_CORNER:-7}
JOBS_PER_GPU=${JOBS_PER_GPU:-12}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
EPISODE_LENGTH=${EPISODE_LENGTH:-500}

[[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

NUM_SKILLS=$($PYTHON -c "import json; print(json.load(open('$SKILL_CKPT/flags.json'))['agent']['num_skills'])")
OFFLINE_DATASET=$($PYTHON -c "import json; print(json.load(open('$SKILL_CKPT/flags.json'))['env_name'])")
SKILL_EPOCH=$($PYTHON -c "
import glob, os, re
print(max(int(re.search(r'params_(\d+)\.pkl\$', os.path.basename(p)).group(1))
          for p in glob.glob('$SKILL_CKPT/params_*.pkl')))
")
if [[ "$NUM_SKILLS" != "100" ]]; then
    echo "WARNING: expected num_skills=100 for this checkpoint, got $NUM_SKILLS" >&2
fi
[[ -f "$OGBENCH_DATASET_DIR/$OFFLINE_DATASET.npz" ]] || { echo "ERROR: $OGBENCH_DATASET_DIR/$OFFLINE_DATASET.npz is missing." >&2; exit 1; }

declare -A ALL_ENV_NAME=([center]=antsoccer-arena-center-online-v0 [corner]=antsoccer-arena-corner-online-v0)
declare -A ALL_SAVE_SUBDIR=([center]=online_controller [corner]=online_controller_corner)
declare -A ALL_GPU=([center]=$GPU_CENTER [corner]=$GPU_CORNER)

LOG_DIR=logs/antsoccer100_entropy_sweep_local
mkdir -p "$LOG_DIR"

echo "[sweep] ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH num_skills=$NUM_SKILLS offline_dataset=$OFFLINE_DATASET"
echo "[sweep] $(hostname) GPU state before launch:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

run_one() {  # GPU JOB_SPEC
    local GPU=$1 SPEC=$2
    IFS='|' read -r ENV_KEY ENT_COND SEED <<< "$SPEC"
    local ENV_NAME=${ALL_ENV_NAME[$ENV_KEY]}
    local SAVE_DIR="$SKILL_CKPT/${ALL_SAVE_SUBDIR[$ENV_KEY]}/rlpd"
    mkdir -p "$SAVE_DIR"

    local ENT_FRAC ENT_TAG TES_FLAG=()
    if [[ "$ENT_COND" == "tes" ]]; then
        ENT_FRAC=1.0
        ENT_TAG="tes"
        TES_FLAG=(--agent.use_tes=True)
    else
        ENT_FRAC="$ENT_COND"
        ENT_TAG="$ENT_COND"
    fi

    local NAME="${ENV_KEY}_ent${ENT_TAG}_s${SEED}"
    local LOG="$LOG_DIR/${NAME}.log"
    local cmd=(
        "$PYTHON" -u main_online.py
        --env_name="$ENV_NAME" --seed="$SEED" --save_dir="$SAVE_DIR"
        --agent=agents/online_crl_skill_controller.py
        --agent.skill_checkpoint_path="$SKILL_CKPT"
        --agent.skill_restore_epoch="$SKILL_EPOCH"
        --agent.skill_commitment_k=10
        --agent.target_entropy_frac="$ENT_FRAC"
        "${TES_FLAG[@]}"
        --total_env_steps="$TOTAL_ENV_STEPS"
        --episode_length="$EPISODE_LENGTH"
        --offline_dataset="$OFFLINE_DATASET"
        --log_interval=5000 --eval_interval=15000 --save_interval=100000
        --eval_episodes=100 --video_episodes=0
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "gpu=$GPU $NAME: ${cmd[*]}"
        return 0
    fi
    echo "[sweep] gpu=$GPU launching $NAME -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU SLURM_JOB_ID="local-$$-$NAME" \
    XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${NAME}_$$" \
        "${cmd[@]}" > "$LOG" 2>&1
    echo "[sweep] gpu=$GPU finished $NAME (exit $?)"
}

declare -a QUEUE_CENTER=() QUEUE_CORNER=()
for ENV_KEY in $ENVS; do
    for ENT_COND in $ENT_CONDS; do
        for SEED in $SEEDS; do
            job="$ENV_KEY|$ENT_COND|$SEED"
            if [[ "$ENV_KEY" == "center" ]]; then QUEUE_CENTER+=("$job"); else QUEUE_CORNER+=("$job"); fi
        done
    done
done
n=$(( ${#QUEUE_CENTER[@]} + ${#QUEUE_CORNER[@]} ))
echo "[sweep] built $n jobs: ${#QUEUE_CENTER[@]} on GPU $GPU_CENTER (center), ${#QUEUE_CORNER[@]} on GPU $GPU_CORNER (corner)"
[[ "$DRY_RUN" == "1" ]] && echo "[sweep] DRY_RUN=1: printing commands only, not launching"

gpu_worker() {
    local GPU=$1; shift
    local queue=("$@")
    for spec in "${queue[@]}"; do
        while (( $(jobs -rp | wc -l) >= JOBS_PER_GPU )); do
            wait -n
        done
        run_one "$GPU" "$spec" &
        sleep 1   # stagger subshell startup; SLURM_JOB_ID=local-$$-<name> already guarantees
                  # exp_name uniqueness even for same-second launches, this is just politeness.
    done
    wait
}

gpu_worker "$GPU_CENTER" "${QUEUE_CENTER[@]+"${QUEUE_CENTER[@]}"}" &
gpu_worker "$GPU_CORNER" "${QUEUE_CORNER[@]+"${QUEUE_CORNER[@]}"}" &
wait
echo "[sweep] all $n jobs finished"
