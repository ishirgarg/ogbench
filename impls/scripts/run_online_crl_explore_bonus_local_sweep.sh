#!/usr/bin/env bash
# Local (non-Slurm) exploration-bonus sweep for flat online CRL + RLPD, packed onto GPUs
# 0/1/2 of THIS machine (rnn.ist.berkeley.edu -- a shared lab box; `nvidia-smi` may already
# show other users' jobs on these GPUs, and packing more work on top will slow everything
# down further, this sweep included). See agents/online_crl.py ("Exploration reward bonus")
# and scripts/slurm/submit_flat_crl_explore_bonus_sweep.sh for the earlier (Slurm) version.
#
# 3 cells x 3 starting bonus_scale x 2 explore_reward x 2 add_explore x 4 seeds = 144 runs,
# JOBS_PER_GPU (default 8) concurrent per GPU -> 24 in flight at a time, drained from a
# per-GPU queue as slots free up (bash `wait -n`), not launched all at once.
#
# Differences from the earlier Slurm sweep:
#   * starting bonus_scale in {1, 0.3, 0.1} (that sweep used 0.001-0.3)
#   * every run here explicitly anneals the bonus to 0 by halfway through training
#     (agents/online_crl.py's `explore_reward_time_frac`, added 2026-09-16). The agent's OWN
#     default is "no annealing" (constant bonus_scale) -- this script pins BONUS_TIME_FRAC=0.5
#     itself so this specific sweep stays annealed even as that global default changes.
#     Pass BONUS_TIME_FRAC=0 to disable annealing and match the earlier sweep's constant scale.
#   * emp_entropy_target=False in every run, same as the earlier sweep (isolates the bonus).
#
# exp_name collision note: get_exp_name() (utils/log_utils.py) is `sd<seed>_<timestamp to the
# second>` outside Slurm, so two same-seed jobs launched in the same wall-clock second would
# collide on one save_dir. This script sets SLURM_JOB_ID=local-<pid> per job (log_utils.py's
# only other use of that var) purely to make every exp_name unique -- it does not talk to Slurm.
#
# Usage (from this NAS checkout):
#   bash scripts/run_online_crl_explore_bonus_local_sweep.sh
# Overrides (env vars): DRY_RUN=1, SEEDS, SCALES, MODES, REWARDS, GROUP_KEYS, JOBS_PER_GPU,
#   TOTAL_ENV_STEPS (default 1000000 -- packing 8/GPU will make this much slower than a solo
#   Slurm run; consider lowering for a first look), BONUS_TIME_FRAC, EMP_NUM_SPLUS_SAMPLES.
set -uo pipefail   # no -e: one run's failure must not kill the other 143
cd "$(dirname "$0")/.."   # -> impls/

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}   # EGL is broken headless on this box (see repo memory)
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # mandatory for packing multiple jobs per GPU
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
# The local root disk is nearly full (checked 2026-09-16: 581M free on a 1.8T volume) --
# keep every scratch write (wandb spooling, temp files) off it and on the NAS instead.
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR/wandb_explore_bonus_local"
mkdir -p "$WANDB_DIR"
export JAX_COMPILATION_CACHE_DIR=/nas/ucb/ishirgarg/ogbench/impls/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
# Cap per-process CPU threading: 128 cores / 24 concurrent jobs ~= 5 each; left unbounded,
# every process's BLAS/OMP pool would try to claim all 128 cores and thrash.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}

DRY_RUN=${DRY_RUN:-0}
GPUS=(0 1 2)
JOBS_PER_GPU=${JOBS_PER_GPU:-8}
SEEDS=${SEEDS:-"0 1 2 3"}
SCALES=${SCALES:-"1 0.3 0.1"}
MODES=${MODES:-"reward-to-rlpd reward"}
REWARDS=${REWARDS:-"empowerment max_episodic_empowerment"}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
BONUS_TIME_FRAC=${BONUS_TIME_FRAC:-0.5}   # this sweep is annealed by design, not just by the agent's default
EMP_NUM_SPLUS_SAMPLES=${EMP_NUM_SPLUS_SAMPLES:-64}

EMP_ROOT=ckpts/final/empowerment_final
ALL_CELL_KEYS=(asoc_ctr pmt_nav cube_sgl)
ALL_CELL_ENVS=(
    antsoccer-arena-center-online-v0
    pointmaze-teleport-center-online-v0
    cube-single-center-online-v0
)
ALL_CELL_OFFLINE=(
    antsoccer-arena-navigate-v0
    pointmaze-teleport-navigate-v0
    cube-single-play-v0
)
ALL_CELL_EMP_CKPT=(
    "$EMP_ROOT/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836"
    "$EMP_ROOT/pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836"
    "$EMP_ROOT/cube-single-play/sd000_s_38624008.0.20260908_013305"
)
ALL_CELL_EPISODE_LENGTH=(500 "" "")

read -r -a WANTED <<< "${GROUP_KEYS:-${ALL_CELL_KEYS[*]}}"
CELL_KEYS=(); CELL_ENVS=(); CELL_OFFLINE=(); CELL_EMP_CKPT=(); CELL_EPISODE_LENGTH=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_CELL_KEYS[@]}"; do
        if [[ "${ALL_CELL_KEYS[$i]}" == "$w" ]]; then
            CELL_KEYS+=("${ALL_CELL_KEYS[$i]}"); CELL_ENVS+=("${ALL_CELL_ENVS[$i]}")
            CELL_OFFLINE+=("${ALL_CELL_OFFLINE[$i]}"); CELL_EMP_CKPT+=("${ALL_CELL_EMP_CKPT[$i]}")
            CELL_EPISODE_LENGTH+=("${ALL_CELL_EPISODE_LENGTH[$i]}")
            found=1; break
        fi
    done
    (( found )) || { echo "ERROR: unknown group key '$w' (known: ${ALL_CELL_KEYS[*]})" >&2; exit 1; }
done
for ck in "${CELL_EMP_CKPT[@]}"; do
    if [[ ! -f "$ck/flags.json" ]]; then echo "ERROR: estimator checkpoint $ck has no flags.json." >&2; exit 1; fi
done
for od in "${CELL_OFFLINE[@]}"; do
    if [[ ! -f "$OGBENCH_DATASET_DIR/$od.npz" ]]; then echo "ERROR: $OGBENCH_DATASET_DIR/$od.npz is missing." >&2; exit 1; fi
done

LOG_DIR=logs/local_explore_bonus_sweep
mkdir -p "$LOG_DIR"

echo "[sweep] $(hostname) GPU state before launch:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# Every job goes on one of 3 per-GPU queues (round robin), so same-seed jobs land on different
# GPUs and each queue is drained independently at up to JOBS_PER_GPU concurrent children.
declare -a QUEUE_0=() QUEUE_1=() QUEUE_2=()
QUEUE_NAMES=(QUEUE_0 QUEUE_1 QUEUE_2)
n=0
for c in "${!CELL_KEYS[@]}"; do
    for SCALE in $SCALES; do
        for REWARD in $REWARDS; do
            for MODE in $MODES; do
                for SEED in $SEEDS; do
                    job="${CELL_KEYS[$c]}|${CELL_ENVS[$c]}|${CELL_OFFLINE[$c]}|${CELL_EMP_CKPT[$c]}|${CELL_EPISODE_LENGTH[$c]}|$SCALE|$REWARD|$MODE|$SEED"
                    qname=${QUEUE_NAMES[$((n % 3))]}
                    declare -n qref="$qname"
                    qref+=("$job")
                    unset -n qref
                    n=$((n + 1))
                done
            done
        done
    done
done
echo "[sweep] built $n jobs across 3 GPU queues (${#QUEUE_0[@]}/${#QUEUE_1[@]}/${#QUEUE_2[@]})"
[[ "$DRY_RUN" == "1" ]] && echo "[sweep] DRY_RUN=1: printing commands only, not launching"

run_one() {  # GPU JOB_SPEC
    local GPU=$1 SPEC=$2
    IFS='|' read -r KEY ENV_NAME OFFLINE EMP_CKPT EP_LEN SCALE REWARD MODE SEED <<< "$SPEC"
    local EP_FLAG=(); [[ -n "$EP_LEN" ]] && EP_FLAG=(--episode_length="$EP_LEN")
    local ANN_FLAG=(); [[ -n "$BONUS_TIME_FRAC" ]] && ANN_FLAG=(--agent.explore_reward_time_frac="$BONUS_TIME_FRAC")
    local rtag=emp; [[ "$REWARD" == "max_episodic_empowerment" ]] && rtag=max
    local mtag=rlpd; [[ "$MODE" == "reward" ]] && mtag=onl
    local NAME="${KEY}_${mtag}_${rtag}_a${SCALE}_s${SEED}"
    local LOG="$LOG_DIR/${NAME}.log"
    local cmd=(
        "$PYTHON" -u main_online.py
        --env_name="$ENV_NAME" --seed="$SEED" --agent=agents/online_crl.py
        --total_env_steps="$TOTAL_ENV_STEPS"
        "${EP_FLAG[@]}"
        --offline_dataset="$OFFLINE"
        --agent.emp_checkpoint_path="$EMP_CKPT"
        --agent.emp_entropy_target=False
        --agent.emp_num_splus_samples="$EMP_NUM_SPLUS_SAMPLES"
        --agent.add_explore="$MODE" --agent.explore_reward="$REWARD" --agent.bonus_scale="$SCALE"
        "${ANN_FLAG[@]}"
        --log_interval=5000 --eval_interval=20000 --save_interval=1000000
        --eval_episodes=20 --video_episodes=0
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "gpu=$GPU $NAME: ${cmd[*]}"
        return 0
    fi
    echo "[sweep] gpu=$GPU launching $NAME -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU SLURM_JOB_ID="local-$$" \
    XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${NAME}_$$" \
        "${cmd[@]}" > "$LOG" 2>&1
    echo "[sweep] gpu=$GPU finished $NAME (exit $?)"
}

gpu_worker() {
    local GPU=$1; shift
    local queue=("$@")
    for spec in "${queue[@]}"; do
        while (( $(jobs -rp | wc -l) >= JOBS_PER_GPU )); do
            wait -n
        done
        run_one "$GPU" "$spec" &
        sleep 1   # stagger subshell startup a little; SLURM_JOB_ID=local-$$ already guarantees
                  # exp_name uniqueness even for same-second launches, this is just politeness.
    done
    wait
}

gpu_worker "${GPUS[0]}" "${QUEUE_0[@]+"${QUEUE_0[@]}"}" &
gpu_worker "${GPUS[1]}" "${QUEUE_1[@]+"${QUEUE_1[@]}"}" &
gpu_worker "${GPUS[2]}" "${QUEUE_2[@]+"${QUEUE_2[@]}"}" &
wait
echo "[sweep] all $n jobs finished"
