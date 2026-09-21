#!/usr/bin/env bash
# LOCAL (non-Slurm) twin of scripts/rail_slurm_distill_bonus_sweep_high.sh: same 225-config
# sweep of the distilled empowerment bonus (agents/online_crl.py, add_explore=distill |
# distill-to-rlpd, distill_target=episode_max), packed onto GPUs 0/1/2 of THIS machine
# (rnn.ist.berkeley.edu -- a shared lab box; `nvidia-smi` may already show other users' jobs
# on these GPUs, and packing more work on top slows everything down further, this sweep
# included). Same RLPD-always-on, 100-eval-episodes/25k-steps, emp_entropy_target=False,
# episode_max, bonus_grad_diagnostics-on-seed-0 design as the rail script; see that file's
# header for the full rationale. This one only differs in HOW the runs launch (local process
# packing instead of Slurm array tasks).
#
# Sweep (matches rail exactly): 3 cells x 2 modes x [4 constant alphas {3,10,30,100} +
# 3 annealed alphas {10,30,100}, ANNEAL_FRAC=0.5] x 5 seeds = 210, + 3 cells x 5 seeds
# no-bonus baseline = 15. Total 225 runs.
#
# JOBS_PER_GPU (default 12) concurrent per GPU on GPUs 0/1/2 -> 36 in flight at a time,
# drained from a per-GPU queue as slots free (bash `wait -n`), not launched all at once.
#
# CELL ORDERING, fixed vs. the precedent this reuses (scripts/run_online_crl_explore_bonus_
# local_sweep.sh): that 144-job sweep generated jobs cell-by-cell (all antsoccer first), so
# with a slow per-run wall clock the queues never drained past cell 1 in 5.5h before being
# killed -- pointmaze and cube never got a single job started. Here CELL is the fastest-varying
# axis in the generation order (baseline runs too), so every 3 consecutive jobs cover all three
# cells; round-robin GPU assignment on top of that keeps each GPU's own queue mixed across
# cells from the start instead of draining one cell before touching the next.
#
# REALISTIC EXPECTATIONS: this is a bigger ask than that 144-job precedent (36 concurrent vs.
# 24, 100 eval episodes vs. 20, same 1M-step runs) and that one did not finish a single run in
# 5.5h at the smaller size. Expect this to be slow -- likely many hours to days for even a
# fraction of the sweep to reach 1M steps under 36-way packing. TOTAL_ENV_STEPS and
# EVAL_EPISODES below are overridable for a faster first look; the defaults match the rail
# sweep exactly so results are comparable to it.
#
# GPU / CPU packing: GPU memory is not the constraint (measured ~1.1 GiB/run on this box for
# the same agent; 12 per GPU is ~13 GiB of a 48 GB card). CPU is: 128 cores / 36 concurrent
# processes ~= 3.5 each (env stepping + CPU evals are CPU-bound) -- OMP/MKL/NUMEXPR threads are
# capped per process below so they don't each try to claim all 128 cores and thrash. Host RAM:
# each run holds a 1M-row offline dataset + a growing replay buffer (~3-5 GiB RSS); 36 concurrent
# is a few hundred GiB, fine on this box's 1 TB but worth checking `free -g` if other users are
# also active (this is a shared, `45 users` box as of the last check).
#
# Usage (from this NAS checkout, impls/):
#   bash scripts/run_distill_bonus_local_sweep.sh
# Overrides (env vars): DRY_RUN=1 (print commands, launch nothing), JOBS_PER_GPU, SEEDS,
#   ALPHAS, ANNEAL_ALPHAS, ANNEAL_FRAC, CELL_KEYS (subset, e.g. "asoc_ctr pmt_ctr"),
#   MODES, INCLUDE_BASELINE (0/1), TOTAL_ENV_STEPS, EVAL_EPISODES, EVAL_INTERVAL, DIAG_SEEDS.
set -uo pipefail   # no -e: one run's failure must not kill the other 224
cd "$(dirname "$0")/.."   # -> impls/

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}   # EGL is broken headless on this box (repo memory)
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # mandatory for packing multiple jobs per GPU
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}
# Local root disk has been near-full before -- keep every scratch write off it and on the NAS.
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR/wandb_distill_bonus_local"
mkdir -p "$WANDB_DIR"
export JAX_COMPILATION_CACHE_DIR=/nas/ucb/ishirgarg/ogbench/impls/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
# 128 cores / 36 concurrent jobs ~= 3.5 each; left unbounded, every process's BLAS/OMP pool
# would try to claim all 128 cores and thrash.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-3}

grep -q "distill_target" agents/online_crl.py 2>/dev/null || {
    echo "FATAL: agents/online_crl.py has no distill_target -- wrong checkout/branch." >&2; exit 1; }

DRY_RUN=${DRY_RUN:-0}
GPUS=(0 1 2)
JOBS_PER_GPU=${JOBS_PER_GPU:-12}
SEEDS=${SEEDS:-"0 1 2 3 4"}
MODES=${MODES:-"distill distill-to-rlpd"}
ALPHAS=${ALPHAS:-"3 10 30 100"}                 # constant-alpha arm
ANNEAL_ALPHAS=${ANNEAL_ALPHAS:-"10 30 100"}      # annealed arm (initial bonus_scale)
ANNEAL_FRAC=${ANNEAL_FRAC:-0.5}
DISTILL_TARGET=${DISTILL_TARGET:-episode_max}
DIAG_SEEDS=${DIAG_SEEDS:-"0"}                    # seeds that log bonus_grad/* (extra backward passes)
INCLUDE_BASELINE=${INCLUDE_BASELINE:-1}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-25000}
EMP_NUM_SPLUS_SAMPLES=${EMP_NUM_SPLUS_SAMPLES:-64}

# Checkpoints, RLPD dataset cache, and this sweep's own results all live in the MAIN checkout
# (this worktree's ckpts/ is empty -- untracked scratch data that only exists there).
MAIN=${MAIN:-/nas/ucb/ishirgarg/ogbench/impls}
EMP_ROOT=$MAIN/ckpts/final/empowerment_final
ALL_CELL_KEYS=(asoc_ctr pmt_ctr cube_sgl)
ALL_CELL_NAMES=(antsoccer-arena-center pointmaze-teleport-center cube-single-center)
ALL_CELL_ENVS=(
    antsoccer-arena-center-online-v0
    pointmaze-teleport-center-online-v0
    cube-single-center-online-v0
)
ALL_CELL_OFFLINE=(
    antsoccer-arena-navigate-v0
    pointmaze-teleport-stitch-v0
    cube-single-play-v0
)
ALL_CELL_EMP_CKPT=(
    "$EMP_ROOT/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836"
    "$EMP_ROOT/pointmaze-teleport-stitch/sd000_s_38390675.0.20260901_154836"
    "$EMP_ROOT/cube-single-play/sd000_s_38624008.0.20260908_013305"
)
ALL_CELL_EPISODE_LENGTH=(500 "" "")

read -r -a WANTED <<< "${CELL_KEYS:-${ALL_CELL_KEYS[*]}}"
KEYS=(); NAMES=(); ENVS=(); OFFLINE=(); EMP_CKPT=(); EP_LEN=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_CELL_KEYS[@]}"; do
        if [[ "${ALL_CELL_KEYS[$i]}" == "$w" ]]; then
            KEYS+=("${ALL_CELL_KEYS[$i]}"); NAMES+=("${ALL_CELL_NAMES[$i]}"); ENVS+=("${ALL_CELL_ENVS[$i]}")
            OFFLINE+=("${ALL_CELL_OFFLINE[$i]}"); EMP_CKPT+=("${ALL_CELL_EMP_CKPT[$i]}")
            EP_LEN+=("${ALL_CELL_EPISODE_LENGTH[$i]}")
            found=1; break
        fi
    done
    (( found )) || { echo "FATAL: unknown cell key '$w' (known: ${ALL_CELL_KEYS[*]})" >&2; exit 1; }
done
for ck in "${EMP_CKPT[@]}"; do
    [[ -f "$ck/flags.json" ]] || { echo "FATAL: estimator checkpoint $ck has no flags.json." >&2; exit 1; }
done
for od in "${OFFLINE[@]}"; do
    [[ -f "$OGBENCH_DATASET_DIR/$od.npz" ]] || { echo "FATAL: $OGBENCH_DATASET_DIR/$od.npz is missing." >&2; exit 1; }
done

# Results saved into the MAIN checkout too (this worktree could be cleaned up mid-sweep).
SAVE_ROOT=${SAVE_ROOT:-$MAIN/exp/distill_bonus_local}
LOG_DIR=${LOG_DIR:-$MAIN/logs/distill_bonus_local_sweep}
mkdir -p "$LOG_DIR"

echo "[sweep] $(hostname) GPU state before launch:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# ---------------------------------------------------------------------------
# Build the job list with CELL as the fastest-varying axis (see header): every 3 consecutive
# entries cover all wanted cells, so round-robin GPU assignment keeps each GPU's queue mixed
# across cells instead of draining one cell before the others get a single slot.
# ---------------------------------------------------------------------------
declare -a JOBS=()
add_job() {  # KEY_IDX MODE ALPHA ANNEAL SEED RUN_TAG
    local ci=$1 MODE=$2 ALPHA=$3 ANNEAL=$4 SEED=$5 RUN_TAG=$6
    JOBS+=("${KEYS[$ci]}|${NAMES[$ci]}|${ENVS[$ci]}|${OFFLINE[$ci]}|${EMP_CKPT[$ci]}|${EP_LEN[$ci]}|$MODE|$ALPHA|$ANNEAL|$SEED|$RUN_TAG")
}
MODE_TAGS_ed=ed
MODE_TAGS_edrlpd_LOOKUP() { [[ "$1" == distill ]] && echo ed || echo edrlpd; }

if [[ "$INCLUDE_BASELINE" == "1" ]]; then
    for SEED in $SEEDS; do
        for ci in "${!KEYS[@]}"; do
            add_job "$ci" none 0 "" "$SEED" "rlpd_baseline"
        done
    done
fi
for MODE in $MODES; do
    MTAG=$(MODE_TAGS_edrlpd_LOOKUP "$MODE")
    for ALPHA in $ALPHAS; do
        for SEED in $SEEDS; do
            for ci in "${!KEYS[@]}"; do
                add_job "$ci" "$MODE" "$ALPHA" "" "$SEED" "rlpd_noent_${MTAG}${ALPHA}"
            done
        done
    done
    for ALPHA in $ANNEAL_ALPHAS; do
        for SEED in $SEEDS; do
            for ci in "${!KEYS[@]}"; do
                add_job "$ci" "$MODE" "$ALPHA" "$ANNEAL_FRAC" "$SEED" "rlpd_noent_ann${ANNEAL_FRAC}_${MTAG}${ALPHA}"
            done
        done
    done
done
n=${#JOBS[@]}
echo "[sweep] built $n jobs across ${#KEYS[@]} cell(s)"

# Round-robin the (already cell-interleaved) job list across 3 per-GPU queues.
declare -a QUEUE_0=() QUEUE_1=() QUEUE_2=()
QUEUE_NAMES=(QUEUE_0 QUEUE_1 QUEUE_2)
for i in "${!JOBS[@]}"; do
    qname=${QUEUE_NAMES[$((i % 3))]}
    declare -n qref="$qname"
    qref+=("${JOBS[$i]}")
    unset -n qref
done
echo "[sweep] queues: ${#QUEUE_0[@]}/${#QUEUE_1[@]}/${#QUEUE_2[@]} (JOBS_PER_GPU=$JOBS_PER_GPU -> $((JOBS_PER_GPU * 3)) concurrent)"
[[ "$DRY_RUN" == "1" ]] && echo "[sweep] DRY_RUN=1: printing commands only, not launching"

run_one() {  # GPU JOB_SPEC
    local GPU=$1 SPEC=$2
    IFS='|' read -r KEY CELL_NAME ENV_NAME OFF EMPCK EPL MODE ALPHA ANNEAL SEED RUN_TAG <<< "$SPEC"
    local EPF=(); [[ -n "$EPL" ]] && EPF=(--episode_length="$EPL")
    local BONUS=()
    if [[ "$MODE" != "none" ]]; then
        local DIAG=False
        for s in $DIAG_SEEDS; do [[ "$s" == "$SEED" ]] && DIAG=True; done
        BONUS=(
            --agent.emp_checkpoint_path="$EMPCK"
            --agent.emp_num_splus_samples="$EMP_NUM_SPLUS_SAMPLES"
            --agent.emp_entropy_target=False
            --agent.add_explore="$MODE"
            --agent.distill_target="$DISTILL_TARGET"
            --agent.bonus_scale="$ALPHA"
            --agent.bonus_grad_diagnostics="$DIAG"
        )
        [[ -n "$ANNEAL" ]] && BONUS+=(--agent.explore_reward_time_frac="$ANNEAL")
    fi
    local SAVE_DIR="$SAVE_ROOT/$CELL_NAME/$RUN_TAG"
    mkdir -p "$SAVE_DIR"
    local NAME="${KEY}_${RUN_TAG}_s${SEED}"
    local LOG="$LOG_DIR/${NAME}.log"
    local cmd=(
        "$PYTHON" -u main_online.py
        --env_name="$ENV_NAME" --seed="$SEED" --save_dir="$SAVE_DIR" --agent=agents/online_crl.py
        "${BONUS[@]}"
        --offline_dataset="$OFF"
        --total_env_steps="$TOTAL_ENV_STEPS"
        "${EPF[@]}"
        --log_interval=5000 --eval_interval="$EVAL_INTERVAL" --eval_episodes="$EVAL_EPISODES"
        --video_episodes=0 --save_interval="$TOTAL_ENV_STEPS"
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
        sleep 1   # stagger subshell startup; SLURM_JOB_ID=local-$$ already guarantees exp_name
                  # uniqueness even for same-second launches, this is just politeness.
    done
    wait
}

gpu_worker "${GPUS[0]}" "${QUEUE_0[@]+"${QUEUE_0[@]}"}" &
gpu_worker "${GPUS[1]}" "${QUEUE_1[@]+"${QUEUE_1[@]}"}" &
gpu_worker "${GPUS[2]}" "${QUEUE_2[@]+"${QUEUE_2[@]}"}" &
wait
echo "[sweep] all $n jobs finished"
