#!/usr/bin/env bash
# Distilled empowerment bonus sweep (agents/online_crl.py, add_explore=distill |
# distill-to-rlpd, distill_target=episode_max) on THREE HARDER multi-goal / corner
# online envs, packed onto GPUs 0/1/2 of this machine. Same per-env sweep shape as
# scripts/run_distill_bonus_local_sweep.sh (2 modes x [4 constant alphas {3,10,30,100} +
# 3 annealed alphas {10,30,100}, ANNEAL_FRAC=0.5] x 5 seeds = 70, + 5-seed no-bonus
# baseline = 75 configs per env), but run ENV-BY-ENV, SEQUENTIALLY: env N's queues are
# fully drained (every config finished or failed) before env N+1's jobs are ever
# launched -- explicit user request, the opposite of the cell-interleaving the sibling
# script uses to avoid queue starvation. Order (user, 2026-09-20: maze BEFORE soccer):
# cube_mg, amz_corner, asoc_cmg.
#
# Envs (all reuse an EXISTING estimator/RLPD-dataset pair -- no new estimator training
# needed; obs dims verified to match: cube 28, antsoccer 42, antmaze 29):
#   cube_mg     cube-single-multigoal-online-v0        RLPD cube-single-play-v0
#               (registered horizon 200, kept)
#   amz_corner  antmaze-medium-corner-sparse-online-v0 RLPD antmaze-medium-navigate-v0
#               (registered horizon 1000, kept; k50 estimator, matching every other cell)
#   asoc_cmg    antsoccer-arena-corner-multigoal-online-v0 RLPD antsoccer-arena-navigate-v0
#               NEW env (2026-09-20): the corner soccer start (ant (2,2), ball (5,5)) with the
#               48-goal multigoal task set built exactly like antsoccer-arena-multigoal-v0 is
#               from the center env (every integer offset in [-3,3]^2 from the ball except
#               (0,0)); it does NOT contain the single corner goal (5,10). Registered in this
#               worktree's ogbench/locomaze/__init__.py -- see PYTHONPATH below.
#               (registered horizon 1000; episode_length=500 override, same convention
#               every other antsoccer online run in this repo uses)
#
# REALISTIC EXPECTATIONS: the sibling 3-cell sweep (same JOBS_PER_GPU=12x3) reached
# ~700k of 1,000,000 steps in ~10h for its first wave of 36 jobs. Each env here has 75
# configs, i.e. 3 waves of up to 36 concurrent -- roughly 2-3x that per env before the
# NEXT env's jobs get a slot (sequential by design), so full completion of all three is
# likely 1-3 days, not hours.
#
# Usage (from this NAS checkout, impls/):
#   bash scripts/run_distill_bonus_local_sweep_newenvs.sh
# Overrides (env vars): DRY_RUN=1, JOBS_PER_GPU, SEEDS, ALPHAS, ANNEAL_ALPHAS,
#   ANNEAL_FRAC, ENV_KEYS (subset/reorder, e.g. "asoc_cmg cube_mg"), MODES,
#   INCLUDE_BASELINE (0/1), TOTAL_ENV_STEPS, EVAL_EPISODES, EVAL_INTERVAL, DIAG_SEEDS.
# (ENV_KEYS picks/reorders from: cube_mg amz_corner asoc_cmg.)
set -uo pipefail   # no -e: one run's failure must not kill the others
cd "$(dirname "$0")/.."   # -> impls/

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
# ogbench is an editable install pointing at the MAIN checkout, which this branch must not edit;
# the corner-multigoal env (and the multigoal envs that so far only exist as uncommitted edits in
# main) are registered in THIS worktree's copy, so make it shadow the editable install.
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}   # EGL is broken headless on this box (repo memory)
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # mandatory for packing multiple jobs per GPU
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp   # local root disk has been near-full before
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR/wandb_distill_bonus_local_newenvs"
mkdir -p "$WANDB_DIR"
export JAX_COMPILATION_CACHE_DIR=/nas/ucb/ishirgarg/ogbench/impls/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-3}

grep -q "distill_target" agents/online_crl.py 2>/dev/null || {
    echo "FATAL: agents/online_crl.py has no distill_target -- wrong checkout/branch." >&2; exit 1; }
"$PYTHON" - <<'PY' || { echo "FATAL: worktree ogbench does not shadow the editable install / lacks the new envs." >&2; exit 1; }
import os, ogbench
from ogbench.locomaze import __file__ as f
assert os.path.realpath(ogbench.__file__).startswith(os.path.realpath(os.path.join(os.getcwd(), '..'))), ogbench.__file__
import gymnasium as gym
for i in ('antsoccer-arena-corner-multigoal-v0', 'cube-single-multigoal-v0', 'antmaze-medium-corner-sparse-v0'):
    assert i in gym.registry, i
print('[sweep] ogbench from', ogbench.__file__)
PY

DRY_RUN=${DRY_RUN:-0}
GPUS=(0 1 2)
JOBS_PER_GPU=${JOBS_PER_GPU:-12}
SEEDS=${SEEDS:-"0 1 2 3 4"}
MODES=${MODES:-"distill distill-to-rlpd"}
ALPHAS=${ALPHAS:-"3 10 30 100"}
ANNEAL_ALPHAS=${ANNEAL_ALPHAS:-"10 30 100"}
ANNEAL_FRAC=${ANNEAL_FRAC:-0.5}
DISTILL_TARGET=${DISTILL_TARGET:-episode_max}
DIAG_SEEDS=${DIAG_SEEDS:-"0"}
INCLUDE_BASELINE=${INCLUDE_BASELINE:-1}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-25000}
EMP_NUM_SPLUS_SAMPLES=${EMP_NUM_SPLUS_SAMPLES:-64}

MAIN=${MAIN:-/nas/ucb/ishirgarg/ogbench/impls}
EMP_ROOT=$MAIN/ckpts/final/empowerment_final

# Order matters: this is the ENV SEQUENCE (cube_mg -> amz_corner -> asoc_cmg), not a set.
ALL_ENV_KEYS=(cube_mg amz_corner asoc_cmg)
ALL_ENV_NAMES=(cube-single-multigoal antmaze-medium-corner-sparse antsoccer-arena-corner-multigoal)
ALL_ENV_ONLINE=(
    cube-single-multigoal-online-v0
    antmaze-medium-corner-sparse-online-v0
    antsoccer-arena-corner-multigoal-online-v0
)
ALL_ENV_OFFLINE=(
    cube-single-play-v0
    antmaze-medium-navigate-v0
    antsoccer-arena-navigate-v0
)
ALL_ENV_EMP_CKPT=(
    "$EMP_ROOT/cube-single-play/sd000_s_38624008.0.20260908_013305"
    "$EMP_ROOT/antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001"
    "$EMP_ROOT/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836"
)
ALL_ENV_EPISODE_LENGTH=("" "" 500)   # antsoccer: project-wide override (registered 1000 -> 500)

read -r -a WANTED <<< "${ENV_KEYS:-${ALL_ENV_KEYS[*]}}"
KEYS=(); NAMES=(); ONLINE=(); OFFLINE=(); EMP_CKPT=(); EP_LEN=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_ENV_KEYS[@]}"; do
        if [[ "${ALL_ENV_KEYS[$i]}" == "$w" ]]; then
            KEYS+=("${ALL_ENV_KEYS[$i]}"); NAMES+=("${ALL_ENV_NAMES[$i]}"); ONLINE+=("${ALL_ENV_ONLINE[$i]}")
            OFFLINE+=("${ALL_ENV_OFFLINE[$i]}"); EMP_CKPT+=("${ALL_ENV_EMP_CKPT[$i]}")
            EP_LEN+=("${ALL_ENV_EPISODE_LENGTH[$i]}")
            found=1; break
        fi
    done
    (( found )) || { echo "FATAL: unknown env key '$w' (known: ${ALL_ENV_KEYS[*]})" >&2; exit 1; }
done
for ck in "${EMP_CKPT[@]}"; do
    [[ -f "$ck/flags.json" ]] || { echo "FATAL: estimator checkpoint $ck has no flags.json." >&2; exit 1; }
done
for od in "${OFFLINE[@]}"; do
    [[ -f "$OGBENCH_DATASET_DIR/$od.npz" ]] || { echo "FATAL: $OGBENCH_DATASET_DIR/$od.npz is missing." >&2; exit 1; }
done

SAVE_ROOT=${SAVE_ROOT:-$MAIN/exp/distill_bonus_local_newenvs}
LOG_DIR=${LOG_DIR:-$MAIN/logs/distill_bonus_local_newenvs_sweep}
mkdir -p "$LOG_DIR"

echo "[sweep] $(hostname) GPU state before launch:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
echo "[sweep] env order (sequential, each fully drains before the next starts): ${KEYS[*]}"

run_one() {  # GPU ENV_IDX MODE ALPHA ANNEAL SEED RUN_TAG
    local GPU=$1 ei=$2 MODE=$3 ALPHA=$4 ANNEAL=$5 SEED=$6 RUN_TAG=$7
    local ENV_NAME=${ONLINE[$ei]} OFF=${OFFLINE[$ei]} EMPCK=${EMP_CKPT[$ei]} EPL=${EP_LEN[$ei]} NAME_KEY=${KEYS[$ei]}
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
    local SAVE_DIR="$SAVE_ROOT/${NAMES[$ei]}/$RUN_TAG"
    mkdir -p "$SAVE_DIR"
    local NAME="${NAME_KEY}_${RUN_TAG}_s${SEED}"
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
    CUDA_VISIBLE_DEVICES=$GPU SLURM_JOB_ID="local-$BASHPID" \
    XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${NAME}_$BASHPID" \
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
        IFS='|' read -r ei MODE ALPHA ANNEAL SEED RUN_TAG <<< "$spec"
        run_one "$GPU" "$ei" "$MODE" "$ALPHA" "$ANNEAL" "$SEED" "$RUN_TAG" &
        sleep 1
    done
    wait   # drain THIS gpu's queue fully before returning (needed for the outer per-env wait)
}

for ei in "${!KEYS[@]}"; do
    echo ""
    echo "[sweep] ==================== starting env ${KEYS[$ei]} (${ONLINE[$ei]}) ===================="

    # Build this env's 75 job specs, round-robined across 3 GPU queues.
    declare -a JOBS=()
    add_job() { JOBS+=("$ei|$1|$2|$3|$4|$5"); }   # MODE ALPHA ANNEAL SEED RUN_TAG
    if [[ "$INCLUDE_BASELINE" == "1" ]]; then
        for SEED in $SEEDS; do add_job none 0 "" "$SEED" "rlpd_baseline"; done
    fi
    for MODE in $MODES; do
        MTAG=ed; [[ "$MODE" == "distill-to-rlpd" ]] && MTAG=edrlpd
        for ALPHA in $ALPHAS; do
            for SEED in $SEEDS; do add_job "$MODE" "$ALPHA" "" "$SEED" "rlpd_noent_${MTAG}${ALPHA}"; done
        done
        for ALPHA in $ANNEAL_ALPHAS; do
            for SEED in $SEEDS; do add_job "$MODE" "$ALPHA" "$ANNEAL_FRAC" "$SEED" "rlpd_noent_ann${ANNEAL_FRAC}_${MTAG}${ALPHA}"; done
        done
    done
    n=${#JOBS[@]}
    echo "[sweep] ${KEYS[$ei]}: built $n jobs"

    declare -a QUEUE_0=() QUEUE_1=() QUEUE_2=()
    QUEUE_NAMES=(QUEUE_0 QUEUE_1 QUEUE_2)
    for i in "${!JOBS[@]}"; do
        qname=${QUEUE_NAMES[$((i % 3))]}
        declare -n qref="$qname"
        qref+=("${JOBS[$i]}")
        unset -n qref
    done
    echo "[sweep] ${KEYS[$ei]}: queues ${#QUEUE_0[@]}/${#QUEUE_1[@]}/${#QUEUE_2[@]} (JOBS_PER_GPU=$JOBS_PER_GPU)"
    [[ "$DRY_RUN" == "1" ]] && echo "[sweep] DRY_RUN=1: printing commands only, not launching"

    gpu_worker "${GPUS[0]}" "${QUEUE_0[@]+"${QUEUE_0[@]}"}" &
    gpu_worker "${GPUS[1]}" "${QUEUE_1[@]+"${QUEUE_1[@]}"}" &
    gpu_worker "${GPUS[2]}" "${QUEUE_2[@]+"${QUEUE_2[@]}"}" &
    wait   # <-- the sequential barrier: env ei+1 does not start until every job here has exited

    echo "[sweep] ==================== env ${KEYS[$ei]} fully drained (all $n jobs finished) ===================="
    unset JOBS QUEUE_0 QUEUE_1 QUEUE_2
done
echo "[sweep] all envs finished: ${KEYS[*]}"
