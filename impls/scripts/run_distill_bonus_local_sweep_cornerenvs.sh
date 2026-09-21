#!/usr/bin/env bash
# Distilled empowerment bonus sweep (agents/online_crl.py, add_explore=distill |
# distill-to-rlpd, distill_target=episode_max) on the THREE corner-start maze envs,
# packed onto GPUs 0/1/3/4/5 of this machine. The two remaining cells of the same
# 5-env request (antsoccer-arena-far-multigoal, antmaze-medium-corner-sparse) go to the
# rnn Slurm cluster instead -- see scripts/slurm/run_distill_bonus_corner_sweep.sbatch.
#
# ALPHAS ARE {10, 30} ONLY, narrowed from the earlier {3,10,30,100} sweep: on
# cube-single-multigoal (75-config sweep, finished 2026-09-21) alpha=100 was actively
# destructive in all four cells (0.004-0.025 vs a 0.035 baseline) and alpha=3 barely
# cleared baseline (0.059 / 0.090), while 10 and 30 bracketed the optimum (best cell
# distill-to-rlpd ann alpha=30 at 0.493).
#
# Per-env sweep: 2 modes x [2 constant alphas {10,30} + 2 annealed alphas {10,30},
# ANNEAL_FRAC=0.5] x 5 seeds = 40, + 5-seed no-bonus baseline = 45 configs per env.
# 5 GPUs x JOBS_PER_GPU=9 = 45 slots, so ONE env's 45 runs occupy every slot at once
# and finish together (explicit user choice, 2026-09-21). Envs run SEQUENTIALLY: env N
# is fully drained before env N+1 launches.
#
# Envs (ALL on the -navigate- datasets and the matching final/empowerment_final
# navigate estimators, per user request; obs dims verified: antmaze 29, pointmaze 2):
#   amz_all     antmaze-medium-corner-all-squares-online-v0   RLPD antmaze-medium-navigate-v0
#               NEW env (2026-09-21): ant at the medium maze's bottom-left corner
#               (1,1) = xy (0,0), goal = the center of every other free cell (25 tasks).
#               (registered horizon 1000, kept)
#   pmt_corner  pointmaze-teleport-corner-sparse-online-v0    RLPD pointmaze-teleport-navigate-v0
#               NEW env (2026-09-21): point at the teleport maze's bottom-left corner
#               (1,1) = xy (0,0); 3 far-corner goals (36,0), (0,24), (20,24) -- i.e.
#               pointmaze-teleport-sparse-v0's goal set minus the corner that is this
#               env's start. (registered horizon 1000, kept: point moves <= 0.2/step)
#   pmt_all     pointmaze-teleport-corner-all-squares-online-v0 RLPD pointmaze-teleport-navigate-v0
#               NEW env (2026-09-21): same corner start, goal = the center of every
#               other free cell minus the two teleport-IN pads (4,6)/(5,1) and the
#               walled-off (1,7) trap pocket (41 tasks). (registered horizon 1000)
# All three are registered ONLY in this worktree's ogbench/locomaze/__init__.py -- see
# PYTHONPATH below; the editable install points at the MAIN checkout, which lacks them.
#
# TIMING: the previous 12-jobs-per-GPU sweep reached ~700k of 1,000,000 steps in ~10h.
# At 9 per GPU each wave should be somewhat faster, but still budget ~12-16h per env
# and ~2 days for all three.
#
# Usage (from this worktree, impls/):
#   bash scripts/run_distill_bonus_local_sweep_cornerenvs.sh
#   (detach: setsid nohup bash scripts/... > logs/.../_master.log 2>&1 & )
# Overrides (env vars): DRY_RUN=1, GPUS, JOBS_PER_GPU, SEEDS, ALPHAS, ANNEAL_ALPHAS,
#   ANNEAL_FRAC, ENV_KEYS (subset/reorder), MODES, INCLUDE_BASELINE (0/1),
#   TOTAL_ENV_STEPS, EVAL_EPISODES, EVAL_INTERVAL, DIAG_SEEDS, SAVE_ROOT, LOG_DIR.
# (ENV_KEYS picks/reorders from: amz_all pmt_corner pmt_all.)
set -uo pipefail   # no -e: one run's failure must not kill the others
cd "$(dirname "$0")/.."   # -> impls/

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
# ogbench is an editable install pointing at the MAIN checkout, which this branch must not
# edit; the three corner envs are registered in THIS worktree's copy, so shadow the install.
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}   # EGL is broken headless on this box (repo memory)
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # mandatory for packing multiple jobs per GPU
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp   # local root disk has been near-full before
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR/wandb_distill_bonus_cornerenvs"
mkdir -p "$WANDB_DIR"
export JAX_COMPILATION_CACHE_DIR=/nas/ucb/ishirgarg/ogbench/impls/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-3}

grep -q "distill_target" agents/online_crl.py 2>/dev/null || {
    echo "FATAL: agents/online_crl.py has no distill_target -- wrong checkout/branch." >&2; exit 1; }
"$PYTHON" - <<'PY' || { echo "FATAL: worktree ogbench does not shadow the editable install / lacks the corner envs." >&2; exit 1; }
import os, ogbench
assert os.path.realpath(ogbench.__file__).startswith(os.path.realpath(os.path.join(os.getcwd(), '..'))), ogbench.__file__
import gymnasium as gym
for i in ('antmaze-medium-corner-all-squares-v0', 'pointmaze-teleport-corner-sparse-v0',
          'pointmaze-teleport-corner-all-squares-v0'):
    assert i in gym.registry, i
print('[sweep] ogbench from', ogbench.__file__)
PY

DRY_RUN=${DRY_RUN:-0}
read -r -a GPUS <<< "${GPUS:-0 1 3 4 5}"
JOBS_PER_GPU=${JOBS_PER_GPU:-9}
SEEDS=${SEEDS:-"0 1 2 3 4"}
MODES=${MODES:-"distill distill-to-rlpd"}
ALPHAS=${ALPHAS:-"10 30"}
ANNEAL_ALPHAS=${ANNEAL_ALPHAS:-"10 30"}
ANNEAL_FRAC=${ANNEAL_FRAC:-0.5}
DISTILL_TARGET=${DISTILL_TARGET:-episode_max}
DIAG_SEEDS=${DIAG_SEEDS:-"0"}
INCLUDE_BASELINE=${INCLUDE_BASELINE:-1}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-25000}
EMP_NUM_SPLUS_SAMPLES=${EMP_NUM_SPLUS_SAMPLES:-64}

MAIN=${MAIN:-/nas/ucb/ishirgarg/ogbench/impls}   # checkpoints/datasets/results live in the main tree
EMP_ROOT=$MAIN/ckpts/final/empowerment_final

# Order matters: this is the ENV SEQUENCE, not a set.
ALL_ENV_KEYS=(amz_all pmt_corner pmt_all)
ALL_ENV_NAMES=(
    antmaze-medium-corner-all-squares
    pointmaze-teleport-corner-sparse
    pointmaze-teleport-corner-all-squares
)
ALL_ENV_ONLINE=(
    antmaze-medium-corner-all-squares-online-v0
    pointmaze-teleport-corner-sparse-online-v0
    pointmaze-teleport-corner-all-squares-online-v0
)
ALL_ENV_OFFLINE=(
    antmaze-medium-navigate-v0
    pointmaze-teleport-navigate-v0
    pointmaze-teleport-navigate-v0
)
ALL_ENV_EMP_CKPT=(
    "$EMP_ROOT/antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001"
    "$EMP_ROOT/pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836"
    "$EMP_ROOT/pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836"
)
ALL_ENV_EPISODE_LENGTH=("" "" "")   # all three keep their registered horizons

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
    compgen -G "$ck/params_*.pkl" > /dev/null || { echo "FATAL: no params_*.pkl in $ck" >&2; exit 1; }
done
for od in "${OFFLINE[@]}"; do
    [[ -f "$OGBENCH_DATASET_DIR/$od.npz" ]] || { echo "FATAL: $OGBENCH_DATASET_DIR/$od.npz is missing." >&2; exit 1; }
done
# Each estimator must actually be an empowerment_skill run on this cell's RLPD dataset.
for i in "${!KEYS[@]}"; do
    "$PYTHON" - "${EMP_CKPT[$i]}" "${OFFLINE[$i]}" <<'PY' || exit 1
import json, sys
f = json.load(open(sys.argv[1] + '/flags.json'))
name, env = f['agent']['agent_name'], f['env_name']
assert name == 'empowerment_skill', f'{sys.argv[1]} is a {name} run, expected empowerment_skill'
assert env == sys.argv[2], f'{sys.argv[1]} was trained on {env}, but this cell uses {sys.argv[2]}'
PY
done

SAVE_ROOT=${SAVE_ROOT:-$MAIN/exp/distill_bonus_cornerenvs}
LOG_DIR=${LOG_DIR:-$MAIN/logs/distill_bonus_cornerenvs_sweep}
mkdir -p "$LOG_DIR"

echo "[sweep] $(hostname) GPU state before launch:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
echo "[sweep] GPUs ${GPUS[*]} x JOBS_PER_GPU=$JOBS_PER_GPU = $(( ${#GPUS[@]} * JOBS_PER_GPU )) slots"
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

    # Round-robin the jobs into one queue per GPU (any number of GPUs).
    NG=${#GPUS[@]}
    declare -a QUEUES=()
    for (( q=0; q<NG; q++ )); do QUEUES[$q]=""; done
    for i in "${!JOBS[@]}"; do
        q=$(( i % NG ))
        QUEUES[$q]="${QUEUES[$q]}${JOBS[$i]}"$'\n'
    done
    counts=""
    for (( q=0; q<NG; q++ )); do
        c=$(printf '%s' "${QUEUES[$q]}" | grep -c . || true)
        counts="$counts gpu${GPUS[$q]}=$c"
    done
    echo "[sweep] ${KEYS[$ei]}: queues$counts (JOBS_PER_GPU=$JOBS_PER_GPU)"
    [[ "$DRY_RUN" == "1" ]] && echo "[sweep] DRY_RUN=1: printing commands only, not launching"

    for (( q=0; q<NG; q++ )); do
        mapfile -t Q < <(printf '%s' "${QUEUES[$q]}")
        gpu_worker "${GPUS[$q]}" "${Q[@]+"${Q[@]}"}" &
    done
    wait   # <-- the sequential barrier: env ei+1 does not start until every job here has exited

    echo "[sweep] ==================== env ${KEYS[$ei]} fully drained (all $n jobs finished) ===================="
    unset JOBS QUEUES
done
echo "[sweep] all envs finished: ${KEYS[*]}"
