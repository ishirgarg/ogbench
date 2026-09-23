#!/usr/bin/env bash
# Distilled empowerment bonus (add_explore=distill-to-rlpd) COMBINED with the independent RND
# bonus (agent.rnd_bonus_scale), RLPD on, on three envs -- packed onto GPUs 3-7 of this machine,
# 6 jobs per GPU = 30 slots = the whole sweep at once (user request, 2026-09-22).
#
# Per env: 2 (alpha_distill, beta_rnd) pairs x 5 seeds = 10 runs; 3 envs = 30 runs.
#   cube_mg   cube-single-multigoal-online-v0                RLPD cube-single-play-v0
#             estimator cube-single-play/sd000_s_38579169.0.20260904_235556 (user-specified)
#             pairs (10, 1), (5, 0.5); no annealing; registered 200-step horizon; MUJOCO_GL=osmesa
#   amz_all   antmaze-medium-corner-all-squares-online-v0    RLPD antmaze-medium-navigate-v0
#             estimator antmaze-medium-navigate/...37866290..._k50 (the k50 run every prior corner
#             sweep used); pairs (30, 3), (15, 1.5); no annealing; registered 1000-step horizon
#   pmt_all   pointmaze-teleport-corner-all-squares-online-v0 RLPD pointmaze-teleport-navigate-v0
#             estimator pointmaze-teleport-navigate/...38390674...; pairs (30, 1), (15, 0.5);
#             the DISTILL term is annealed (explore_reward_time_frac=ANNEAL_FRAC, default 0.5,
#             the fraction every prior annealed distill sweep used); the RND term is NOT annealed.
# Everything else: distill_target=episode_max, emp_entropy_target=False, 1M env steps,
# 100 eval episodes every 25k steps, bonus_grad_diagnostics on seed 0 only.
#
# The two bonuses are independent switches in agents/online_crl.py: `add_explore` picks the
# distilled E' (bonus_scale / explore_reward_time_frac), `rnd_bonus_scale` / `rnd_time_frac`
# add the RND critic Q_rnd next to it. Both envs' registrations (cube-single-multigoal and the
# two corner-all-squares mazes) live only in THIS worktree's ogbench/, hence PYTHONPATH below.
#
# Usage (from this worktree, impls/):
#   bash scripts/run_distill_rnd_local_sweep.sh
#   (detach: setsid nohup bash scripts/... > logs/.../_master.log 2>&1 & )
# Overrides (env vars): DRY_RUN=1, GPUS, JOBS_PER_GPU, SEEDS, ENV_KEYS (subset/reorder of
#   cube_mg amz_all pmt_all), ANNEAL_FRAC, TOTAL_ENV_STEPS, EVAL_EPISODES, EVAL_INTERVAL,
#   DIAG_SEEDS, SAVE_ROOT, LOG_DIR.
set -uo pipefail   # no -e: one run's failure must not kill the others
cd "$(dirname "$0")/.."   # -> impls/

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
# ogbench is an editable install pointing at the MAIN checkout; the envs used here are registered
# in THIS worktree's copy, so shadow the install.
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}   # EGL is broken headless on this box
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # mandatory for packing multiple jobs per GPU
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp   # the local root disk is full (88M free on 2026-09-22)
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR/wandb_distill_rnd"
mkdir -p "$WANDB_DIR"
export JAX_COMPILATION_CACHE_DIR=/nas/ucb/ishirgarg/ogbench/impls/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-3}

grep -q "rnd_bonus_scale" agents/online_crl.py 2>/dev/null || {
    echo "FATAL: agents/online_crl.py has no rnd_bonus_scale -- wrong checkout/branch." >&2; exit 1; }
grep -q "distill_target" agents/online_crl.py 2>/dev/null || {
    echo "FATAL: agents/online_crl.py has no distill_target -- wrong checkout/branch." >&2; exit 1; }
"$PYTHON" - <<'PY' || { echo "FATAL: worktree ogbench does not shadow the editable install / lacks the envs." >&2; exit 1; }
import os, ogbench
assert os.path.realpath(ogbench.__file__).startswith(os.path.realpath(os.path.join(os.getcwd(), '..'))), ogbench.__file__
import gymnasium as gym
for i in ('cube-single-multigoal-v0', 'antmaze-medium-corner-all-squares-v0', 'pointmaze-teleport-corner-all-squares-v0'):
    assert i in gym.registry, i
print('[sweep] ogbench from', ogbench.__file__)
PY

DRY_RUN=${DRY_RUN:-0}
read -r -a GPUS <<< "${GPUS:-3 4 5 6 7}"
JOBS_PER_GPU=${JOBS_PER_GPU:-6}
SEEDS=${SEEDS:-"0 1 2 3 4"}
ANNEAL_FRAC=${ANNEAL_FRAC:-0.5}
DISTILL_TARGET=${DISTILL_TARGET:-episode_max}
DIAG_SEEDS=${DIAG_SEEDS:-"0"}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-25000}
EMP_NUM_SPLUS_SAMPLES=${EMP_NUM_SPLUS_SAMPLES:-64}

MAIN=${MAIN:-/nas/ucb/ishirgarg/ogbench/impls}   # checkpoints/datasets/results live in the main tree
EMP_ROOT=$MAIN/ckpts/final/empowerment_final

ALL_ENV_KEYS=(cube_mg amz_all pmt_all)
ALL_ENV_NAMES=(
    cube-single-multigoal
    antmaze-medium-corner-all-squares
    pointmaze-teleport-corner-all-squares
)
ALL_ENV_ONLINE=(
    cube-single-multigoal-online-v0
    antmaze-medium-corner-all-squares-online-v0
    pointmaze-teleport-corner-all-squares-online-v0
)
ALL_ENV_OFFLINE=(
    cube-single-play-v0
    antmaze-medium-navigate-v0
    pointmaze-teleport-navigate-v0
)
ALL_ENV_EMP_CKPT=(
    "$EMP_ROOT/cube-single-play/sd000_s_38579169.0.20260904_235556"
    "$EMP_ROOT/antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001"
    "$EMP_ROOT/pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836"
)
# "alpha_distill:beta_rnd" pairs per env, in the user's order.
ALL_ENV_PAIRS=(
    "10:1 5:0.5"
    "30:3 15:1.5"
    "30:1 15:0.5"
)
# Distill-term annealing fraction per env ("" -> constant); pointmaze only.
ALL_ENV_ANNEAL=("" "" "$ANNEAL_FRAC")
ALL_ENV_EPISODE_LENGTH=("" "" "")   # all keep their registered horizons (200 / 1000 / 1000)

read -r -a WANTED <<< "${ENV_KEYS:-${ALL_ENV_KEYS[*]}}"
KEYS=(); NAMES=(); ONLINE=(); OFFLINE=(); EMP_CKPT=(); PAIRS=(); ANNEAL=(); EP_LEN=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_ENV_KEYS[@]}"; do
        if [[ "${ALL_ENV_KEYS[$i]}" == "$w" ]]; then
            KEYS+=("${ALL_ENV_KEYS[$i]}"); NAMES+=("${ALL_ENV_NAMES[$i]}"); ONLINE+=("${ALL_ENV_ONLINE[$i]}")
            OFFLINE+=("${ALL_ENV_OFFLINE[$i]}"); EMP_CKPT+=("${ALL_ENV_EMP_CKPT[$i]}"); PAIRS+=("${ALL_ENV_PAIRS[$i]}")
            ANNEAL+=("${ALL_ENV_ANNEAL[$i]}"); EP_LEN+=("${ALL_ENV_EPISODE_LENGTH[$i]}")
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

SAVE_ROOT=${SAVE_ROOT:-$MAIN/exp/distill_rnd_combo}
LOG_DIR=${LOG_DIR:-$MAIN/logs/distill_rnd_combo_sweep}
mkdir -p "$LOG_DIR"

echo "[sweep] $(hostname) GPU state before launch:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
echo "[sweep] GPUs ${GPUS[*]} x JOBS_PER_GPU=$JOBS_PER_GPU = $(( ${#GPUS[@]} * JOBS_PER_GPU )) slots"

run_one() {  # GPU ENV_IDX ALPHA BETA ANNEAL SEED RUN_TAG
    local GPU=$1 ei=$2 ALPHA=$3 BETA=$4 ANN=$5 SEED=$6 RUN_TAG=$7
    local ENV_NAME=${ONLINE[$ei]} OFF=${OFFLINE[$ei]} EMPCK=${EMP_CKPT[$ei]} EPL=${EP_LEN[$ei]} NAME_KEY=${KEYS[$ei]}
    local EPF=(); [[ -n "$EPL" ]] && EPF=(--episode_length="$EPL")
    local DIAG=False
    for s in $DIAG_SEEDS; do [[ "$s" == "$SEED" ]] && DIAG=True; done
    local BONUS=(
        --agent.emp_checkpoint_path="$EMPCK"
        --agent.emp_num_splus_samples="$EMP_NUM_SPLUS_SAMPLES"
        --agent.emp_entropy_target=False
        --agent.add_explore=distill-to-rlpd
        --agent.distill_target="$DISTILL_TARGET"
        --agent.bonus_scale="$ALPHA"
        --agent.rnd_bonus_scale="$BETA"
        --agent.bonus_grad_diagnostics="$DIAG"
    )
    [[ -n "$ANN" ]] && BONUS+=(--agent.explore_reward_time_frac="$ANN")   # distill term only; RND stays constant
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
        IFS='|' read -r ei ALPHA BETA ANN SEED RUN_TAG <<< "$spec"
        run_one "$GPU" "$ei" "$ALPHA" "$BETA" "$ANN" "$SEED" "$RUN_TAG" &
        sleep 1
    done
    wait
}

# All envs at once (30 runs on 30 slots); jobs interleaved across envs so each GPU gets a mix.
declare -a JOBS=()
for ei in "${!KEYS[@]}"; do
    ANN=${ANNEAL[$ei]}
    for pair in ${PAIRS[$ei]}; do
        ALPHA=${pair%%:*}; BETA=${pair##*:}
        RUN_TAG="rlpd_noent_${ANN:+ann${ANN}_}edrlpd${ALPHA}_rnd${BETA}"
        for SEED in $SEEDS; do JOBS+=("$ei|$ALPHA|$BETA|$ANN|$SEED|$RUN_TAG"); done
    done
done
n=${#JOBS[@]}
echo "[sweep] built $n jobs over envs: ${KEYS[*]}"

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
echo "[sweep] queues$counts (JOBS_PER_GPU=$JOBS_PER_GPU)"
[[ "$DRY_RUN" == "1" ]] && echo "[sweep] DRY_RUN=1: printing commands only, not launching"

for (( q=0; q<NG; q++ )); do
    mapfile -t Q < <(printf '%s' "${QUEUES[$q]}")
    gpu_worker "${GPUS[$q]}" "${Q[@]+"${Q[@]}"}" &
done
wait
echo "[sweep] all $n jobs finished"
