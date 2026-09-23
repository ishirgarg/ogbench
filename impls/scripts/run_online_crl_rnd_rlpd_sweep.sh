#!/usr/bin/env bash
# Online CRL + RND exploration bonus + RLPD sweep, packed onto all 8 GPUs of THIS machine
# (a20 / rnn.ist.berkeley.edu -- 8x RTX A5000 24GB, 24 CPU cores, 503 GB RAM).
#
#   12 online envs x 4 RND coefficients x 5 seeds        = 240 RND runs
#    7 of those envs x 5 seeds, bonus OFF (see BASELINE) =  35 baseline runs
#                                                       -----------------
#                                                          275 runs
#
# RND is `agents/online_crl.py`'s random network distillation bonus: a fixed random target
# net f and a trained predictor f_hat over whitened next-observations; the intrinsic reward
# is ||f_hat(s') - f(s')||^2 normalised by the running std of the intrinsic return, fit by a
# separate critic Q_x that the actor maximises with weight `bonus_scale`. That weight IS the
# "RND coefficient" swept here (--agent.bonus_scale, RND_COEFS below). It needs no
# empowerment estimator; it is online-only by design, so add_explore must be `reward`
# (`reward-to-rlpd` is rejected at agent-create time) -- the RLPD rows are masked out of the
# predictor loss, Q_x's backup, the running stats and the actor's bonus term. No annealing
# by default (constant bonus_scale for the whole run); set BONUS_TIME_FRAC to anneal.
#
# BASELINE runs (BASELINE_CELLS) are the SAME command with every `--agent.*explore*` /
# `--agent.bonus_scale` flag dropped: plain online CRL + RLPD, no intrinsic reward and no
# Q_x at all. They are the control the RND coefficients are measured against, and are run
# for the cells whose RLPD dataset is the harder/messier one -- cube-noisy, antmaze-explore
# and every stitch dataset (7 cells x 5 seeds = 35).
#
# ENV / RLPD-DATASET PAIRING (the point of the sweep -- each online env is paired with a
# SPECIFIC offline dataset; the `navigate` / `stitch` / `explore` / `noisy` token in the cell
# key names the DATASET, not the env). NOTE the pointmaze envs are on the TELEPORT maze, so
# they take the pointmaze-teleport datasets, not pointmaze-medium:
#
#   key            online env                                        RLPD dataset          base
#   -------------  ------------------------------------------------  --------------------  ----
#   cube_mg        cube-single-multigoal-online-v0                   cube-single-play-v0
#   cube_mg_noisy  cube-single-multigoal-online-v0                   cube-single-noisy-v0   yes
#   amz_cs_nav     antmaze-medium-corner-sparse-online-v0            antmaze-medium-navigate-v0
#   amz_cs_stitch  antmaze-medium-corner-sparse-online-v0            antmaze-medium-stitch-v0   yes
#   amz_cs_expl    antmaze-medium-corner-sparse-online-v0            antmaze-medium-explore-v0  yes
#   pmz_cs_nav     pointmaze-teleport-corner-sparse-online-v0        pointmaze-teleport-navigate-v0
#   pmz_cs_stitch  pointmaze-teleport-corner-sparse-online-v0        pointmaze-teleport-stitch-v0 yes
#   amz_c_nav      antmaze-medium-corner-all-squares-online-v0       antmaze-medium-navigate-v0
#   amz_c_stitch   antmaze-medium-corner-all-squares-online-v0       antmaze-medium-stitch-v0   yes
#   amz_c_expl     antmaze-medium-corner-all-squares-online-v0       antmaze-medium-explore-v0  yes
#   pmz_c_nav      pointmaze-teleport-corner-all-squares-online-v0   pointmaze-teleport-navigate-v0
#   pmz_c_stitch   pointmaze-teleport-corner-all-squares-online-v0   pointmaze-teleport-stitch-v0 yes
#
# The `-corner-sparse-` envs start in their maze's bottom-left corner (1, 1) = xy (0, 0)
# with only the 3 far corners as goals; the `-corner-all-squares-` envs are the same start
# with the center of every other free cell as a goal (25 tasks on antmaze-medium, 41 on
# pointmaze-teleport, which drops the two teleport-in pads and the (1, 7) trap pocket).
# `cube-single-multigoal` is one fixed cube start with a 5x6 grid of 30 goal positions.
# All five envs come from the repo (ogbench/locomaze, ogbench/manipspace) -- none are
# defined by this script. Every pair below was checked to have matching observation/action
# shapes against its dataset; the preflight re-checks that both `<dataset>.npz` and
# `<dataset>-val.npz` are on disk (the val file is loaded unconditionally by
# `make_env_and_datasets`) and that every env id actually resolves.
#
# PYTHONPATH note: `ogbench` is pip-installed in develop mode pointing at the MAIN checkout
# (/rscratch/ishirgarg/ogbench), not at this worktree, so `import ogbench` would otherwise
# miss the `*-corner-*` / `*-multigoal-*` env registrations this branch carries. REPO_ROOT is
# put on PYTHONPATH to shadow it. Remove that line once the branch is merged into the main
# checkout.
#
# exp_name collision note: get_exp_name() (utils/log_utils.py) is `sd<seed>_<timestamp to the
# second>` outside Slurm, so two same-seed jobs launched in the same wall-clock second would
# collide on one save_dir. Each job gets SLURM_JOB_ID=local-<job index> (log_utils.py's only
# other use of that var) purely to make every exp_name unique -- nothing talks to Slurm.
#
# Usage:
#   bash scripts/run_online_crl_rnd_rlpd_sweep.sh                    # all 275 runs
#   DRY_RUN=1 bash scripts/run_online_crl_rnd_rlpd_sweep.sh          # print commands only
#   CELL_KEYS="cube_mg amz_cs_nav" SEEDS="0 1" bash scripts/...      # subset
#   BASELINE_CELLS="" bash scripts/...                               # RND runs only
#
# Overrides (env vars): DRY_RUN, GPUS, JOBS_PER_GPU, CELL_KEYS, RND_COEFS, SEEDS,
#   BASELINE_CELLS, TOTAL_ENV_STEPS, BONUS_TIME_FRAC, EVAL_EPISODES, EVAL_INTERVAL,
#   SAVE_INTERVAL, OGBENCH_DATASET_DIR, RUN_GROUP_PREFIX.
#
# CAPACITY: 8 GPUs x JOBS_PER_GPU=5 is 40 concurrent JAX processes on a box with 24 CPU
# cores (~0.6 cores each). MuJoCo env stepping and the (CPU) evaluation are the bottleneck,
# not the GPUs -- a solo run does ~160 env steps/s, so expect each 1M-step run to be several
# times slower than that here. 40-way was chosen over 80-way (10/GPU) deliberately: at 0.3
# cores each the whole set finishes LATER despite everything being in flight at once.
set -uo pipefail   # no -e: one run's failure must not kill the other 274
cd "$(dirname "$0")/.."   # -> impls/
REPO_ROOT="$(cd .. && pwd)"

PYTHON=/rscratch/ishirgarg/conda/envs/ogbench/bin/python
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # see "PYTHONPATH note" above
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # mandatory for packing multiple jobs per GPU
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-/rscratch/ishirgarg/.ogbench/data}
# Keep every scratch write (wandb spooling, XLA autotune) off the 216G root volume.
export TMPDIR=${TMPDIR:-/rscratch/ishirgarg/tmp}
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR/wandb_crl_rnd_rlpd"
mkdir -p "$WANDB_DIR"
export JAX_COMPILATION_CACHE_DIR="$REPO_ROOT/impls/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
# 24 cores / 40 concurrent jobs ~= 0.6 each: left unbounded, every process's BLAS/OMP pool
# would try to claim all 24 cores and thrash.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

DRY_RUN=${DRY_RUN:-0}
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
JOBS_PER_GPU=${JOBS_PER_GPU:-5}
RND_COEFS=${RND_COEFS:-"0.1 0.3 1 3"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
BONUS_TIME_FRAC=${BONUS_TIME_FRAC:-}   # empty -> agent default: constant bonus_scale, no annealing
EVAL_EPISODES=${EVAL_EPISODES:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-50000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500000}
LOG_INTERVAL=${LOG_INTERVAL:-5000}
RUN_GROUP_PREFIX=${RUN_GROUP_PREFIX:-crlrnd}
# Resume support. SKIP_FILE: a file with one run NAME per line (the "<cell>_<arm>_s<seed>" the
# script builds below); those jobs are skipped. Use it to relaunch a partially-done sweep
# without redoing finished runs, or to avoid double-launching runs still in flight from an
# earlier driver. MAX_TOTAL_JOBS caps concurrency across the WHOLE machine rather than
# per-GPU, counting main_online.py processes this driver did not start (e.g. orphans left
# by a previous driver) -- without it a new driver would happily add its own
# GPUS*JOBS_PER_GPU on top of whatever is already running. 0 disables the global cap.
SKIP_FILE=${SKIP_FILE:-}
MAX_TOTAL_JOBS=${MAX_TOTAL_JOBS:-0}
declare -A SKIP=()
if [[ -n "$SKIP_FILE" ]]; then
    [[ -f "$SKIP_FILE" ]] || { echo "ERROR: SKIP_FILE=$SKIP_FILE does not exist" >&2; exit 1; }
    while read -r _n; do [[ -n "$_n" ]] && SKIP["$_n"]=1; done < "$SKIP_FILE"
    echo "[sweep] SKIP_FILE=$SKIP_FILE: ${#SKIP[@]} run names will be skipped"
fi
total_running() {  # main_online.py processes owned by this user, machine-wide
    pgrep -u "$(id -u)" -f '[m]ain_online\.py' 2>/dev/null | wc -l
}
await_global_slot() {
    [[ "$MAX_TOTAL_JOBS" -le 0 ]] && return 0
    while (( $(total_running) >= MAX_TOTAL_JOBS )); do sleep 20; done
}

# Cells that ALSO get a no-bonus control arm (plain online CRL + RLPD): the cube-noisy,
# antmaze-explore and stitch datasets. Set BASELINE_CELLS="" to skip them entirely.
BASELINE_CELLS=${BASELINE_CELLS-"cube_mg_noisy amz_cs_stitch amz_cs_expl pmz_cs_stitch amz_c_stitch amz_c_expl pmz_c_stitch"}

# ── The 12 (key, online env, RLPD dataset, episode_length) cells ──────────────────────────
# episode_length empty -> the env's registered horizon (1000 for the mazes, 200 for cube).
ALL_CELL_KEYS=(
    cube_mg cube_mg_noisy
    amz_cs_nav amz_cs_stitch amz_cs_expl
    pmz_cs_nav pmz_cs_stitch
    amz_c_nav amz_c_stitch amz_c_expl
    pmz_c_nav pmz_c_stitch
)
ALL_CELL_ENVS=(
    cube-single-multigoal-online-v0
    cube-single-multigoal-online-v0
    antmaze-medium-corner-sparse-online-v0
    antmaze-medium-corner-sparse-online-v0
    antmaze-medium-corner-sparse-online-v0
    pointmaze-teleport-corner-sparse-online-v0
    pointmaze-teleport-corner-sparse-online-v0
    antmaze-medium-corner-all-squares-online-v0
    antmaze-medium-corner-all-squares-online-v0
    antmaze-medium-corner-all-squares-online-v0
    pointmaze-teleport-corner-all-squares-online-v0
    pointmaze-teleport-corner-all-squares-online-v0
)
ALL_CELL_OFFLINE=(
    cube-single-play-v0
    cube-single-noisy-v0
    antmaze-medium-navigate-v0
    antmaze-medium-stitch-v0
    antmaze-medium-explore-v0
    pointmaze-teleport-navigate-v0
    pointmaze-teleport-stitch-v0
    antmaze-medium-navigate-v0
    antmaze-medium-stitch-v0
    antmaze-medium-explore-v0
    pointmaze-teleport-navigate-v0
    pointmaze-teleport-stitch-v0
)
ALL_CELL_EPISODE_LENGTH=("" "" "" "" "" "" "" "" "" "" "" "")

read -r -a WANTED <<< "${CELL_KEYS:-${ALL_CELL_KEYS[*]}}"
CELL_KEYS_SEL=(); CELL_ENVS=(); CELL_OFFLINE=(); CELL_EPISODE_LENGTH=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_CELL_KEYS[@]}"; do
        if [[ "${ALL_CELL_KEYS[$i]}" == "$w" ]]; then
            CELL_KEYS_SEL+=("${ALL_CELL_KEYS[$i]}"); CELL_ENVS+=("${ALL_CELL_ENVS[$i]}")
            CELL_OFFLINE+=("${ALL_CELL_OFFLINE[$i]}"); CELL_EPISODE_LENGTH+=("${ALL_CELL_EPISODE_LENGTH[$i]}")
            found=1; break
        fi
    done
    (( found )) || { echo "ERROR: unknown cell key '$w' (known: ${ALL_CELL_KEYS[*]})" >&2; exit 1; }
done
for b in $BASELINE_CELLS; do
    found=0
    for k in "${ALL_CELL_KEYS[@]}"; do [[ "$k" == "$b" ]] && { found=1; break; }; done
    (( found )) || { echo "ERROR: unknown BASELINE_CELLS key '$b' (known: ${ALL_CELL_KEYS[*]})" >&2; exit 1; }
done
is_baseline_cell() {  # KEY -> 0 if this cell also gets a no-bonus control arm
    local k
    for k in $BASELINE_CELLS; do [[ "$k" == "$1" ]] && return 0; done
    return 1
}

# ── Preflight: both dataset files present, and every env id actually registered ───────────
for od in "${CELL_OFFLINE[@]}"; do
    for f in "$OGBENCH_DATASET_DIR/$od.npz" "$OGBENCH_DATASET_DIR/$od-val.npz"; do
        [[ -f "$f" ]] || { echo "ERROR: $f is missing (download it first)." >&2; exit 1; }
    done
done
echo "[sweep] preflight: resolving env registrations..."
if ! "$PYTHON" - "${CELL_ENVS[@]}" <<'PY'
import sys
import ogbench  # noqa: F401  (registers the envs)
from utils.env_utils import make_env_only
for name in dict.fromkeys(sys.argv[1:]):
    env = make_env_only(name)
    print(f'  {name}: {len(env.unwrapped.task_infos)} tasks, obs={env.observation_space.shape}, '
          f'act={env.action_space.shape}, horizon={env.spec.max_episode_steps}')
PY
then
    echo "ERROR: env preflight failed (is REPO_ROOT on PYTHONPATH? are the *-corner-* envs registered?)" >&2
    exit 1
fi

LOG_DIR=logs/crl_rnd_rlpd_sweep
mkdir -p "$LOG_DIR"

echo "[sweep] $(hostname) GPU state before launch:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# ── Build the job list ────────────────────────────────────────────────────────────────────
# Seed-outermost so that an interrupted sweep still leaves COMPLETE (env x coef) grids for
# the seeds that did finish, rather than 5 seeds of a few cells and none of the rest.
# COEF=="none" is the no-bonus baseline arm.
NGPU=${#GPUS[@]}
declare -a QUEUES
for ((g = 0; g < NGPU; g++)); do QUEUES[$g]=""; done
n=0
nbase=0
for SEED in $SEEDS; do
    for c in "${!CELL_KEYS_SEL[@]}"; do
        arms="$RND_COEFS"
        if is_baseline_cell "${CELL_KEYS_SEL[$c]}"; then arms="$arms none"; fi
        for COEF in $arms; do
            [[ "$COEF" == "none" ]] && nbase=$((nbase + 1))
            job="$n|${CELL_KEYS_SEL[$c]}|${CELL_ENVS[$c]}|${CELL_OFFLINE[$c]}|${CELL_EPISODE_LENGTH[$c]}|$COEF|$SEED"
            g=$((n % NGPU))
            QUEUES[$g]="${QUEUES[$g]}${job}"$'\n'
            n=$((n + 1))
        done
    done
done
echo "[sweep] built $n jobs ($((n - nbase)) RND + $nbase no-bonus baseline) across ${NGPU} GPU queues (${GPUS[*]}), ${JOBS_PER_GPU} concurrent per GPU"
[[ "$DRY_RUN" == "1" ]] && echo "[sweep] DRY_RUN=1: printing commands only, not launching"

run_one() {  # GPU JOB_SPEC
    local GPU=$1 SPEC=$2
    IFS='|' read -r IDX KEY ENV_NAME OFFLINE EP_LEN COEF SEED <<< "$SPEC"
    local EP_FLAG=(); [[ -n "$EP_LEN" ]] && EP_FLAG=(--episode_length="$EP_LEN")
    local BONUS_FLAG=() GROUP NAME
    if [[ "$COEF" == "none" ]]; then
        # Control arm: plain online CRL + RLPD, no intrinsic reward and no Q_x.
        GROUP="${RUN_GROUP_PREFIX}_${KEY}_nornd"
        NAME="${KEY}_nornd_s${SEED}"
    else
        BONUS_FLAG=(--agent.add_explore=reward --agent.explore_reward=rnd --agent.bonus_scale="$COEF")
        [[ -n "$BONUS_TIME_FRAC" ]] && BONUS_FLAG+=(--agent.explore_reward_time_frac="$BONUS_TIME_FRAC")
        GROUP="${RUN_GROUP_PREFIX}_${KEY}_c${COEF}"
        NAME="${KEY}_c${COEF}_s${SEED}"
    fi
    if [[ -n "${SKIP[$NAME]:-}" ]]; then
        echo "[sweep] gpu=$GPU skipping $NAME (in SKIP_FILE)"
        return 0
    fi
    local LOG="$LOG_DIR/${NAME}.log"
    local cmd=(
        "$PYTHON" -u main_online.py
        --run_group="$GROUP"
        --env_name="$ENV_NAME" --seed="$SEED" --agent=agents/online_crl.py
        --total_env_steps="$TOTAL_ENV_STEPS"
        "${EP_FLAG[@]}"
        --offline_dataset="$OFFLINE"
        "${BONUS_FLAG[@]}"
        --log_interval="$LOG_INTERVAL" --eval_interval="$EVAL_INTERVAL" --save_interval="$SAVE_INTERVAL"
        --eval_episodes="$EVAL_EPISODES" --video_episodes=0
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "gpu=$GPU $NAME (env=$ENV_NAME rlpd=$OFFLINE): ${cmd[*]}"
        return 0
    fi
    await_global_slot
    echo "[sweep] gpu=$GPU launching $NAME env=$ENV_NAME rlpd=$OFFLINE -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU SLURM_JOB_ID="local-$IDX" \
    XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${NAME}" \
        "${cmd[@]}" > "$LOG" 2>&1
    echo "[sweep] gpu=$GPU finished $NAME (exit $?)"
}

gpu_worker() {  # GPU  (job specs on stdin, one per line)
    local GPU=$1
    local spec
    while IFS= read -r spec; do
        [[ -z "$spec" ]] && continue
        while (( $(jobs -rp | wc -l) >= JOBS_PER_GPU )); do
            wait -n
        done
        run_one "$GPU" "$spec" &
        sleep 2   # stagger process startup; SLURM_JOB_ID=local-<idx> already guarantees
                  # exp_name uniqueness even for same-second launches, this is just politeness.
    done
    wait
}

for ((g = 0; g < NGPU; g++)); do
    printf '%s' "${QUEUES[$g]}" | gpu_worker "${GPUS[$g]}" &
done
wait
echo "[sweep] all $n jobs finished"
