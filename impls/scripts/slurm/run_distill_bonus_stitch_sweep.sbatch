#!/bin/bash
#SBATCH --job-name=ogb_distill_stitch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --qos=default
#SBATCH --array=0-79
#SBATCH --output=/nas/ucb/ishirgarg/ogbench/impls/logs/distill_bonus_stitch_slurm/%A_%a.out

# Distilled empowerment bonus sweep for flat online CRL + RLPD (agents/online_crl.py,
# add_explore=distill | distill-to-rlpd, distill_target=episode_max): a second network E'(s, a)
# regressed onto max_k E(s_k) over the row's WHOLE trajectory, added to the actor loss as
# bonus_scale * E'(s, a).
#
# Cluster: rnn.ist.berkeley.edu / BAIR "wild west" Slurm (NOT Savio). No --account, no --partition,
# GPUs requested generically via --gpus, paths under /nas/ucb. The sweep STRUCTURE (cfg decoding,
# preflights, resolve_ckpt, DRY_RUN) mirrors scripts/rail_slurm_distill_bonus_sweep_asoc_cmg_high.sh;
# the environment setup (wandb key file, TMPDIR, JAX cache, dataset dir) mirrors
# scripts/slurm/run_online_crl_seed.sbatch.
#
# TWO CELLS, ONE run per array task (JOBS_PER_GPU=1, CFG = SLURM_ARRAY_TASK_ID):
#   cell 0  antsoccer-arena-far-multigoal   (antsoccer-arena-far-multigoal-online-v0, episode_length 500)
#   cell 1  antmaze-medium-corner-sparse    (antmaze-medium-corner-sparse-online-v0, registered horizon)
#
# Per cell, 45 configs, laid out CONTIGUOUSLY (cell 0 = cfg 0..44, cell 1 = cfg 45..89):
#        CONSTANT alpha  2 modes x 2 alphas {10,30} x 5 seeds = 20 runs
#        ANNEALED alpha  2 modes x 2 alphas {10,30} x 5 seeds = 20 runs
#                        (bonus_scale decays linearly to 0 by ANNEAL_FRAC=0.5 of the env steps, 0 after)
#        + the matched NO-BONUS baseline, 5 seeds                =  5 runs
#        = 20 per cell (distill-to-rlpd ONLY, no baseline), 4 cells = 80 runs -> --array=0-79.
#        CFG / 20 selects the cell; within a cell, CFG%20 < 10 is constant alpha and >= 10 annealed.
#        ALL FOUR cells use the STITCH dataset for RLPD *and* a STITCH-trained estimator
#        (user request 2026-09-22), so they are NOT comparable to the navigate sweeps.
#        NO baseline arm: the user has stitch CRL baselines elsewhere.
#   NOTE: alphas are {10, 30} ONLY here; the rail template sweeps {3,10,30,100} / {10,30,100}.
#   CELL_IDX = CFG / 45        R = CFG % 45
#   R < 20        (constant): MODE = R / 10        ALPHA = (R / 5) % 2   SEED = R % 5
#   20 <= R < 40  (annealed): A = R - 20; MODE = A / 10   ALPHA = (A / 5) % 2   SEED = A % 5
#   R >= 40       (baseline): SEED = R - 40
#     MODE 0 -> distill (online only)   MODE 1 -> distill-to-rlpd (offline + online)
#   INCLUDE_BASELINE=1 would add 5 baseline seeds per cell (100 runs -> --array=0-99;
#   the script prints the required range and the header above must be edited to match).
#
# Run tags (same convention as scripts/run_online_crl.sh):
#   baseline  rlpd_baseline
#   constant  rlpd_noent_ed10 / rlpd_noent_ed30 / rlpd_noent_edrlpd10 / rlpd_noent_edrlpd30
#   annealed  rlpd_noent_ann0.5_ed10 / ... (the ann prefix comes BEFORE the mode tag)
#
# Fixed: RLPD on for the whole run (each cell's own offline dataset, rlpd_frac_time default 1.0),
# 1M env steps, 100 eval episodes every 25k env steps, no videos, emp_entropy_target=False (scalar
# SAC alpha, isolates the bonus), emp_num_splus_samples=64. Annealing is agents/online_crl.py's
# explore_reward_time_frac: only the ACTOR's weight on E' decays, E' itself keeps being fit all run.
# bonus_grad_diagnostics is on for seed 0 of every config only (DIAG_SEEDS): it logs the CRL-vs-bonus
# actor gradient norms (training/bonus_grad/*) at the price of three extra actor backward passes per
# update, so the other four seeds run without it. Only the final checkpoint is kept
# (save_interval = total steps): a params file is ~115 MB because it carries the frozen estimator.
# No --run_group is ever passed (project rule).
#
# HEADER DEVIATIONS from scripts/slurm/run_online_crl_seed.sbatch, both deliberate:
#   --mem=32gb (not 16gb): a run holds the 1M-row offline dataset + a 1M-row replay buffer
#     (~3-5 GiB RSS) and the empowerment pass over the calibration rows allocates on top; the
#     equivalent Savio runs were OOM-killed at the 8000M default.
#   --time=20:00:00 (not 08:00:00): 1M env steps with RLPD does not finish in 8 hours.
#
# WHICH CHECKOUT -- this matters. The distilled-bonus code exists ONLY on the worktree branch
# claude/empowerment-bonus-network-crl-daeab5; the main checkout at /nas/ucb/ishirgarg/ogbench is on
# master and has none of it. So REPO_DIR defaults to the WORKTREE, and PYTHONPATH=$REPO_DIR so the
# worktree's ogbench/ package SHADOWS the editable install (which points at the main checkout) --
# the three corner/far envs are registered only in the worktree's ogbench/locomaze/__init__.py.
# Both facts are preflighted below and fail loudly.
# The estimator checkpoints under CKPT_ROOT and the datasets under OGBENCH_DATASET_DIR live in the
# MAIN checkout / home tree: that is intentional, the worktree has no ckpts/ and no dataset cache.
# CKPT_ROOT must be WRITABLE: the estimator's E over the offline dataset is cached next to the
# checkpoint (<ckpt>/empowerment_values/<dataset>_e<epoch>_n64_fast1_seed<seed>.npy), computed by the
# first run of each (cell, seed) and written atomically.
#
# Submit (the Slurm --output directory is NOT created by sbatch -- mkdir it first or every task dies
# before running a line):
#   mkdir -p /nas/ucb/ishirgarg/ogbench/impls/logs/distill_bonus_corner_slurm
#   cd /nas/ucb/ishirgarg/ogbench/.claude/worktrees/empowerment-bonus-network-crl-daeab5/impls
#   sbatch scripts/slurm/run_distill_bonus_corner_sweep.sbatch
# Overrides (env vars): REPO_DIR, CKPT_ROOT, SAVE_ROOT, PYTHON, OGBENCH_DATASET_DIR,
#                       TOTAL_ENV_STEPS, JOBS_PER_GPU (keep --array in step), INCLUDE_BASELINE,
#                       DIAG_SEEDS, ANNEAL_FRAC,
#                       DRY_RUN (=1: preflight and print each run's flags, launch nothing).

set -uo pipefail

# -----------------------------
# Paths / environment (rnn conventions, from scripts/slurm/run_online_crl_seed.sbatch)
# -----------------------------
# The worktree branch, NOT /nas/ucb/ishirgarg/ogbench (master has no distill code).
REPO_DIR=${REPO_DIR:-/nas/ucb/ishirgarg/ogbench/.claude/worktrees/empowerment-bonus-network-crl-daeab5}
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}

export MUJOCO_GL=${MUJOCO_GL:-egl}
# Make the worktree's ogbench/ win over the editable install that points at the main checkout:
# the *-far-multigoal / *-corner-sparse envs are registered only in this branch's locomaze/__init__.py.
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
# Slurm jobs do NOT see your rnn home dir, so a ~/.netrc wandb login there is invisible here. The key
# lives in a private (chmod 600) file OUTSIDE the repo -- never inline it in a tracked script.
WANDB_KEY_FILE=${WANDB_KEY_FILE:-/nas/ucb/ishirgarg/.wandb_api_key}
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[distill_corner] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
fi
# Disposable wandb staging data -> the 60-day scratch area, not the permanent NAS quota.
export TMPDIR=${TMPDIR:-/nas/ttl=60d/ishirgarg/tmp}
mkdir -p "$TMPDIR"
# Shared, persistent XLA compile cache on the NAS so 90 array tasks (likely on different nodes/GPU
# types) don't each pay a cold-start compile.
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/nas/ucb/ishirgarg/ogbench/impls/.jax_cache}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
# The per-fusion AUTOTUNE sub-cache inside the compilation cache dir is NOT safe for concurrent
# writers (jobs 1194122 / 1194127, 2026-09-04: NOT_FOUND on a temp .textproto another job renamed
# away). Give each job its own autotune dir; the module cache above stays shared.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
# Grow GPU memory on demand: --gpus=1 gives a whole GPU, but the cluster's A4000/A6000/A100 mix means
# a fixed preallocation fraction sized for one card is wrong for another.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
# A Slurm job's $HOME is the compute node's own (unsynced) home, so OGBench's default ~/.ogbench/data
# would miss and trigger a download -- and compute nodes may have no internet egress. Point at the
# NAS-resident cache every job shares.
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}

# Pretrained empowerment_skill estimator runs (they live in the MAIN checkout's ckpts/ tree).
CKPT_ROOT=${CKPT_ROOT:-/nas/ucb/ishirgarg/ogbench/impls/ckpts/final/empowerment_final}
SAVE_ROOT=${SAVE_ROOT:-/nas/ucb/ishirgarg/ogbench/impls/exp/distill_bonus_stitch_slurm}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
INCLUDE_BASELINE=${INCLUDE_BASELINE:-0}   # user has stitch baselines elsewhere (2026-09-22)
DIAG_SEEDS=${DIAG_SEEDS:-"0"}   # seeds (space separated) that log bonus_grad/*; "" -> none
ANNEAL_FRAC=${ANNEAL_FRAC:-0.5} # annealed arm: bonus_scale -> 0 linearly by this fraction of TOTAL_ENV_STEPS

# -----------------------------
# Sweep definitions
# -----------------------------
# CELL_RUNS are the pretrained 50-skill empowerment_skill runs (the E(s) estimators), by path under
# $CKPT_ROOT. CELL_DATASETS is both the RLPD dataset and the env_name the estimator was trained on;
# it is checked against the checkpoint's own flags.json so a wrong run name fails loudly.
CELL_NAMES=(
    antmaze-medium-corner-sparse
    antmaze-medium-corner-all-squares
    pointmaze-teleport-corner-sparse
    pointmaze-teleport-corner-all-squares
)
CELL_ENVS=(
    antmaze-medium-corner-sparse-online-v0
    antmaze-medium-corner-all-squares-online-v0
    pointmaze-teleport-corner-sparse-online-v0
    pointmaze-teleport-corner-all-squares-online-v0
)
CELL_DATASETS=(
    antmaze-medium-stitch-v0
    antmaze-medium-stitch-v0
    pointmaze-teleport-stitch-v0
    pointmaze-teleport-stitch-v0
)
CELL_RUNS=(
    "antmaze-medium-stitch/sd000_s_37866313.0.20260821_030454_k50_s0.01_bc0.001"
    "antmaze-medium-stitch/sd000_s_37866313.0.20260821_030454_k50_s0.01_bc0.001"
    "pointmaze-teleport-stitch/sd000_s_38390675.0.20260901_154836"
    "pointmaze-teleport-stitch/sd000_s_38390675.0.20260901_154836"
)
# Empty -> use the env's registered horizon (antmaze-medium-corner-sparse); antsoccer needs 500.
CELL_EPISODE_LENGTHS=("" "" "" "")   # all four keep their registered 1000-step horizon
MODES=(distill-to-rlpd)
MODE_TAGS=(edrlpd)   # same tags as scripts/run_online_crl.sh
ALPHAS=(10 30)          # constant-alpha arm
ANNEAL_ALPHAS=(10 30)   # annealed arm (initial bonus_scale)
NUM_SEEDS=5
DISTILL_TARGET=episode_max

NUM_CELLS=${#CELL_NAMES[@]}                           # 4
PER_ALPHA=$NUM_SEEDS                                  # 5
PER_MODE=$(( ${#ALPHAS[@]} * PER_ALPHA ))             # 10
CONST_PER_CELL=$(( ${#MODES[@]} * PER_MODE ))         # 20
ANN_PER_MODE=$(( ${#ANNEAL_ALPHAS[@]} * NUM_SEEDS ))  # 10
ANN_PER_CELL=$(( ${#MODES[@]} * ANN_PER_MODE ))       # 20
BASE_PER_CELL=0
if [[ "$INCLUDE_BASELINE" == "1" ]]; then BASE_PER_CELL=$NUM_SEEDS; fi   # 5
PER_CELL=$(( CONST_PER_CELL + ANN_PER_CELL + BASE_PER_CELL ))            # 45
NUM_CONFIGS=$(( NUM_CELLS * PER_CELL ))                                  # 90

# `set -u` only warns inside arithmetic, so a renamed array would silently launch ZERO runs.
(( CONST_PER_CELL == 10 && ANN_PER_CELL == 10 && NUM_CELLS == 4 )) || {
    echo "FATAL: CONST_PER_CELL=$CONST_PER_CELL ANN_PER_CELL=$ANN_PER_CELL NUM_CELLS=$NUM_CELLS," >&2
    echo "       expected 10 / 10 / 4 (sweep arrays edited? fix the header comment + --array too)." >&2; exit 1; }
NEED_TASKS=$(( (NUM_CONFIGS + JOBS_PER_GPU - 1) / JOBS_PER_GPU ))
(( NEED_TASKS == 80 )) || echo "NOTE: $NUM_CONFIGS configs at $JOBS_PER_GPU per GPU need --array=0-$(( NEED_TASKS - 1 )) (header says 0-79)." >&2

# -----------------------------
# Preflight
# -----------------------------
cd "$IMPLS_DIR" || { echo "FATAL: no ogbench checkout at REPO_DIR=$REPO_DIR (expected $IMPLS_DIR)" >&2; exit 1; }
[[ -f main_online.py ]] || {
    echo "FATAL: $IMPLS_DIR is not the impls/ directory (no main_online.py). Set REPO_DIR to the" >&2
    echo "       worktree root (claude/empowerment-bonus-network-crl-daeab5), not to impls/." >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "FATAL: PYTHON=$PYTHON is not executable (conda env moved?)." >&2; exit 1; }
grep -q "distill_target" agents/online_crl.py 2>/dev/null || {
    echo "FATAL: $IMPLS_DIR/agents/online_crl.py has no distill_target -- this checkout predates the" >&2
    echo "       distilled bonus. REPO_DIR must be the claude/empowerment-bonus-network-crl-daeab5" >&2
    echo "       worktree (the main checkout at /nas/ucb/ishirgarg/ogbench is on master)." >&2; exit 1; }
grep -q "distill_target" utils/online_rollout.py utils/rlpd.py 2>/dev/null || {
    echo "FATAL: utils/online_rollout.py / utils/rlpd.py lack the distill_target row fields (partial pull?)." >&2; exit 1; }
# `import ogbench` MUST resolve inside REPO_DIR: the editable install points at the main checkout,
# whose master ogbench/ does not register these envs. PYTHONPATH above is what makes this true.
"$PYTHON" - "$REPO_DIR" <<'PY' || {
import os, sys
import ogbench
p = os.path.realpath(os.path.dirname(ogbench.__file__))
root = os.path.realpath(sys.argv[1])
if not p.startswith(root + os.sep):
    print('resolved ogbench package: %s' % p, file=sys.stderr)
    sys.exit(1)
print('ogbench package resolves inside REPO_DIR: %s' % p)
PY
    echo "FATAL: 'import ogbench' does NOT resolve inside REPO_DIR=$REPO_DIR (see the path above)." >&2
    echo "       The editable install shadows the worktree -- PYTHONPATH=\$REPO_DIR must come first." >&2; exit 1; }
# main_online.py derives the *-online-v0 env from the registered *-v0 id, so check those.
"$PYTHON" - "${CELL_ENVS[@]}" <<'PY' || {
import sys
import gymnasium, ogbench.locomaze  # noqa: F401
missing = [e for e in (a.replace('-online-v0', '-v0') for a in sys.argv[1:]) if e not in gymnasium.registry]
if missing:
    print('missing env ids: %s' % ', '.join(missing), file=sys.stderr)
    sys.exit(1)
PY
    echo "FATAL: the sweep's env ids are not registered by the ogbench package on PYTHONPATH (see above)." >&2
    echo "       They exist only in $REPO_DIR/ogbench/locomaze/__init__.py on this branch." >&2; exit 1; }
[[ -d "$CKPT_ROOT" ]] || {
    echo "FATAL: CKPT_ROOT=$CKPT_ROOT does not exist. It should be the empowerment_final tree in the" >&2
    echo "       MAIN checkout, holding <env>/<run>/flags.json + params_*.pkl." >&2; exit 1; }
echo "using REPO_DIR=$REPO_DIR  CKPT_ROOT=$CKPT_ROOT  SAVE_ROOT=$SAVE_ROOT  PYTHON=$PYTHON"

resolve_ckpt() {
    local d="$CKPT_ROOT/$1"
    if [[ ! -f "$d/flags.json" ]]; then
        echo "FATAL: no flags.json at $d" >&2
        echo "       Expected the pretrained estimator run '$1' under CKPT_ROOT=$CKPT_ROOT." >&2
        echo "       Available there: $(ls -d "$CKPT_ROOT/$(dirname "$1")"/* 2>/dev/null | head -3 | xargs -n1 basename 2>/dev/null | tr '\n' ' ')..." >&2
        return 1
    fi
    echo "$d"
}

# -----------------------------
# Launch up to JOBS_PER_GPU runs concurrently on this task's single GPU (JOBS_PER_GPU=1 here)
# -----------------------------
IDX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit with sbatch, or set it for DRY_RUN=1}
declare -a PIDS=() TAGS=()

for (( j=0; j<JOBS_PER_GPU; j++ )); do
    CFG=$(( IDX * JOBS_PER_GPU + j ))
    (( CFG < NUM_CONFIGS )) || break

    # Cell-major layout: each cell's 45 configs are contiguous (cell 0 = 0..44, cell 1 = 45..89).
    CELL_IDX=$(( CFG / PER_CELL ))
    R=$(( CFG % PER_CELL ))
    ANNEAL=""
    if (( R < CONST_PER_CELL )); then
        MODE_IDX=$(( (R / PER_MODE) % ${#MODES[@]} ))
        ALPHA_IDX=$(( (R / PER_ALPHA) % ${#ALPHAS[@]} ))
        SEED=$(( R % NUM_SEEDS ))
        MODE=${MODES[$MODE_IDX]}
        ALPHA=${ALPHAS[$ALPHA_IDX]}
        RUN_TAG="rlpd_noent_${MODE_TAGS[$MODE_IDX]}${ALPHA}"
    elif (( R < CONST_PER_CELL + ANN_PER_CELL )); then
        A=$(( R - CONST_PER_CELL ))
        MODE_IDX=$(( (A / ANN_PER_MODE) % ${#MODES[@]} ))
        ALPHA_IDX=$(( (A / NUM_SEEDS) % ${#ANNEAL_ALPHAS[@]} ))
        SEED=$(( A % NUM_SEEDS ))
        MODE=${MODES[$MODE_IDX]}
        ALPHA=${ANNEAL_ALPHAS[$ALPHA_IDX]}
        ANNEAL=$ANNEAL_FRAC
        RUN_TAG="rlpd_noent_ann${ANNEAL}_${MODE_TAGS[$MODE_IDX]}${ALPHA}"   # tag order as scripts/run_online_crl.sh
    else
        SEED=$(( R - CONST_PER_CELL - ANN_PER_CELL ))
        MODE=none
        ALPHA=0
        RUN_TAG="rlpd_baseline"
    fi

    CELL=${CELL_NAMES[$CELL_IDX]}
    CELL_RUN=${CELL_RUNS[$CELL_IDX]}
    OFFLINE_DATASET=${CELL_DATASETS[$CELL_IDX]}
    ENV_NAME=${CELL_ENVS[$CELL_IDX]}
    EPISODE_LENGTH=${CELL_EPISODE_LENGTHS[$CELL_IDX]}

    # RLPD dataset: honour OGBENCH_DATASET_DIR, else the NAS cache.
    DS_DIR=${OGBENCH_DATASET_DIR:-}
    if [[ -z "$DS_DIR" || ! -f "$DS_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "FATAL: RLPD needs $OFFLINE_DATASET.npz but it is not in OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-<unset>}." >&2
        echo "       Pre-download it on the login node; compute nodes may have no internet egress." >&2
        exit 1
    fi

    EMP_CKPT=""
    BONUS_FLAGS=()
    if [[ "$MODE" != "none" ]]; then
        EMP_CKPT=$(resolve_ckpt "$CELL_RUN") || exit 1
        compgen -G "$EMP_CKPT/params_*.pkl" > /dev/null || { echo "FATAL: no params_*.pkl in $EMP_CKPT" >&2; exit 1; }
        [[ -w "$EMP_CKPT" ]] || { echo "FATAL: $EMP_CKPT is not writable (the offline E cache is written under it)." >&2; exit 1; }
        read -r AGENT_NAME EMP_DATASET < <("$PYTHON" - "$EMP_CKPT" <<'PY'
import json, sys
f = json.load(open(sys.argv[1] + '/flags.json'))
print(f['agent']['agent_name'], f['env_name'])
PY
)
        [[ "$AGENT_NAME" == "empowerment_skill" ]] || { echo "FATAL: $EMP_CKPT is a $AGENT_NAME run, expected empowerment_skill" >&2; exit 1; }
        [[ "$EMP_DATASET" == "$OFFLINE_DATASET" ]] || {
            echo "FATAL: $EMP_CKPT was trained on $EMP_DATASET, but cell '$CELL' expects $OFFLINE_DATASET." >&2
            echo "       The run name in CELL_RUNS is wrong or names a different run." >&2; exit 1; }
        DIAG=False
        for s in $DIAG_SEEDS; do if [[ "$s" == "$SEED" ]]; then DIAG=True; fi; done
        BONUS_FLAGS=(
            --agent.emp_checkpoint_path="$EMP_CKPT"
            --agent.emp_num_splus_samples=64
            --agent.emp_entropy_target=False
            --agent.add_explore="$MODE"
            --agent.distill_target="$DISTILL_TARGET"
            --agent.bonus_scale="$ALPHA"
            --agent.bonus_grad_diagnostics="$DIAG"
        )
        if [[ -n "$ANNEAL" ]]; then BONUS_FLAGS+=(--agent.explore_reward_time_frac="$ANNEAL"); fi
    fi

    SAVE_DIR="$SAVE_ROOT/$CELL/$RUN_TAG"
    [[ "${DRY_RUN:-0}" == "1" ]] || mkdir -p "$SAVE_DIR"   # DRY_RUN must not litter the exp/ tree
    RUN_LOG="$SAVE_DIR/launch_seed${SEED}_${SLURM_JOB_ID:-local}.log"

    EP_FLAG=()
    if [[ -n "$EPISODE_LENGTH" ]]; then EP_FLAG=(--episode_length="$EPISODE_LENGTH"); fi

    TAG="cfg${CFG}[$CELL mode=$MODE alpha=$ALPHA anneal=${ANNEAL:-none} seed=$SEED tag=$RUN_TAG]"
    echo "LAUNCH $TAG"
    echo "   env=$ENV_NAME ep_len=${EPISODE_LENGTH:-<registered>} rlpd=$OFFLINE_DATASET from $DS_DIR"
    echo "   estimator=${EMP_CKPT:-<none, baseline>}  log=$RUN_LOG"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then   # DRY_RUN=1 SLURM_ARRAY_TASK_ID=<n> bash <this file>: preflight + print, launch nothing
        echo "   DRY_RUN: main_online.py --env_name=$ENV_NAME --seed=$SEED --save_dir=$SAVE_DIR ${BONUS_FLAGS[*]:-} --offline_dataset=$OFFLINE_DATASET ${EP_FLAG[*]:-}"
        continue
    fi

    # main_online.py names a run by seed + SLURM_JOB_ID + timestamp; give each packed run its own id
    # so runs of one array task at JOBS_PER_GPU > 1 cannot share a run dir.
    SLURM_JOB_ID="${SLURM_JOB_ID:-local}.cfg${CFG}" "$PYTHON" -u main_online.py \
        --env_name="$ENV_NAME" \
        --seed="$SEED" \
        --save_dir="$SAVE_DIR" \
        --agent=agents/online_crl.py \
        "${BONUS_FLAGS[@]}" \
        --offline_dataset="$OFFLINE_DATASET" \
        --total_env_steps="$TOTAL_ENV_STEPS" \
        "${EP_FLAG[@]}" \
        --log_interval=5000 \
        --eval_interval=25000 \
        --eval_episodes=100 \
        --video_episodes=0 \
        --save_interval="$TOTAL_ENV_STEPS" \
        > "$RUN_LOG" 2>&1 &

    PIDS+=($!); TAGS+=("$TAG")
done

echo "array task $IDX on $(hostname): launched ${#PIDS[@]} run(s) on one GPU; waiting."

# Wait for all of them, reporting each individually -- a bare `wait` would hide which failed.
RC=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "OK   ${TAGS[$i]}"
    else
        code=$?
        echo "FAIL ${TAGS[$i]} (exit $code) -- see its log above" >&2
        RC=1
    fi
done
exit $RC
