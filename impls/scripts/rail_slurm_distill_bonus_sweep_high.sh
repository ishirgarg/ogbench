#!/bin/bash
#SBATCH --job-name=ogb_distill_bonus
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --array=0-44

# Distilled empowerment bonus sweep for flat online CRL + RLPD (agents/online_crl.py,
# add_explore=distill | distill-to-rlpd, distill_target=episode_max): a second network E'(s, a)
# regressed onto max_k E(s_k) over the row's WHOLE trajectory, added to the actor loss as
# bonus_scale * E'(s, a). BRC/Savio (rail), HIGH-priority QOS, FIVE runs packed per GPU.
#
# Sweep: CONSTANT alpha  3 cells x 2 modes x 4 alphas {3,10,30,100} x 5 seeds = 120 runs
#        ANNEALED alpha  3 cells x 2 modes x 3 alphas {10,30,100}   x 5 seeds =  90 runs
#                        (bonus_scale decays linearly to 0 by ANNEAL_FRAC=0.5 of the env steps, 0 after)
#        + the matched NO-BONUS baseline, 3 cells x 5 seeds = 15 runs  (same code, same eval cadence)
#        = 225 runs, 5 per GPU -> 45 array tasks (0..44), every task full.
#
#   CFG = SLURM_ARRAY_TASK_ID * 5 + j     for j in 0..4, skipping CFG >= 225
#   (with 5 seeds and 5 runs per GPU, one array task = the 5 seeds of ONE config)
#   CFG < 120        (constant): CELL = CFG / 40   MODE = (CFG / 20) % 2   ALPHA = (CFG / 5) % 4   SEED = CFG % 5
#   120 <= CFG < 210 (annealed): A = CFG - 120;  CELL = A / 30   MODE = (A / 15) % 2   ALPHA = (A / 5) % 3   SEED = A % 5
#   CFG >= 210       (baseline): B = CFG - 210;  CELL = B / 5                                                SEED = B % 5
#     CELL 0 -> antsoccer-arena-center-online-v0     RLPD antsoccer-arena-navigate-v0    (ep len 500)
#     CELL 1 -> pointmaze-teleport-center-online-v0  RLPD pointmaze-teleport-stitch-v0
#     CELL 2 -> cube-single-center-online-v0         RLPD cube-single-play-v0
#     MODE 0 -> distill          (online only: E' fit on online rows, actor term on online rows only)
#     MODE 1 -> distill-to-rlpd  (offline + online: E' fit on online + RLPD rows, actor term on every row)
#     ALPHA constant 0/1/2/3 -> bonus_scale 3 / 10 / 30 / 100;  annealed 0/1/2 -> initial bonus_scale 10 / 30 / 100
#   Set INCLUDE_BASELINE=0 to skip the 15 baseline runs (tasks 42..44 then launch nothing and exit 0).
#
# Fixed: RLPD on for the whole run (each cell's own offline dataset, rlpd_frac_time default 1.0),
# 1M env steps, 100 eval episodes every 25k env steps, no videos, emp_entropy_target=False (scalar
# SAC alpha, isolates the bonus), emp_num_splus_samples=64. Annealing is agents/online_crl.py's
# explore_reward_time_frac: only the ACTOR's weight on E' decays, E' itself keeps being fit all run.
# bonus_grad_diagnostics is on for seed 0 of every sweep config only (DIAG_SEEDS): it logs the
# CRL-vs-bonus actor gradient norms (training/bonus_grad/*) over the full run at the price of
# three extra actor backward passes per update, so the other four seeds run without it.
# Only the final checkpoint is kept (save_interval = total steps): a params file is ~115 MB
# because it carries the frozen estimator, and 225 runs x 10 saves would be ~250 GB.
#
# The estimator's E over the offline dataset is cached next to the estimator checkpoint
# (<ckpt>/empowerment_values/<dataset>_e<epoch>_n64_fast1_seed<seed>.npy). On a cluster that has
# never run the flat-CRL empowerment code the five per-seed files do not exist yet: the first run
# of each (cell, seed) computes them at start-up (a few minutes, written atomically so packed /
# concurrent runs cannot read a partial file). $CKPT_ROOT must therefore be writable.
#
# GPU PACKING (5 runs share this task's single GPU) REQUIRES, both handled below:
#   * XLA_PYTHON_CLIENT_PREALLOCATE=false -- JAX otherwise grabs 75% of the GPU on first use and
#     the other four processes OOM immediately.
#   * one log file per run -- five processes sharing the SBATCH --output would interleave.
# Measured on rnn with 12 of these runs on one A6000: ~1.1 GiB GPU each. Host RAM is the number
# to watch: each run holds the 1M-row offline dataset + a 1M-row replay buffer (~3-5 GiB RSS).
# 5 MuJoCo+JAX processes on 4 cores is ~0.8 core each (env stepping + CPU evals are CPU-bound), a
# deliberate oversubscription: 4 cores is the CPU:GPU ratio every rail script here uses, so it is
# known to schedule. 5 x ~1.1 GiB = ~5.5 GiB of the 24 GB A5000; host RSS ~15-25 GiB per task.
#
# BEFORE SUBMITTING, confirm on BRC (written on the rnn side, cannot see /global/*; each item is
# preflighted below and fails with a specific message):
#   1. the checkout at $IMPLS_DIR has the distill code (agents/online_crl.py with `distill_target`),
#      i.e. branch claude/empowerment-bonus-network-crl-daeab5 (or master once merged) is pulled,
#   2. the three estimator exp_names in CELL_RUNS exist under $CKPT_ROOT,
#   3. the three OGBench datasets are present for RLPD (compute nodes may have no internet),
#   4. the three *-center-online-v0 envs are registered by the installed ogbench package,
#   5. `python` on the compute node is the env with jax/flax/ogbench installed.
#
# Submit from the impls/ directory:  cd <repo>/impls && sbatch scripts/rail_slurm_distill_bonus_sweep_high.sh
# Overrides (env vars): CKPT_ROOT, SAVE_ROOT, SCRATCH_ROOT, IMPLS_DIR, OGBENCH_DATASET_DIR,
#                       TOTAL_ENV_STEPS, JOBS_PER_GPU (keep --array in step), INCLUDE_BASELINE, DIAG_SEEDS,
#                       ANNEAL_FRAC, DRY_RUN (=1: preflight and print each run's flags, launch nothing).

set -uo pipefail

export MUJOCO_GL=egl
# Mandatory for GPU packing -- see above.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# The checkout is wherever you ran `sbatch` from (same convention as the other rail scripts).
IMPLS_DIR=${IMPLS_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
# Everything BRC-side lives under scratch (home quota is small).
SCRATCH_ROOT=${SCRATCH_ROOT:-/global/scratch/users/ishirgarg/ogbench}
SAVE_ROOT=${SAVE_ROOT:-$SCRATCH_ROOT/distill_bonus}
# Pretrained estimator runs live in the flat main.py save tree on BRC, addressed by exp_name.
CKPT_ROOT=${CKPT_ROOT:-$SCRATCH_ROOT/OGBench/Debug}
export WANDB_DIR=${WANDB_DIR:-$SCRATCH_ROOT}
mkdir -p "$WANDB_DIR"
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
JOBS_PER_GPU=${JOBS_PER_GPU:-5}
INCLUDE_BASELINE=${INCLUDE_BASELINE:-1}
DIAG_SEEDS=${DIAG_SEEDS:-"0"}   # seeds (space separated) that log bonus_grad/*; "" -> none
ANNEAL_FRAC=${ANNEAL_FRAC:-0.5}   # annealed arm: bonus_scale -> 0 linearly by this fraction of TOTAL_ENV_STEPS

# -----------------------------
# Sweep definitions
# -----------------------------
# The three pretrained 50-skill empowerment_skill runs (the E(s) estimators), by exp_name under
# $CKPT_ROOT -- the same runs as ckpts/final/empowerment_final/<env>/ on rnn. CELL_DATASETS is
# both the RLPD dataset and the env_name the estimator was trained on; it is checked against the
# checkpoint's own flags.json so a wrong exp_name fails loudly.
CELL_RUNS=(
    "sd000_s_38390672.0.20260901_154836"
    "sd000_s_38390675.0.20260901_154836"
    "sd000_s_38624008.0.20260908_013305"
)
CELL_NAMES=(
    antsoccer-arena-center
    pointmaze-teleport-center
    cube-single-center
)
CELL_DATASETS=(
    antsoccer-arena-navigate-v0
    pointmaze-teleport-stitch-v0
    cube-single-play-v0
)
CELL_ENVS=(
    antsoccer-arena-center-online-v0
    pointmaze-teleport-center-online-v0
    cube-single-center-online-v0
)
# antsoccer is registered at 1000 but every online run in this project uses 500 for it.
CELL_EPISODE_LENGTHS=(500 "" "")
MODES=(distill distill-to-rlpd)
MODE_TAGS=(ed edrlpd)   # same tags as scripts/run_online_crl.sh
ALPHAS=(3 10 30 100)          # constant-alpha arm
ANNEAL_ALPHAS=(10 30 100)     # annealed arm (initial bonus_scale)
NUM_SEEDS=5
DISTILL_TARGET=episode_max

NUM_CELLS=${#CELL_NAMES[@]}
PER_ALPHA=$NUM_SEEDS                                  # 5
PER_MODE=$(( ${#ALPHAS[@]} * PER_ALPHA ))             # 20
PER_CELL=$(( ${#MODES[@]} * PER_MODE ))               # 40
NUM_CONST=$(( NUM_CELLS * PER_CELL ))                 # 120
ANN_PER_MODE=$(( ${#ANNEAL_ALPHAS[@]} * NUM_SEEDS ))  # 15
ANN_PER_CELL=$(( ${#MODES[@]} * ANN_PER_MODE ))       # 30
NUM_ANNEAL=$(( NUM_CELLS * ANN_PER_CELL ))            # 90
NUM_SWEEP=$(( NUM_CONST + NUM_ANNEAL ))               # 210
NUM_BASELINE=0
if [[ "$INCLUDE_BASELINE" == "1" ]]; then NUM_BASELINE=$(( NUM_CELLS * NUM_SEEDS )); fi   # 15
NUM_CONFIGS=$(( NUM_SWEEP + NUM_BASELINE ))           # 225

# `set -u` only warns inside arithmetic, so a renamed array would silently launch ZERO runs.
(( NUM_CONST == 120 && NUM_ANNEAL == 90 )) || { echo "FATAL: NUM_CONST=$NUM_CONST NUM_ANNEAL=$NUM_ANNEAL, expected 120 / 90 (sweep arrays edited? fix the header + --array)." >&2; exit 1; }
NEED_TASKS=$(( (NUM_CONFIGS + JOBS_PER_GPU - 1) / JOBS_PER_GPU ))
(( NEED_TASKS == 45 )) || echo "NOTE: $NUM_CONFIGS configs at $JOBS_PER_GPU per GPU need --array=0-$(( NEED_TASKS - 1 )) (header says 0-44)." >&2

cd "$IMPLS_DIR" || { echo "FATAL: no ogbench checkout at IMPLS_DIR=$IMPLS_DIR" >&2; exit 1; }
[[ -f main_online.py ]] || {
    echo "FATAL: $IMPLS_DIR is not the impls/ directory (no main_online.py). Submit with" >&2
    echo "       'cd <repo>/impls && sbatch scripts/$(basename "$0")', or set IMPLS_DIR." >&2; exit 1; }
grep -q "distill_target" agents/online_crl.py 2>/dev/null || {
    echo "FATAL: $IMPLS_DIR/agents/online_crl.py has no distill_target -- this checkout predates the distilled" >&2
    echo "       bonus. Pull branch claude/empowerment-bonus-network-crl-daeab5 (or master once it is merged)." >&2; exit 1; }
grep -q "distill_target" utils/online_rollout.py utils/rlpd.py 2>/dev/null || {
    echo "FATAL: utils/online_rollout.py / utils/rlpd.py lack the distill_target row fields (partial pull?)." >&2; exit 1; }
[[ -d "$CKPT_ROOT" ]] || {
    echo "FATAL: CKPT_ROOT=$CKPT_ROOT does not exist. It should be the main.py save tree holding the" >&2
    echo "       pretrained runs, i.e. <save_dir>/<project>/<run_group> (e.g. .../ogbench/OGBench/Debug)." >&2; exit 1; }
echo "using IMPLS_DIR=$IMPLS_DIR  CKPT_ROOT=$CKPT_ROOT  SAVE_ROOT=$SAVE_ROOT"

resolve_ckpt() {
    local d="$CKPT_ROOT/$1"
    if [[ ! -f "$d/flags.json" ]]; then
        echo "FATAL: no flags.json at $d" >&2
        echo "       Expected the pretrained estimator run '$1' directly under CKPT_ROOT=$CKPT_ROOT." >&2
        echo "       Available there: $(ls -d "$CKPT_ROOT"/sd* 2>/dev/null | head -3 | xargs -n1 basename 2>/dev/null | tr '\n' ' ')..." >&2
        return 1
    fi
    echo "$d"
}

# -----------------------------
# Launch up to JOBS_PER_GPU runs concurrently on this task's single GPU
# -----------------------------
IDX=${SLURM_ARRAY_TASK_ID}
declare -a PIDS=() TAGS=()

for (( j=0; j<JOBS_PER_GPU; j++ )); do
    CFG=$(( IDX * JOBS_PER_GPU + j ))
    (( CFG < NUM_CONFIGS )) || break

    ANNEAL=""
    if (( CFG < NUM_CONST )); then
        CELL_IDX=$(( CFG / PER_CELL ))
        MODE_IDX=$(( (CFG / PER_MODE) % ${#MODES[@]} ))
        ALPHA_IDX=$(( (CFG / PER_ALPHA) % ${#ALPHAS[@]} ))
        SEED=$(( CFG % NUM_SEEDS ))
        MODE=${MODES[$MODE_IDX]}
        ALPHA=${ALPHAS[$ALPHA_IDX]}
        RUN_TAG="rlpd_noent_${MODE_TAGS[$MODE_IDX]}${ALPHA}"
    elif (( CFG < NUM_SWEEP )); then
        A=$(( CFG - NUM_CONST ))
        CELL_IDX=$(( A / ANN_PER_CELL ))
        MODE_IDX=$(( (A / ANN_PER_MODE) % ${#MODES[@]} ))
        ALPHA_IDX=$(( (A / NUM_SEEDS) % ${#ANNEAL_ALPHAS[@]} ))
        SEED=$(( A % NUM_SEEDS ))
        MODE=${MODES[$MODE_IDX]}
        ALPHA=${ANNEAL_ALPHAS[$ALPHA_IDX]}
        ANNEAL=$ANNEAL_FRAC
        RUN_TAG="rlpd_noent_ann${ANNEAL}_${MODE_TAGS[$MODE_IDX]}${ALPHA}"   # tag order as scripts/run_online_crl.sh
    else
        B=$(( CFG - NUM_SWEEP ))
        CELL_IDX=$(( B / NUM_SEEDS ))
        SEED=$(( B % NUM_SEEDS ))
        MODE=none
        ALPHA=0
        RUN_TAG="rlpd_baseline"
    fi

    CELL=${CELL_NAMES[$CELL_IDX]}
    CELL_RUN=${CELL_RUNS[$CELL_IDX]}
    OFFLINE_DATASET=${CELL_DATASETS[$CELL_IDX]}
    ENV_NAME=${CELL_ENVS[$CELL_IDX]}
    EPISODE_LENGTH=${CELL_EPISODE_LENGTHS[$CELL_IDX]}

    # RLPD dataset: honour an explicit OGBENCH_DATASET_DIR, else whichever standard location holds it.
    DS_DIR=${OGBENCH_DATASET_DIR:-}
    if [[ -z "$DS_DIR" ]]; then
        for cand in /global/scratch/users/ishirgarg/.ogbench/data "$HOME/.ogbench/data"; do
            if [[ -f "$cand/$OFFLINE_DATASET.npz" ]]; then DS_DIR="$cand"; break; fi
        done
    fi
    if [[ -z "$DS_DIR" || ! -f "$DS_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "FATAL: RLPD needs $OFFLINE_DATASET.npz but it is not in OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-<unset>}," >&2
        echo "       /global/scratch/users/ishirgarg/.ogbench/data or $HOME/.ogbench/data." >&2
        exit 1
    fi

    BONUS_FLAGS=()
    if [[ "$MODE" != "none" ]]; then
        EMP_CKPT=$(resolve_ckpt "$CELL_RUN") || exit 1
        compgen -G "$EMP_CKPT/params_*.pkl" > /dev/null || { echo "FATAL: no params_*.pkl in $EMP_CKPT" >&2; exit 1; }
        [[ -w "$EMP_CKPT" ]] || { echo "FATAL: $EMP_CKPT is not writable (the offline E cache is written under it)." >&2; exit 1; }
        read -r AGENT_NAME EMP_DATASET < <(python - "$EMP_CKPT" <<'PY'
import json, sys
f = json.load(open(sys.argv[1] + '/flags.json'))
print(f['agent']['agent_name'], f['env_name'])
PY
)
        [[ "$AGENT_NAME" == "empowerment_skill" ]] || { echo "FATAL: $EMP_CKPT is a $AGENT_NAME run, expected empowerment_skill" >&2; exit 1; }
        [[ "$EMP_DATASET" == "$OFFLINE_DATASET" ]] || {
            echo "FATAL: $EMP_CKPT was trained on $EMP_DATASET, but cell '$CELL' expects $OFFLINE_DATASET." >&2
            echo "       The exp_name in CELL_RUNS is wrong or names a different run on this cluster." >&2; exit 1; }
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
    mkdir -p "$SAVE_DIR"
    RUN_LOG="$SAVE_DIR/launch_seed${SEED}_${SLURM_JOB_ID:-local}.log"

    EP_FLAG=()
    if [[ -n "$EPISODE_LENGTH" ]]; then EP_FLAG=(--episode_length="$EPISODE_LENGTH"); fi

    TAG="cfg${CFG}[$CELL mode=$MODE alpha=$ALPHA anneal=${ANNEAL:-none} seed=$SEED]"
    echo "LAUNCH $TAG"
    echo "   env=$ENV_NAME ep_len=${EPISODE_LENGTH:-<registered>} rlpd=$OFFLINE_DATASET from $DS_DIR"
    echo "   estimator=${EMP_CKPT:-<none, baseline>}  log=$RUN_LOG"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then   # DRY_RUN=1 SLURM_ARRAY_TASK_ID=<n> bash <this file>: preflight + print, launch nothing
        echo "   DRY_RUN: main_online.py --env_name=$ENV_NAME --seed=$SEED --save_dir=$SAVE_DIR ${BONUS_FLAGS[*]:-} --offline_dataset=$OFFLINE_DATASET ${EP_FLAG[*]:-}"
        continue
    fi

    # Packed runs start in the same second: main_online.py names a run by seed + SLURM_JOB_ID +
    # timestamp, and packed runs of one task could share all three at other JOBS_PER_GPU values, so
    # give each its own id or they would share a run dir.
    SLURM_JOB_ID="${SLURM_JOB_ID:-local}.cfg${CFG}" OGBENCH_DATASET_DIR="$DS_DIR" python -u main_online.py \
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
