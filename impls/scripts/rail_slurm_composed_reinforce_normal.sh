#!/bin/bash
#SBATCH --job-name=ogb_composed_rf
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --array=0-16

# Composed (non-frozen low level) online policy with the REINFORCE high-level gradient --
# agents/online_composed_skill_policy.py with --agent.composed_grad_method=reinforce.
# BRC/Savio (rail) sibling of the rnn sweep scripts/slurm/submit_composed_online_lowlr_sweep.sh,
# which runs the same cells at 5 seeds with the default 'enumerate' gradient.
#
# Sweep: 3 skill checkpoints x 3 low_lr x 3 target_entropy_frac x 5 seeds = 135 runs,
#        packed EIGHT PER GPU -> 17 array tasks (0..16). Tasks 0..15 run 8 each, task 16 runs 7.
#
#   CFG = SLURM_ARRAY_TASK_ID * 8 + j     for j in 0..7, skipping CFG >= 135
#     CELL_IDX = CFG / 45   LR_IDX = (CFG / 15) % 3   ENT_IDX = (CFG / 5) % 3   SEED = CFG % 5
#     CELL 0 -> antsoccer-arena-navigate  -> antsoccer-arena-center-online-v0  (ep len 500)
#     CELL 1 -> pointmaze-teleport-stitch -> pointmaze-teleport-sparse-online-v0
#     CELL 2 -> cube-single-play          -> cube-single-center-online-v0
#     LR  0/1/2 -> low_lr 3e-4 / 1e-4 / 3e-5
#     ENT 0/1/2 -> target_entropy_frac 0.1 / 0.5 / 0.25
#
# Fixed: RLPD on with each checkpoint's OWN offline dataset, 1M env steps.
#
# On the entropy axis. H_target = frac * log(50), and it constrains the CATEGORICAL pi_hi only
# -- with low_temperature=0 the low level acts at its mode, so this is the agent's entire
# exploration budget. In effective live skills, exp(H_target):
#     frac 0.5  -> 1.956 nats -> 7.07 skills   (the rnn sweep's setting; matched controller runs exist)
#     frac 0.25 -> 0.978 nats -> 2.66 skills   (matched controller runs exist at this frac too)
#     frac 0.1  -> 0.391 nats -> 1.48 skills
# 0.1 and 0.25 concentrate the low level's gradient on few skills per state, since
# d/d(theta_lo) is weighted by pi_hi(k|s,g) -- that is the point of sweeping them here, but it
# is also the regime where a selector can absorb into a single skill and stop exploring.
#
# 'reinforce' samples ONE skill per row: the low level takes its ordinary pathwise gradient
# through the emitted action, the high level a score-function gradient with the low level's
# loss as a negative reward (agent docstring, Eqs. 6-7). Unbiased for the same objective as
# 'enumerate' -- verified numerically -- but with variance, at 1 low-level forward pass per
# row instead of K=50.
#
# GPU PACKING (8 runs share this task's single GPU). Measured on the rnn side by running four
# concurrent reinforce processes at these exact shapes (K=50, B=1024): 826 MiB GPU resident each
# -- reinforce's own JAX peak is only 0.36 GiB, the rest is per-process CUDA context + cuBLAS
# workspace, which does NOT amortise across processes -- and ~3.0-3.4 GiB host RSS each. So:
#     GPU  8 x 826 MiB  = ~6.6 GiB of a 24 GB A5000   (comfortable)
#     host 8 x ~3.2 GiB = ~26 GiB                     (check the node's RAM-per-GPU allowance)
# NOTE the 'enumerate' gradient would NOT fit the same way: its JAX peak is 3.18 GiB per run
# (K=50 low-level + critic passes per row) versus reinforce's 0.36 GiB.
# Two things packing REQUIRES, both handled below:
#   * XLA_PYTHON_CLIENT_PREALLOCATE=false -- JAX otherwise grabs 75% of the GPU on first use
#     and the other seven processes OOM immediately.
#   * one log file per run -- eight processes sharing the SBATCH --output would interleave.
# CPU is the binding constraint, not memory, and this deliberately oversubscribes it: 8
# MuJoCo+JAX processes on 4 cores is ~0.5 core each. That is a chosen trade-off -- 4 cores is the
# CPU:GPU ratio every other rail script in this repo uses, so it is known to schedule, and env
# stepping being CPU-bound mainly slows each individual run rather than losing aggregate
# throughput. Expect per-run wall clock well above the ~5.5 h an unpacked run takes; raise
# --cpus-per-task (or lower JOBS_PER_GPU) if that matters more than tasks-in-flight.
#
# BEFORE SUBMITTING, confirm on BRC (written on the rnn side, cannot see /global/*; each item
# is preflighted per-run below and fails with a specific message):
#   1. the checkout at $IMPLS_DIR contains agents/online_composed_skill_policy.py (2026-09-15),
#   2. the three empowerment_skill checkpoints exist under $CKPT_ROOT,
#   3. the three OGBench datasets are present for RLPD (compute nodes may have no internet),
#   4. `python` on the compute node is the env with jax/flax/ogbench installed.
#
# Submit from the impls/ directory:  cd <repo>/impls && sbatch scripts/<this file>
# Overrides (env vars): CKPT_ROOT, SAVE_ROOT, SCRATCH_ROOT, IMPLS_DIR, OGBENCH_DATASET_DIR,
#                       TOTAL_ENV_STEPS, JOBS_PER_GPU.

set -uo pipefail

export MUJOCO_GL=egl
# Mandatory for GPU packing -- see above.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# The checkout is wherever you ran `sbatch` from -- the same convention as the other rail
# sbatch scripts (run_empowerment_skill_resume_*.sh, run_dds_resume_*.sh), which just call
# `python main.py` and let Slurm put the job in the submit directory. No hardcoded prefix.
IMPLS_DIR=${IMPLS_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
# Everything BRC-side lives under scratch (home quota is small); this is the same root the
# other rail scripts use for WANDB_DIR and their save dirs.
SCRATCH_ROOT=${SCRATCH_ROOT:-/global/scratch/users/ishirgarg/ogbench}
SAVE_ROOT=${SAVE_ROOT:-$SCRATCH_ROOT/composed_reinforce}
# Local wandb run data goes to scratch too.
export WANDB_DIR=${WANDB_DIR:-$SCRATCH_ROOT}
mkdir -p "$WANDB_DIR"
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
JOBS_PER_GPU=${JOBS_PER_GPU:-8}

# -----------------------------
# Sweep definitions
# -----------------------------
# cube-single-play is given as an explicit leaf: that env dir holds FIVE sd000_* runs and this
# is the one every other sweep in this project uses. The other two are env dirs holding exactly
# one sd000_* run, resolved below.
CELL_CKPTS=(
    "antsoccer-arena-navigate"
    "pointmaze-teleport-stitch"
    "cube-single-play/sd000_s_38624008.0.20260908_013305"
)
CELL_ENVS=(
    antsoccer-arena-center-online-v0
    pointmaze-teleport-sparse-online-v0
    cube-single-center-online-v0
)
# antsoccer is registered at 1000 but every online run in this project uses 500 for it.
CELL_EPISODE_LENGTHS=(500 "" "")
LOW_LRS=(3e-4 1e-4 3e-5)
ENTROPY_FRACS=(0.1 0.5 0.25)
NUM_SEEDS=5
NUM_CONFIGS=$(( ${#CELL_CKPTS[@]} * ${#LOW_LRS[@]} * ${#ENTROPY_FRACS[@]} * NUM_SEEDS ))   # 135

GRAD_METHOD=reinforce

cd "$IMPLS_DIR" || { echo "FATAL: no ogbench checkout at IMPLS_DIR=$IMPLS_DIR" >&2; exit 1; }
[[ -f main_online.py ]] || {
    echo "FATAL: $IMPLS_DIR is not the impls/ directory (no main_online.py). Submit with" >&2
    echo "       'cd <repo>/impls && sbatch scripts/$(basename "$0")', or set IMPLS_DIR." >&2; exit 1; }
[[ -f agents/online_composed_skill_policy.py ]] || {
    echo "FATAL: $IMPLS_DIR has no agents/online_composed_skill_policy.py -- pull master, which has it." >&2; exit 1; }

# Pretrained 50-skill empowerment checkpoints: take CKPT_ROOT if set, else the first of the
# plausible BRC locations that actually exists, rather than guessing one.
if [[ -z "${CKPT_ROOT:-}" ]]; then
    for cand in "$SCRATCH_ROOT/ckpts/final/empowerment_final" "$IMPLS_DIR/ckpts/final/empowerment_final"; do
        if [[ -d "$cand" ]]; then CKPT_ROOT="$cand"; break; fi
    done
fi
if [[ -z "${CKPT_ROOT:-}" || ! -d "$CKPT_ROOT" ]]; then
    echo "FATAL: no empowerment_final checkpoint tree found. Tried" >&2
    echo "       $SCRATCH_ROOT/ckpts/final/empowerment_final and $IMPLS_DIR/ckpts/final/empowerment_final." >&2
    echo "       Set CKPT_ROOT=<dir containing antsoccer-arena-navigate/, pointmaze-teleport-stitch/, ...>." >&2
    exit 1
fi
echo "using IMPLS_DIR=$IMPLS_DIR  CKPT_ROOT=$CKPT_ROOT  SAVE_ROOT=$SAVE_ROOT" 

# Resolve a cell's checkpoint dir: (a) an exact leaf with flags.json, or (b) an env dir holding
# exactly one sd000_* run. Echoes the resolved path.
resolve_ckpt() {
    local d="$CKPT_ROOT/$1"
    if [[ -f "$d/flags.json" ]]; then echo "$d"; return 0; fi
    local m=("$d"/sd000_*/)
    if (( ${#m[@]} != 1 )) || [[ ! -d "${m[0]}" ]]; then
        echo "FATAL: no flags.json in $d and not exactly one sd000_* run under it." >&2
        echo "       Set CKPT_ROOT to where the empowerment_final checkpoints live on BRC." >&2
        return 1
    fi
    echo "${m[0]%/}"
}

# -----------------------------
# Launch up to JOBS_PER_GPU runs concurrently on this task's single GPU
# -----------------------------
IDX=${SLURM_ARRAY_TASK_ID}
declare -a PIDS=() TAGS=()

for (( j=0; j<JOBS_PER_GPU; j++ )); do
    CFG=$(( IDX * JOBS_PER_GPU + j ))
    (( CFG < NUM_CONFIGS )) || break

    CELL_IDX=$(( CFG / 45 ))
    LR_IDX=$(( (CFG / 15) % 3 ))
    ENT_IDX=$(( (CFG / 5) % 3 ))
    SEED=$(( CFG % 5 ))

    CELL=${CELL_CKPTS[$CELL_IDX]}
    ENV_NAME=${CELL_ENVS[$CELL_IDX]}
    EPISODE_LENGTH=${CELL_EPISODE_LENGTHS[$CELL_IDX]}
    LOW_LR=${LOW_LRS[$LR_IDX]}
    TARGET_ENTROPY_FRAC=${ENTROPY_FRACS[$ENT_IDX]}

    SKILL_CKPT=$(resolve_ckpt "$CELL") || exit 1
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "FATAL: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    read -r AGENT_NAME NUM_SKILLS SKILL_DATASET < <(python - "$SKILL_CKPT" <<'PY'
import json, sys
f = json.load(open(sys.argv[1] + '/flags.json'))
print(f['agent']['agent_name'], f['agent']['num_skills'], f['env_name'])
PY
)
    [[ "$AGENT_NAME" == "empowerment_skill" ]] || { echo "FATAL: $SKILL_CKPT is a $AGENT_NAME run, expected empowerment_skill" >&2; exit 1; }
    [[ "$NUM_SKILLS" == "50" ]] || { echo "FATAL: $SKILL_CKPT has num_skills=$NUM_SKILLS, expected 50" >&2; exit 1; }

    SKILL_EPOCH=$(python - "$SKILL_CKPT" <<'PY'
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
PY
)

    # RLPD dataset: honour an explicit OGBENCH_DATASET_DIR, else take whichever standard
    # location actually holds the .npz (compute nodes may have no internet to download it).
    DS_DIR=${OGBENCH_DATASET_DIR:-}
    if [[ -z "$DS_DIR" ]]; then
        for cand in /global/scratch/users/ishirgarg/.ogbench/data "$HOME/.ogbench/data"; do
            if [[ -f "$cand/$SKILL_DATASET.npz" ]]; then DS_DIR="$cand"; break; fi
        done
    fi
    if [[ -z "$DS_DIR" || ! -f "$DS_DIR/$SKILL_DATASET.npz" ]]; then
        echo "FATAL: RLPD needs $SKILL_DATASET.npz but it is not in OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-<unset>}," >&2
        echo "       /global/scratch/users/ishirgarg/.ogbench/data or $HOME/.ogbench/data." >&2
        exit 1
    fi

    SAVE_DIR="$SAVE_ROOT/$(basename "$CELL")/rlpd_lowlr${LOW_LR}_ent${TARGET_ENTROPY_FRAC}_${GRAD_METHOD}"
    mkdir -p "$SAVE_DIR"
    RUN_LOG="$SAVE_DIR/launch_seed${SEED}_${SLURM_JOB_ID:-local}.log"

    EP_FLAG=()
    if [[ -n "$EPISODE_LENGTH" ]]; then EP_FLAG=(--episode_length="$EPISODE_LENGTH"); fi

    TAG="cfg${CFG}[$(basename "$CELL") lr=$LOW_LR ent=$TARGET_ENTROPY_FRAC seed=$SEED]"
    echo "LAUNCH $TAG"
    echo "   ckpt=$SKILL_CKPT epoch=$SKILL_EPOCH  rlpd=$SKILL_DATASET from $DS_DIR"
    echo "   env=$ENV_NAME ep_len=${EPISODE_LENGTH:-<registered>}  log=$RUN_LOG"

    OGBENCH_DATASET_DIR="$DS_DIR" python -u main_online.py \
        --env_name="$ENV_NAME" \
        --seed="$SEED" \
        --save_dir="$SAVE_DIR" \
        --agent=agents/online_composed_skill_policy.py \
        --agent.skill_checkpoint_path="$SKILL_CKPT" \
        --agent.skill_restore_epoch="$SKILL_EPOCH" \
        --agent.composed_grad_method="$GRAD_METHOD" \
        --agent.low_lr="$LOW_LR" \
        --agent.target_entropy_frac="$TARGET_ENTROPY_FRAC" \
        --offline_dataset="$SKILL_DATASET" \
        --total_env_steps="$TOTAL_ENV_STEPS" \
        "${EP_FLAG[@]}" \
        --log_interval=5000 \
        --eval_interval=50000 \
        --save_interval=100000 \
        --eval_episodes=50 \
        --video_episodes=0 \
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
