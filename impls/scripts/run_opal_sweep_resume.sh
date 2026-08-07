#!/bin/bash
#SBATCH --job-name=opal_sweep_resume
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-7

# Continuation of scripts/run_opal_sweep.sh's 4 envs x 2 configs (continuous /
# discrete) sweep, same IDX -> (ENV, CONFIG) mapping. All eight runs now have
# surviving checkpoints and are resumed in place.
#
# Submit from impls/:  sbatch scripts/run_opal_sweep_resume.sh
#
#   IDX 0 : antmaze-medium-navigate-v0   continuous  -- RESUME sd000_s_36462376.0.20260806_143857
#   IDX 1 : antmaze-medium-navigate-v0   discrete    -- RESUME sd000_s_36392681.0.20260805_024210
#   IDX 2 : antsoccer-arena-navigate-v0  continuous  -- RESUME sd000_s_36462406.0.20260806_143915
#   IDX 3 : antsoccer-arena-navigate-v0  discrete    -- RESUME sd000_s_36392683.0.20260805_024216
#   IDX 4 : antmaze-medium-stitch-v0     continuous  -- RESUME sd000_s_36462416.0.20260806_143923
#   IDX 5 : antmaze-medium-stitch-v0     discrete    -- RESUME sd000_s_36462417.0.20260806_143926
#   IDX 6 : antsoccer-arena-stitch-v0    continuous  -- RESUME sd000_s_36462419.0.20260806_143930
#   IDX 7 : antsoccer-arena-stitch-v0    discrete    -- RESUME sd000_s_36462375.0.20260806_143932
#
# For resumed runs, main.py --resume_dir replays the run's own flags.json, so
# the AGENT_FLAGS below are not restated — only the run folder matters.
# Training restarts at the checkpointed step with the exact params, Adam
# state and TrainState.step (relevant for the discrete path's
# `cluster_steps`-gated offline-DADS clustering phase). Each resumed run
# continues in ITS OWN existing folder: the same params_*.pkl series, the
# same train.csv / eval.csv (appended, not truncated), and the same wandb run
# id, so the dashboard curve extends rather than a new run appearing.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Fresh runs are unaffected.

IDX=${SLURM_ARRAY_TASK_ID}

ENVS=(
    # navigate first ...
    antmaze-medium-navigate-v0
    antsoccer-arena-navigate-v0
    # ... then stitch
    antmaze-medium-stitch-v0
    antsoccer-arena-stitch-v0
)
CONFIGS=(continuous discrete)
SEED=0

# Run folder to resume, keyed by IDX. Empty string means "start fresh".
RESUME_RUNS=(
    "sd000_s_36462376.0.20260806_143857"   # 0 -- antmaze-medium-navigate-v0   continuous
    "sd000_s_36392681.0.20260805_024210"   # 1 -- antmaze-medium-navigate-v0   discrete
    "sd000_s_36462406.0.20260806_143915"   # 2 -- antsoccer-arena-navigate-v0  continuous
    "sd000_s_36392683.0.20260805_024216"   # 3 -- antsoccer-arena-navigate-v0  discrete
    "sd000_s_36462416.0.20260806_143923"   # 4 -- antmaze-medium-stitch-v0     continuous
    "sd000_s_36462417.0.20260806_143926"   # 5 -- antmaze-medium-stitch-v0     discrete
    "sd000_s_36462419.0.20260806_143930"   # 6 -- antsoccer-arena-stitch-v0    continuous
    "sd000_s_36462375.0.20260806_143932"   # 7 -- antsoccer-arena-stitch-v0    discrete
)

if [ -z "$IDX" ] || [ "$IDX" -ge ${#RESUME_RUNS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RESUME_RUNS[@]} runs; use --array=0-$((${#RESUME_RUNS[@]} - 1))." >&2
    exit 1
fi

ENV=${ENVS[$((IDX / 2))]}
CONFIG=${CONFIGS[$((IDX % 2))]}
RUN=${RESUME_RUNS[$IDX]}

SAVE_DIR=/global/scratch/users/ishirgarg/ogbench
BASE=$SAVE_DIR

export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

set -e

if [ -n "$RUN" ]; then
    # ── Resume ──────────────────────────────────────────────────────────────
    AGENT_NAME=opal

    # Resolve the run folder wherever it lives under $BASE. Fail loudly on 0
    # or >1 match rather than silently resuming the wrong run.
    shopt -s nullglob globstar
    MATCHES=("$BASE"/**/"$RUN")
    shopt -u nullglob globstar
    if [ ${#MATCHES[@]} -ne 1 ]; then
        echo "ERROR: expected 1 match for $BASE/**/$RUN, got ${#MATCHES[@]}: ${MATCHES[*]}" >&2
        exit 1
    fi
    RESUME_DIR=${MATCHES[0]}

    if ! ls "$RESUME_DIR"/params_*.pkl >/dev/null 2>&1; then
        echo "ERROR: no params_*.pkl in $RESUME_DIR — this run died before its first save and cannot be resumed." >&2
        exit 1
    fi

    FOUND_AGENT=$(python -c "import json,sys;print(json.load(open(sys.argv[1]+'/flags.json'))['agent']['agent_name'])" "$RESUME_DIR")
    if [ "$FOUND_AGENT" != "$AGENT_NAME" ]; then
        echo "ERROR: $RESUME_DIR was trained with agent '$FOUND_AGENT', expected '$AGENT_NAME'." >&2
        exit 1
    fi

    FOUND_ENV=$(python -c "import json,sys;print(json.load(open(sys.argv[1]+'/flags.json'))['env_name'])" "$RESUME_DIR")
    if [ "$FOUND_ENV" != "$ENV" ]; then
        echo "ERROR: $RESUME_DIR was trained on env '$FOUND_ENV', expected '$ENV'." >&2
        exit 1
    fi

    echo "IDX=$IDX  ENV=$ENV  CONFIG=$CONFIG  SEED=$SEED  RESUME_DIR=$RESUME_DIR"
    echo "checkpoints present: $(ls "$RESUME_DIR"/params_*.pkl 2>/dev/null | wc -l)"

    python main.py \
        --agent=agents/"$AGENT_NAME".py \
        --resume_dir="$RESUME_DIR"
else
    # ── Fresh launch (matches run_opal_sweep.sh) ──────────────────────────────
    if [ "$CONFIG" = "continuous" ]; then
        AGENT_FLAGS=(
            --agent.latent_type=continuous
            --agent.skill_dim=8
            --agent.kl_coef=0.1
            --agent.chunk_size=10
            --agent.sequence_length=10
        )
    else
        AGENT_FLAGS=(
            --agent.latent_type=discrete
            --agent.num_skills=15
            --agent.chunk_size=10
            --agent.sequence_length=10
            --agent.cluster_steps=500000
        )
    fi

    echo "IDX=$IDX  ENV=$ENV  CONFIG=$CONFIG  SEED=$SEED  (fresh)"

    python main.py \
        --env_name=$ENV \
        --agent=agents/opal.py \
        "${AGENT_FLAGS[@]}" \
        --seed=$SEED \
        --train_steps=1000000 \
        --video_episodes=0 \
        --save_dir=$SAVE_DIR
fi
