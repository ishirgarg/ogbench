#!/bin/bash
#SBATCH --job-name=opal_resume
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-7

# Resume the 8 interrupted opal runs from their latest checkpoint, in place.
# Submit from impls/:  sbatch scripts/run_opal_resume.sh
#
# Each run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so the dashboard curves extend rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original launch (env, seed, latent_type, num_skills/skill_dim, chunk_size,
# cluster_steps, ...) is restated here — only the run folders below. Training
# restarts at the checkpointed step with the exact params, Adam state and
# TrainState.step, so opal's step-gated VAE/cluster/policy phases
# (agents/opal.py) resume in the correct phase rather than restarting.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.

IDX=${SLURM_ARRAY_TASK_ID}

BASE=/global/scratch/users/ishirgarg/ogbench
AGENT_NAME=opal

# Run folder names (globally unique: sd<seed>_s_<jobid>.<step>.<timestamp>).
# The enclosing <wandb project>/<run_group> path is globbed, so it does not
# matter which sweep revision or project produced these.
RUNS=(
    "sd000_s_36524708.0.20260807_020721"  # antsoccer-arena-navigate-v0, discrete, num_skills=15
    "sd000_s_36524707.0.20260807_020721"  # antsoccer-arena-navigate-v0, continuous, skill_dim=10
    "sd000_s_36524711.0.20260807_020721"  # antsoccer-arena-stitch-v0, continuous, skill_dim=10
    "sd000_s_36524703.0.20260807_020721"  # antsoccer-arena-stitch-v0, discrete, num_skills=15
    "sd000_s_36524709.0.20260807_020721"  # antmaze-medium-stitch-v0, continuous, skill_dim=10
    "sd000_s_36524710.0.20260807_020721"  # antmaze-medium-stitch-v0, discrete, num_skills=15
    "sd000_s_36524706.0.20260807_020721"  # antmaze-medium-navigate-v0, discrete, num_skills=15
    "sd000_s_36524705.0.20260807_020720"  # antmaze-medium-navigate-v0, continuous, skill_dim=10
)

if [ -z "$IDX" ] || [ "$IDX" -ge ${#RUNS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RUNS[@]} runs; use --array=0-$((${#RUNS[@]} - 1))." >&2
    exit 1
fi
RUN=${RUNS[$IDX]}

# Resolve the run folder wherever it lives under $BASE. Fail loudly on 0 or >1
# match rather than silently resuming the wrong run.
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

# Guard against a mistyped folder name silently resuming a different agent's
# run. main.py re-checks this too, but failing here keeps the error legible.
FOUND_AGENT=$(python -c "import json,sys;print(json.load(open(sys.argv[1]+'/flags.json'))['agent']['agent_name'])" "$RESUME_DIR")
if [ "$FOUND_AGENT" != "$AGENT_NAME" ]; then
    echo "ERROR: $RESUME_DIR was trained with agent '$FOUND_AGENT', expected '$AGENT_NAME'." >&2
    exit 1
fi

ENV=$(python -c "import json,sys;print(json.load(open(sys.argv[1]+'/flags.json'))['env_name'])" "$RESUME_DIR")

echo "IDX=$IDX  AGENT=$AGENT_NAME  ENV=$ENV  RESUME_DIR=$RESUME_DIR"
echo "checkpoints present: $(ls "$RESUME_DIR"/params_*.pkl 2>/dev/null | wc -l)"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --agent=agents/"$AGENT_NAME".py \
    --resume_dir="$RESUME_DIR"
