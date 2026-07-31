#!/bin/bash
#SBATCH --job-name=emp_dv_resume
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

# Resume the two empowerment_dv navigate runs launched by
# scripts/run_empowerment_comparison_navigate.sh (sweep IDX 2 = antsoccer,
# IDX 5 = antmaze) from their latest checkpoint, in place.
#
# Submit from impls/:  sbatch scripts/run_empowerment_dv_resume.sh
#
# Each run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so the dashboard curves extend rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original sweep (env, seed, train_steps, agent hyperparameters, ...) is
# restated here — only the run folders below. Training restarts at the
# checkpointed step with the exact params, Adam state and TrainState.step, so
# the step-gated critic phase (`pretrain_steps`, agents/empowerment_dv.py —
# logged as phase/in_critic) resumes in the correct phase rather than
# restarting its bc/dyn-only pretrain.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.

IDX=${SLURM_ARRAY_TASK_ID}

BASE=/global/scratch/users/ishirgarg/ogbench
AGENT_NAME=empowerment_dv

# Run folder names (globally unique: sd<seed>_s_<jobid>.<step>.<timestamp>).
# The enclosing <wandb project>/<run_group> path is globbed, so it does not
# matter which sweep revision or project produced these.
RUNS=(
    "sd000_s_35814832.0.20260723_201013"  # antsoccer-arena-navigate-v0
    "sd000_s_35818791.0.20260724_020744"  # antmaze-medium-navigate-v0
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
