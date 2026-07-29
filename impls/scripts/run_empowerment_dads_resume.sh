#!/bin/bash
#SBATCH --job-name=emp_dads_resume
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

# Resume the 2 interrupted empowerment_dads navigate runs from their latest
# checkpoint, in place. Submit from impls/:  sbatch scripts/run_empowerment_dads_resume.sh
#
# Each run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so the dashboard curves extend rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original sweep (env, seed, train_steps, agent hyperparameters, ...) is
# restated here — only the run folders below. Training restarts at the
# checkpointed step with the exact params, Adam state and TrainState.step.
# empowerment_dads' label assignment is computed from the network itself (no
# state outside the checkpoint), so it resumes with the labels it had.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.

IDX=${SLURM_ARRAY_TASK_ID}

BASE=/global/scratch/users/ishirgarg/ogbench/OGBench

# Run folder names (globally unique: sd<seed>_s_<jobid>.<step>.<timestamp>).
# The enclosing run_group directory is globbed, so it does not matter which
# sweep revision produced these.
RUNS=(
    "sd000_s_35789541.0.20260724_040949"  # antmaze-medium-navigate-v0
    "sd000_s_35819248.0.20260724_035517"  # antsoccer-arena-navigate-v0
)

if [ -z "$IDX" ] || [ "$IDX" -ge ${#RUNS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RUNS[@]} runs; use --array=0-$((${#RUNS[@]} - 1))." >&2
    exit 1
fi
RUN=${RUNS[$IDX]}

# Resolve the run folder under whatever run_group it lives in. Fail loudly on 0
# or >1 match rather than silently resuming the wrong run.
shopt -s nullglob
MATCHES=("$BASE"/*/"$RUN")
shopt -u nullglob
if [ ${#MATCHES[@]} -ne 1 ]; then
    echo "ERROR: expected 1 match for $BASE/*/$RUN, got ${#MATCHES[@]}: ${MATCHES[*]}" >&2
    exit 1
fi
RESUME_DIR=${MATCHES[0]}

if ! ls "$RESUME_DIR"/params_*.pkl >/dev/null 2>&1; then
    echo "ERROR: no params_*.pkl in $RESUME_DIR — this run died before its first save and cannot be resumed." >&2
    exit 1
fi

# Take the agent module from the run itself, so this never drifts from the run.
AGENT=$(python -c "import json,sys;print(json.load(open(sys.argv[1]+'/flags.json'))['agent']['agent_name'])" "$RESUME_DIR")

echo "IDX=$IDX  AGENT=$AGENT  RESUME_DIR=$RESUME_DIR"
echo "checkpoints present: $(ls "$RESUME_DIR"/params_*.pkl 2>/dev/null | wc -l)"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --agent=agents/"$AGENT".py \
    --resume_dir="$RESUME_DIR"
