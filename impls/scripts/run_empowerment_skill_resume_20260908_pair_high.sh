#!/bin/bash
#SBATCH --job-name=empowerment_skill_resume_20260908_pair
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

# Resume two empowerment_skill runs (agents/empowerment_skill.py) from their
# latest checkpoints, in place, on the high-priority RAIL queue:
#
#   array 0  sd000_s_38624730.0.20260908_014657
#   array 1  sd000_s_38624729.0.20260908_014657
#
# Submit from impls/:  sbatch scripts/run_empowerment_skill_resume_20260908_pair_high.sh
# One run only:        sbatch --array=0 scripts/run_empowerment_skill_resume_20260908_pair_high.sh
#
# Each run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so each dashboard curve extends rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original launch (env, seed, num_skills, bc_alpha, noise, save_interval, ...)
# is restated here -- only the run folder. Training restarts at the
# checkpointed step with the exact params, Adam state and TrainState.step.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.

set -euo pipefail

BASE=/global/scratch/users/ishirgarg/ogbench
AGENT_NAME=empowerment_skill

RUNS=(
    "sd000_s_38624730.0.20260908_014657"
    "sd000_s_38624729.0.20260908_014657"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge ${#RUNS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range for ${#RUNS[@]} runs; use --array=0-$((${#RUNS[@]} - 1))." >&2
    exit 1
fi
RUN=${RUNS[$IDX]}

# Resolve the run folder under its <wandb project>/<run_group> parents. main.py
# saves to <save_dir>/<project>/<run_group>/<exp_name> (main.py:190), so two
# glob levels are exact -- and far cheaper than a recursive ** walk, since $BASE
# is also WANDB_DIR and holds the whole wandb tree. Fail loudly on 0 or >1 match
# rather than silently resuming the wrong run.
shopt -s nullglob
MATCHES=("$BASE"/*/*/"$RUN")
shopt -u nullglob
if [ ${#MATCHES[@]} -ne 1 ]; then
    echo "ERROR: expected 1 match for $BASE/*/*/$RUN, got ${#MATCHES[@]}: ${MATCHES[*]-}" >&2
    exit 1
fi
RESUME_DIR=${MATCHES[0]}

if ! ls "$RESUME_DIR"/params_*.pkl >/dev/null 2>&1; then
    echo "ERROR: no params_*.pkl in $RESUME_DIR -- this run died before its first save and cannot be resumed." >&2
    exit 1
fi

# Read the run's identity out of flags.json in one pass. agent_name alone does
# not distinguish this from any other empowerment_skill sweep cell, so
# env_name / num_skills are echoed too: check them in the job log to confirm
# which cell this run was.
if ! INFO=$(python -c "import json,sys;f=json.load(open(sys.argv[1]+'/flags.json'));a=f['agent'];print(a['agent_name'],f['env_name'],a.get('num_skills'))" "$RESUME_DIR"); then
    echo "ERROR: could not read $RESUME_DIR/flags.json -- the run folder is incomplete or corrupt." >&2
    exit 1
fi
read -r FOUND_AGENT ENV NUM_SKILLS <<<"$INFO"

# Guard against a mistyped folder name silently resuming a different agent's
# run. main.py re-checks this too, but failing here keeps the error legible.
if [ "$FOUND_AGENT" != "$AGENT_NAME" ]; then
    echo "ERROR: $RESUME_DIR was trained with agent '$FOUND_AGENT', expected '$AGENT_NAME'." >&2
    exit 1
fi

echo "TASK=$IDX  AGENT=$AGENT_NAME  ENV=$ENV  NUM_SKILLS=$NUM_SKILLS  RESUME_DIR=$RESUME_DIR"
echo "checkpoints present: $(ls "$RESUME_DIR"/params_*.pkl 2>/dev/null | wc -l)"
echo "latest checkpoint:   $(ls "$RESUME_DIR"/params_*.pkl | sed -E 's/.*params_([0-9]+)\.pkl/\1/' | sort -n | tail -n 1)"

if [ -f "$RESUME_DIR/wandb_run_id.txt" ]; then
    echo "wandb_run_id.txt found ($(cat "$RESUME_DIR/wandb_run_id.txt")) -- will re-attach to the existing wandb run."
else
    echo "WARNING: no wandb_run_id.txt in $RESUME_DIR -- main.py will start a NEW wandb run (training still resumes in place from the latest checkpoint)."
fi

# -----------------------------
# Run
# -----------------------------
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python -u main.py \
    --agent=agents/"$AGENT_NAME".py \
    --resume_dir="$RESUME_DIR"
