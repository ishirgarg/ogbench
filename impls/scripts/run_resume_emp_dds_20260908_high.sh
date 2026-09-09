#!/bin/bash
#SBATCH --job-name=resume_emp_dds_20260908
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

# Resume the first two of the four runs launched together on 2026-09-08 01:49
# from their latest checkpoints, in place, on the high-priority RAIL queue:
#
#   array 0  sd000_s_38624851  empowerment_skill
#   array 1  sd000_s_38624850  empowerment_skill
#
# The two dds runs from that launch (sd000_s_38624849, sd000_s_38624852) are
# deliberately NOT resumed here.
#
# Submit from impls/:  sbatch scripts/run_resume_emp_dds_20260908_high.sh
# One run only:        sbatch --array=1 scripts/run_resume_emp_dds_20260908_high.sh
#
# Each run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so each dashboard curve extends rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original launch (env, seed, num_skills, save_interval, ...) is restated here
# -- only the run folder. Training restarts at the checkpointed step with the
# exact params, Adam state and TrainState.step, so DDS's step-gated
# skill-pretrain/high-level phases (agents/dds.py) resume in the correct phase
# rather than restarting.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.

set -euo pipefail

BASE=/global/scratch/users/ishirgarg/ogbench

# Index-aligned: RUNS[i] is resumed with AGENTS[i]. The agent is restated per
# run (rather than read out of flags.json) so a mistyped folder name fails the
# check below instead of silently resuming the wrong thing.
RUNS=(
    "sd000_s_38624851.0.20260908_014900"
    "sd000_s_38624850.0.20260908_014900"
)
AGENTS=(
    empowerment_skill
    empowerment_skill
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
RUN=${RUNS[$IDX]}
AGENT_NAME=${AGENTS[$IDX]}

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
# not distinguish this from any other sweep cell, so env_name / num_skills are
# echoed too: check them in the job log to confirm which cell this run was.
if ! INFO=$(python -c "import json,sys;f=json.load(open(sys.argv[1]+'/flags.json'));print(f['agent']['agent_name'],f['env_name'],f['agent'].get('num_skills'))" "$RESUME_DIR"); then
    echo "ERROR: could not read $RESUME_DIR/flags.json -- the run folder is incomplete or corrupt." >&2
    exit 1
fi
read -r FOUND_AGENT ENV NUM_SKILLS <<<"$INFO"

# main.py re-checks this too, but failing here keeps the error legible.
if [ "$FOUND_AGENT" != "$AGENT_NAME" ]; then
    echo "ERROR: $RESUME_DIR was trained with agent '$FOUND_AGENT', expected '$AGENT_NAME'." >&2
    exit 1
fi

echo "TASK=$IDX  AGENT=$AGENT_NAME  ENV=$ENV  NUM_SKILLS=$NUM_SKILLS  RESUME_DIR=$RESUME_DIR"
echo "checkpoints present: $(ls "$RESUME_DIR"/params_*.pkl 2>/dev/null | wc -l)"

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
