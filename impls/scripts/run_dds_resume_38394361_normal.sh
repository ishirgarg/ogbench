#!/bin/bash
#SBATCH --job-name=dds_resume_38394361_normal
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00

# Resume the failed DDS run sd000_s_38394361.0.20260901_181500 from its latest
# checkpoint, in place, on the normal-priority queue (rail_gpu4_normal) instead
# of whatever queue it originally died on.
#
# Submit from impls/:  sbatch scripts/run_dds_resume_38394361_normal.sh
#
# The run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so the dashboard curve extends rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original launch (env, seed, num_skills, save_interval, ...) is restated here
# -- only the run folder below. Training restarts at the checkpointed step
# with the exact params, Adam state and TrainState.step, so DDS's step-gated
# skill-pretrain/high-level phases (agents/dds.py) resume in the correct phase
# rather than restarting.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.

set -euo pipefail

BASE=/global/scratch/users/ishirgarg/ogbench
AGENT_NAME=dds
RUN="sd000_s_38394361.0.20260901_181500"

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
# not distinguish this from any other dds sweep cell, so NUM_SKILLS is echoed
# too: check it in the job log if you need to confirm which sweep cell this
# run was.
if ! INFO=$(python -c "import json,sys;f=json.load(open(sys.argv[1]+'/flags.json'));print(f['agent']['agent_name'],f['env_name'],f['agent'].get('num_skills'))" "$RESUME_DIR"); then
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

echo "AGENT=$AGENT_NAME  ENV=$ENV  NUM_SKILLS=$NUM_SKILLS  RESUME_DIR=$RESUME_DIR"
echo "checkpoints present: $(ls "$RESUME_DIR"/params_*.pkl 2>/dev/null | wc -l)"

if [ -f "$RESUME_DIR/wandb_run_id.txt" ]; then
    echo "wandb_run_id.txt found ($(cat "$RESUME_DIR/wandb_run_id.txt")) -- will re-attach to the existing wandb run."
else
    echo "WARNING: no wandb_run_id.txt in $RESUME_DIR -- main.py will start a NEW wandb run (training still resumes in place from the latest checkpoint)."
fi

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --agent=agents/"$AGENT_NAME".py \
    --resume_dir="$RESUME_DIR"
