#!/bin/bash
#SBATCH --job-name=resume_20260913_batch
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-7

# Resume 8 runs from 2026-09-13 from their latest checkpoints, in place, on
# the high-priority RAIL queue:
#
#   array 0  sd000_s_38754294.0.20260913_031839
#   array 1  sd000_s_38754297.0.20260913_031836
#   array 2  sd000_s_38754296.0.20260913_031831
#   array 3  sd000_s_38754782.0.20260913_040310
#   array 4  sd000_s_38754781.0.20260913_040310
#   array 5  sd000_s_38754783.0.20260913_040310
#   array 6  sd000_s_38754759.0.20260913_040310
#   array 7  sd000_s_38754757.0.20260913_040103
#
# Submit from a Savio login node, from impls/:
#   sbatch scripts/run_resume_20260913_batch_high.sh
# One run only, e.g. just the 040310 group:
#   sbatch --array=3-6 scripts/run_resume_20260913_batch_high.sh
# (BRC's /global/* paths are not visible from rnn, so this cannot be
# submitted there.)
#
# Each run continues in ITS OWN existing folder: the same params_*.pkl
# series, the same train.csv / eval.csv (appended, not truncated), and the
# same wandb run id (from wandb_run_id.txt), so each dashboard curve extends
# rather than a new run appearing.
#
# main.py --resume_dir replays each run's own flags.json, so nothing about
# the original launch (agent, env, seed, ...) is restated here -- only the
# run folder. The agent module is read out of flags.json below and passed
# explicitly, since main.py requires --agent to already match the saved
# agent_name. Runs in this batch are not assumed to share an agent, so it is
# read per-run rather than hardcoded. Training restarts at the checkpointed
# step with the exact params, Adam state and TrainState.step, so step-gated
# phases (e.g. DDS/empowerment_crl pretrain) resume in the correct phase
# rather than restarting.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after
# a resume differs from an uninterrupted run. Everything the optimizer sees
# is exact.

set -euo pipefail

BASE=/global/scratch/users/ishirgarg/ogbench

RUNS=(
    "sd000_s_38754294.0.20260913_031839"
    "sd000_s_38754297.0.20260913_031836"
    "sd000_s_38754296.0.20260913_031831"
    "sd000_s_38754782.0.20260913_040310"
    "sd000_s_38754781.0.20260913_040310"
    "sd000_s_38754783.0.20260913_040310"
    "sd000_s_38754759.0.20260913_040310"
    "sd000_s_38754757.0.20260913_040103"
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

# Read the run's identity out of flags.json in one pass. The agent name is not
# hardcoded here since this batch spans more than one agent -- it's read back
# and passed to --agent below.
if ! INFO=$(python -c "import json,sys;f=json.load(open(sys.argv[1]+'/flags.json'));a=f['agent'];print(a['agent_name'],f['env_name'],a.get('num_skills'))" "$RESUME_DIR"); then
    echo "ERROR: could not read $RESUME_DIR/flags.json -- the run folder is incomplete or corrupt." >&2
    exit 1
fi
read -r AGENT_NAME ENV NUM_SKILLS <<<"$INFO"

echo "TASK=$IDX  RUN=$RUN  AGENT=$AGENT_NAME  ENV=$ENV  NUM_SKILLS=$NUM_SKILLS  RESUME_DIR=$RESUME_DIR"
echo "checkpoints present: $(ls "$RESUME_DIR"/params_*.pkl 2>/dev/null | wc -l)"
echo "latest checkpoint:   $(ls "$RESUME_DIR"/params_*.pkl | sed -E 's/.*params_([0-9]+)\.pkl/\1/' | sort -n | tail -n 1)"

if [ -f "$RESUME_DIR/wandb_run_id.txt" ]; then
    echo "wandb_run_id.txt found ($(cat "$RESUME_DIR/wandb_run_id.txt")) -- will re-attach to the existing wandb run."
else
    echo "WARNING: no wandb_run_id.txt in $RESUME_DIR -- main.py will start a NEW wandb run (training still resumes in place from the latest checkpoint)."
fi

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python -u main.py \
    --agent=agents/"$AGENT_NAME".py \
    --resume_dir="$RESUME_DIR"
