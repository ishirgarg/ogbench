#!/bin/bash
#SBATCH --job-name=opal_discrete_resume_20260911
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-7

# Resume eight OPAL discrete runs (agents/opal.py latent_type=discrete) from
# their latest checkpoints, in place, on the high-priority RAIL queue:
#
#   array 0  sd000_s_38579157.0.20260904_235645
#   array 1  sd000_s_38579176.0.20260904_235644
#   array 2  sd000_s_38394976.0.20260901_185415
#   array 3  sd000_s_38394982.0.20260901_185411
#   array 4  sd000_s_38394981.0.20260901_185409
#   array 5  sd000_s_38394979.0.20260901_185403
#   array 6  sd000_s_38394978.0.20260901_185401
#   array 7  sd000_s_38394977.0.20260901_185358
#
# Submit from impls/:  sbatch scripts/run_opal_discrete_resume_20260911_high.sh
# One run only:        sbatch --array=3 scripts/run_opal_discrete_resume_20260911_high.sh
# A subset:            sbatch --array=0,1,5 scripts/run_opal_discrete_resume_20260911_high.sh
#
# Each run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so each dashboard curve extends rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original launch (env, seed, num_skills, chunk_size, cluster_steps,
# save_interval, ...) is restated here -- only the run folder. Training restarts
# at the checkpointed step with the exact params, Adam state and
# TrainState.step, so OPAL's step-gated clustering -> BC phases resume in the
# correct phase rather than restarting the EM clustering.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.

set -euo pipefail

BASE=/global/scratch/users/ishirgarg/ogbench
AGENT_NAME=opal
LATENT_TYPE=discrete

RUNS=(
    "sd000_s_38579157.0.20260904_235645"
    "sd000_s_38579176.0.20260904_235644"
    "sd000_s_38394976.0.20260901_185415"
    "sd000_s_38394982.0.20260901_185411"
    "sd000_s_38394981.0.20260901_185409"
    "sd000_s_38394979.0.20260901_185403"
    "sd000_s_38394978.0.20260901_185401"
    "sd000_s_38394977.0.20260901_185358"
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
# not distinguish this from any other opal sweep cell, so env_name / num_skills
# / latent_type are echoed too: check them in the job log to confirm which cell
# this run was.
if ! INFO=$(python -c "import json,sys;f=json.load(open(sys.argv[1]+'/flags.json'));a=f['agent'];print(a['agent_name'],f['env_name'],a.get('num_skills'),a.get('latent_type'))" "$RESUME_DIR"); then
    echo "ERROR: could not read $RESUME_DIR/flags.json -- the run folder is incomplete or corrupt." >&2
    exit 1
fi
read -r FOUND_AGENT ENV NUM_SKILLS FOUND_LATENT <<<"$INFO"

# Guard against a mistyped folder name silently resuming a different agent's
# (or the continuous-VAE opal) run. main.py re-checks the agent too, but
# failing here keeps the error legible.
if [ "$FOUND_AGENT" != "$AGENT_NAME" ]; then
    echo "ERROR: $RESUME_DIR was trained with agent '$FOUND_AGENT', expected '$AGENT_NAME'." >&2
    exit 1
fi
if [ "$FOUND_LATENT" != "$LATENT_TYPE" ]; then
    echo "ERROR: $RESUME_DIR has latent_type='$FOUND_LATENT', expected '$LATENT_TYPE'." >&2
    exit 1
fi

echo "TASK=$IDX  AGENT=$AGENT_NAME  LATENT_TYPE=$FOUND_LATENT  ENV=$ENV  NUM_SKILLS=$NUM_SKILLS  RESUME_DIR=$RESUME_DIR"
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
