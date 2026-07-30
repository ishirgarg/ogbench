#!/bin/bash
#SBATCH --job-name=emp_skill_slice50_resume
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00

# Resume the interrupted empowerment_skill run on antsoccer-arena-stitch-slice50-v0
# from its latest checkpoint, in place. Submit from impls/:
#   sbatch scripts/run_empowerment_skill_slice50_resume.sh
#
# The run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so the dashboard curves extend rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original sweep (env, seed, num_skills, bc_alpha, train_steps, ...) is
# restated here — only the run folder below. Training restarts at the
# checkpointed step with the exact params, Adam state and TrainState.step.
# The -slice50- dataset is rebuilt/cached in ~/.ogbench/data exactly as in the
# original run, since the env_name comes back from flags.json unchanged.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.

BASE=/global/scratch/users/ishirgarg/ogbench/OGBench

# Run folder name (globally unique: sd<seed>_s_<jobid>.<step>.<timestamp>).
# The enclosing run_group directory is globbed, so it does not matter which
# sweep revision produced it.
RUN="sd000_s_35873141.0.20260726_160900"  # antsoccer-arena-stitch-slice50-v0

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

echo "AGENT=$AGENT  RESUME_DIR=$RESUME_DIR"
echo "checkpoints present: $(ls "$RESUME_DIR"/params_*.pkl 2>/dev/null | wc -l)"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --agent=agents/"$AGENT".py \
    --resume_dir="$RESUME_DIR"
