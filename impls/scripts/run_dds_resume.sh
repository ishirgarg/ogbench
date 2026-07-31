#!/bin/bash
#SBATCH --job-name=dds_resume
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-3

# Resume 4 of the interrupted DDS paper-sweep runs (scripts/run_dds_paper_sweep.sh)
# from their latest checkpoint, in place. Submit from impls/:
#   sbatch scripts/run_dds_resume.sh
#
# Each run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so the dashboard curves extend rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original sweep (env, K, seed, train_steps, ...) is restated here — only the
# run folders below. Training restarts at the checkpointed step with the exact
# params, Adam state and TrainState.step, so DDS' phase gate
# (`skill_pretrain_steps`) resumes in the correct phase rather than restarting
# the skill VQ-VAE pretrain.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.

IDX=${SLURM_ARRAY_TASK_ID}

BASE=/global/scratch/users/ishirgarg/ogbench/OGBench

# Run folder names (globally unique: sd<seed>_s_<jobid>.<step>.<timestamp>).
# The enclosing run_group directory is globbed, so it does not matter which
# sweep revision produced these (these predate dropping --run_group, so they
# live under dds_<env>_K<k>/).
RUNS=(
    "sd000_s_35757533.0.20260721_190340"  # antmaze-medium-navigate-v0
    "sd000_s_35757534.0.20260721_190340"  # antmaze-medium-navigate-v0
    "sd000_s_35757532.0.20260721_190340"  # antsoccer-arena-stitch-v0
    "sd000_s_35757539.0.20260721_190340"  # antsoccer-arena-stitch-v0
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
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

python main.py \
    --agent=agents/"$AGENT".py \
    --resume_dir="$RESUME_DIR"
