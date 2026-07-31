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
# scripts/run_empowerment_comparison_navigate.sh (IDX 2 = antsoccer-arena-navigate-v0,
# IDX 5 = antmaze-medium-navigate-v0) from their latest checkpoint, in place.
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
# restated here. Training restarts at the checkpointed step with the exact
# params, Adam state and TrainState.step.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.
#
# Unlike scripts/run_empowerment_crl_resume.sh, the run folders are not
# hardcoded here: they are discovered by reading agent_name out of each run's
# flags.json, so a relaunched sweep with different job ids still resolves. Set
# EMP_DV_RUNS to a space-separated list of absolute run dirs to override
# discovery (e.g. to resume a specific subset).

IDX=${SLURM_ARRAY_TASK_ID}

BASE=/global/scratch/users/ishirgarg/ogbench/OGBench
AGENT_NAME=empowerment_dv

# ── Discover the empowerment_dv runs ──────────────────────────────────────────
# Layout: $BASE/<run_group>/<sd000_s_<jobid>.<step>.<timestamp>>/flags.json
# Only runs that actually have a checkpoint are resumable, so runs that died
# before their first save are skipped rather than crashing an array task.
if [ -n "$EMP_DV_RUNS" ]; then
    read -r -a RUNS <<< "$EMP_DV_RUNS"
else
    RUNS=()
    shopt -s nullglob
    for FLAGS in "$BASE"/*/*/flags.json; do
        DIR=$(dirname "$FLAGS")
        ls "$DIR"/params_*.pkl >/dev/null 2>&1 || continue
        NAME=$(python -c "
import json, sys
try:
    print(json.load(open(sys.argv[1]))['agent']['agent_name'])
except Exception:
    pass
" "$FLAGS")
        [ "$NAME" = "$AGENT_NAME" ] && RUNS+=("$DIR")
    done
    shopt -u nullglob
    # Deterministic order across array tasks.
    IFS=$'\n' RUNS=($(printf '%s\n' "${RUNS[@]}" | sort)); unset IFS
fi

echo "found ${#RUNS[@]} resumable $AGENT_NAME run(s):"
printf '  %s\n' "${RUNS[@]}"

if [ ${#RUNS[@]} -eq 0 ]; then
    echo "ERROR: no resumable $AGENT_NAME runs with a params_*.pkl under $BASE/*/*." >&2
    exit 1
fi
if [ -z "$IDX" ] || [ "$IDX" -ge ${#RUNS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range for ${#RUNS[@]} runs; use --array=0-$((${#RUNS[@]} - 1))." >&2
    exit 1
fi
RESUME_DIR=${RUNS[$IDX]}

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
