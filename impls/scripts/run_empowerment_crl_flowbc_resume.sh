#!/bin/bash
#SBATCH --job-name=emp_flowbc_resume
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-1

# Resume the two empowerment_crl_flowbc navigate runs launched by
# scripts/run_empowerment_comparison_navigate.sh (IDX 0 = antsoccer-arena-navigate-v0,
# IDX 3 = antmaze-medium-navigate-v0) from their latest checkpoint, in place.
#
# Submit from impls/:  sbatch scripts/run_empowerment_crl_flowbc_resume.sh
#
# Sibling of scripts/run_empowerment_dv_resume.sh; identical mechanics, only
# AGENT_NAME differs.
#
# Each run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so the dashboard curves extend rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original sweep (env, seed, train_steps, agent hyperparameters, ...) is
# restated here. Training restarts at the checkpointed step with the exact
# params, Adam state and TrainState.step, so the step-gated critic phase
# (`pretrain_steps`, agents/empowerment_crl_flowbc.py — logged as
# phase/in_critic) resumes in the correct phase rather than restarting its
# bc/dyn-only pretrain.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.
#
# Unlike scripts/run_empowerment_crl_resume.sh, the run folders are not
# hardcoded here: they are discovered by reading agent_name out of each run's
# flags.json, so a relaunched sweep with different job ids still resolves. Set
# EMP_FLOWBC_RUNS to a space-separated list of absolute run dirs to override
# discovery (e.g. to resume a specific subset).

IDX=${SLURM_ARRAY_TASK_ID}

BASE=/global/scratch/users/ishirgarg/ogbench
AGENT_NAME=empowerment_crl_flowbc

# ── Discover the empowerment_crl_flowbc runs ──────────────────────────────────
# Layout (main.py): $BASE/<wandb project>/<run_group>/<sd<seed>_s_<jobid>.<step>.<timestamp>>
# The search is recursive from $BASE rather than globbing a fixed
# <project>/<run_group> depth, so every run_group AND every wandb project is
# covered — same approach as main.py's --resume auto-resume scan.
# Only runs that actually have a checkpoint are resumable, so runs that died
# before their first save are skipped rather than crashing an array task.
# The agent_name test is exact equality, NOT a prefix match, so the plain
# `empowerment_crl` runs (sweep IDX 6-7) are not picked up here.
if [ -n "$EMP_FLOWBC_RUNS" ]; then
    read -r -a RUNS <<< "$EMP_FLOWBC_RUNS"
else
    RUNS=()
    shopt -s nullglob globstar
    for FLAGS in "$BASE"/**/flags.json; do
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
    shopt -u nullglob globstar
    # Deterministic order across array tasks.
    IFS=$'\n' RUNS=($(printf '%s\n' "${RUNS[@]}" | sort)); unset IFS
fi

echo "found ${#RUNS[@]} resumable $AGENT_NAME run(s):"
printf '  %s\n' "${RUNS[@]}"

if [ ${#RUNS[@]} -eq 0 ]; then
    echo "ERROR: no resumable $AGENT_NAME runs with a params_*.pkl under $BASE." >&2
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
