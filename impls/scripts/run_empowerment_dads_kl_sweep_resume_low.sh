#!/bin/bash
#SBATCH --job-name=emp_dads_kl_sweep_resume_low
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_low
#SBATCH --requeue
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --array=0-7

# Resume all 8 empowerment_dads KL-sweep runs launched by
# scripts/run_empowerment_dads_kl_sweep.sh (2026-08-22, rail_gpu4_lowest) from
# their latest checkpoint, in place, on the low-priority queue.
#
# Submit from impls/:  sbatch scripts/run_empowerment_dads_kl_sweep_resume_low.sh
#
# Sibling of scripts/run_empowerment_dads_kl_sweep_resume_normal.sh; identical
# mechanics, only the queue/requeue setting differs.
#
# Each run continues in ITS OWN existing folder: the same params_*.pkl series,
# the same train.csv / eval.csv (appended, not truncated), and the same wandb
# run id (from wandb_run_id.txt), so the dashboard curves extend rather than a
# new run appearing.
#
# main.py --resume_dir replays the run's own flags.json, so nothing about the
# original sweep (env, seed, kl_coef, num_skills, train_steps, ...) is
# restated here — only the run folders below. Training restarts at the
# checkpointed step with the exact params, Adam state and TrainState.step.
#
# NOTE: only the data-sampling RNG is not checkpointed, so batch order after a
# resume differs from an uninterrupted run. Everything the optimizer sees is
# exact.

IDX=${SLURM_ARRAY_TASK_ID}

BASE=/global/scratch/users/ishirgarg/ogbench/OGBench
AGENT_NAME=empowerment_dads

# Run folder names (globally unique: sd<seed>_s_<jobid>.<procid>.<timestamp>).
# The enclosing run_group directory is globbed, so it does not matter which
# run_group these were logged under.
RUNS=(
    "sd000_s_37936188.0.20260822_053601"
    "sd000_s_37936192.0.20260822_053617"
    "sd000_s_37936189.0.20260822_053607"
    "sd000_s_37936190.0.20260822_053606"
    "sd000_s_37936191.0.20260822_053615"
    "sd000_s_37936155.0.20260822_053629"
    "sd000_s_37936194.0.20260822_053625"
    "sd000_s_37936193.0.20260822_053623"
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

# Guard against a mistyped folder name silently resuming a different agent's
# run. main.py re-checks this too, but failing here keeps the error legible.
FOUND_AGENT=$(python -c "import json,sys;print(json.load(open(sys.argv[1]+'/flags.json'))['agent']['agent_name'])" "$RESUME_DIR")
if [ "$FOUND_AGENT" != "$AGENT_NAME" ]; then
    echo "ERROR: $RESUME_DIR was trained with agent '$FOUND_AGENT', expected '$AGENT_NAME'." >&2
    exit 1
fi

ENV=$(python -c "import json,sys;print(json.load(open(sys.argv[1]+'/flags.json'))['env_name'])" "$RESUME_DIR")
KL=$(python -c "import json,sys;print(json.load(open(sys.argv[1]+'/flags.json'))['agent'].get('kl_coef'))" "$RESUME_DIR")

echo "IDX=$IDX  AGENT=$AGENT_NAME  ENV=$ENV  KL_COEF=$KL  RESUME_DIR=$RESUME_DIR"
echo "checkpoints present: $(ls "$RESUME_DIR"/params_*.pkl 2>/dev/null | wc -l)"

# ── Run ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export WANDB_DIR=/global/scratch/users/ishirgarg/ogbench
mkdir -p "$WANDB_DIR"

python main.py \
    --agent=agents/"$AGENT_NAME".py \
    --resume_dir="$RESUME_DIR"
