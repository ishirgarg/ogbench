#!/usr/bin/env bash
# Re-launch the ONE offline empowerment_skill high-level controller run that failed in the
# 2026-09-06 alpha sweep: antsoccer-arena-stitch @ alpha=0.3, seed 0.
#
# Why this exists as its own script: job 1194926 (the alpha0.3/asoc_sti cell of
# submit_empowerment_controller_alpha_sweep.sh) was CANCELLED DUE TO TIME LIMIT after
# 16:00:20, having reached only ~530k of 1M steps -- it landed on ppo.ist.berkeley.edu and
# ran ~4x slower than its three siblings, which finished 1M steps in ~4h on other nodes.
# Its output dir therefore stops at params_500000.pkl with success 0.0 through step 500k.
# Nothing about the run was wrong, so this re-submits the identical configuration with a
# longer wall clock instead of re-running the whole 4x4 grid.
#
# The training logic is NOT duplicated here: this wraps
#   scripts/slurm/run_empowerment_controller_alpha_seed.sbatch
# (the same script the sweep submits), pinning its three positional args and overriding
# only --time. sbatch command-line flags take precedence over the #SBATCH directives in
# that file, so --time=48:00:00 replaces its default of 16:00:00. QoS `default` caps out
# at 3-00:00:00, so 48h is within limits and ~1.6x the timed-out job's observed pace.
#
# The previous run's directory is left untouched: get_exp_name folds SLURM_JOB_ID into the
# run folder name, so this lands in a NEW sd000_s_<jobid>.<ts>/ beside it under
#   <SKILL_CKPT>/controller_awr_sweep/alpha0.3/OGBench/Debug/
# Delete the stale sd000_s_1194926.* folder by hand if you don't want it in the sweep plots.
#
# Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/rerun_empowerment_controller_asoc_stitch_alpha03.sh
# Overrides:
#   DRY_RUN=1        print the sbatch command without submitting
#   TIME=72:00:00    a different wall clock (<= 3-00:00:00 on qos=default)
#   ALPHA=1 SEED=1   re-target the same checkpoint at a different cell
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_empowerment_controller_alpha_seed.sbatch
LOG_DIR=logs/slurm/empowerment_controller_alpha
EMP_ROOT=${EMP_ROOT:-ckpts/final/empowerment_final}
CKPT_DIR=antsoccer-arena-stitch
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}

DRY_RUN=${DRY_RUN:-0}
ALPHA=${ALPHA:-0.3}
SEED=${SEED:-0}
TIME=${TIME:-48:00:00}

mkdir -p "$LOG_DIR"
[[ -f "$SBATCH_SCRIPT" ]] || { echo "ERROR: missing $SBATCH_SCRIPT (run me from the repo, not a copy)" >&2; exit 1; }

# Resolve by glob so a re-rsync under a different job id still works; this env dir holds
# exactly one sd000_* run (unlike antmaze-medium-*, which hold a k=50 and a k=15 run).
matches=("$EMP_ROOT/$CKPT_DIR"/sd000_*/)
if (( ${#matches[@]} != 1 )) || [[ ! -d "${matches[0]}" ]]; then
    echo "ERROR: expected exactly one sd000_* run under $EMP_ROOT/$CKPT_DIR, found ${#matches[@]}" >&2
    exit 1
fi
SKILL_CKPT=${matches[0]%/}
[[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

# Offline training reads the checkpoint's own dataset; compute nodes have no internet
# egress, so a miss here is a job that dies minutes in rather than downloading.
ENV_NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
if [[ ! -f "$DATASET_DIR/$ENV_NAME.npz" ]]; then
    echo "WARNING: $DATASET_DIR/$ENV_NAME.npz is missing; this job will fail on a compute node." >&2
fi

JOB_NAME="empctrl_a${ALPHA}_asoc_sti_s${SEED}"
OUT="$LOG_DIR/${JOB_NAME}_%j.log"
cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$TIME"
     "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ALPHA" "$SEED")

echo "ckpt=$SKILL_CKPT env=$ENV_NAME alpha=$ALPHA seed=$SEED time=$TIME"
echo "${cmd[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "(dry run; nothing submitted)"
else
    "${cmd[@]}"
fi
