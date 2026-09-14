#!/usr/bin/env bash
# Submit the OFFLINE empowerment_skill high-level (gciql / IQL) controller AWR-alpha
# sweep for ONE specific checkpoint: the antsoccer-arena-navigate *noisy_policy* run
#   ckpts/empowerment/antsoccer-arena-navigate/noisy_policy/sd000_s_36524700.0.20260807_020714_k50_0.1_0.01
# (K=50 skills, action_noise_std=0.1, bc_alpha=0.01, 1M steps).
#
# Same stage and same alpha grid {0.3, 1, 3, 10} as
# submit_empowerment_controller_alpha_sweep.sh -- this is a sibling of that script for a
# checkpoint that lives OUTSIDE ckpts/final/empowerment_final and so does not fit that
# script's one-run-per-env-dir glob. It reuses the identical sbatch body
# (run_empowerment_controller_alpha_seed.sbatch), so results are directly comparable.
#
# 4 alphas x 1 checkpoint x 1 seed = 4 jobs. Results land in
#   <SKILL_CKPT>/controller_awr_sweep/alpha<ALPHA>/OGBench/Debug/sd000_s_<jobid>.<ts>/
# and nothing under <SKILL_CKPT> itself is modified.
#
# Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_empowerment_controller_alpha_sweep_noisy_asoc_nav.sh
# Overrides:
#   DRY_RUN=1          print the sbatch commands without submitting
#   ALPHAS="0.3 1"     a different alpha grid
#   SEEDS="0 1"        more than the single seed 0
#   SKILL_CKPT=...     a different checkpoint directory
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_empowerment_controller_alpha_seed.sbatch
LOG_DIR=logs/slurm/empowerment_controller_alpha
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
ALPHAS=${ALPHAS:-"0.3 1 3 10"}
SEEDS=${SEEDS:-"0"}
TAG=${TAG:-asoc_nav_noisy}

SKILL_CKPT=${SKILL_CKPT:-$PWD/ckpts/empowerment/antsoccer-arena-navigate/noisy_policy/sd000_s_36524700.0.20260807_020714_k50_0.1_0.01}
SKILL_CKPT=${SKILL_CKPT%/}

[[ -d "$SKILL_CKPT" ]] || { echo "ERROR: no such checkpoint dir: $SKILL_CKPT" >&2; exit 1; }
[[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

# Offline training reads the checkpoint's own dataset; warn early if it isn't on the NAS
# (compute nodes have no internet egress, so a cache miss is a hard failure there).
ENV_NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
if [[ ! -f "$DATASET_DIR/$ENV_NAME.npz" ]]; then
    echo "WARNING: $DATASET_DIR/$ENV_NAME.npz is missing; these jobs will fail on a compute node." >&2
fi

n_submitted=0
for ALPHA in $ALPHAS; do
    for SEED in $SEEDS; do
        JOB_NAME="empctrl_a${ALPHA}_${TAG}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
             "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ALPHA" "$SEED")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs for $SKILL_CKPT (env=$ENV_NAME)"
