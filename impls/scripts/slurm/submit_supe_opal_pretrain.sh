#!/usr/bin/env bash
# Submit the ONE-OFF continuous OPAL pretraining SUPE needs, one job per dataset, to the
# rnn.ist.berkeley.edu Slurm cluster (scripts/slurm/run_supe_opal_pretrain.sbatch). The
# resulting run dirs ($CKPT_ROOT/<dataset>/OGBench/Debug/sd000_*) are what every SUPE online
# run of that dataset's cells loads frozen (scripts/slurm/supe_cells.sh `supe_opal_run`), so
# this is submitted once and never again for a given (dataset, chunk).
#
# Default datasets = the five our online cells draw RLPD data from (supe_cells.sh):
#   antmaze-medium-navigate-v0  pointmaze-teleport-navigate-v0  antsoccer-arena-navigate-v0
#   cube-single-play-v0 (kl 0.2)  cube-double-play-v0 (kl 0.2)
# kl_coef follows the paper's per-env commands: 0.2 on cube, 0.1 elsewhere.
#
# Run from the rnn login node:
#   bash scripts/slurm/submit_supe_opal_pretrain.sh
# Overrides: DRY_RUN=1, DATASETS="antmaze-medium-navigate-v0 ...", SEED (0), CHUNK (4: SUPE's
#            hpolicy_horizon; pass 10 for a run comparable to our k=10 controllers), CKPT_ROOT.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/
# The checkout this script lives in: every sbatch job runs THIS code (the sbatch scripts only fall
# back to the original worktree path when REPO_DIR is unset).
export REPO_DIR=${REPO_DIR:-$(cd .. && pwd)}

SBATCH_SCRIPT=scripts/slurm/run_supe_opal_pretrain.sbatch
LOG_DIR=logs/slurm/supe_opal
mkdir -p "$LOG_DIR"
DRY_RUN=${DRY_RUN:-0}
SEED=${SEED:-0}
CHUNK=${CHUNK:-4}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
DEFAULT_DATASETS="antmaze-medium-navigate-v0 pointmaze-teleport-navigate-v0 antsoccer-arena-navigate-v0 cube-single-play-v0 cube-double-play-v0"
read -r -a DATASETS <<< "${DATASETS:-$DEFAULT_DATASETS}"

n=0
for DATASET in "${DATASETS[@]}"; do
    [[ -f "$DATASET_DIR/$DATASET.npz" ]] || { echo "ERROR: $DATASET_DIR/$DATASET.npz is missing" >&2; exit 1; }
    KL=0.1
    [[ "$DATASET" == cube-* || "$DATASET" == scene-* ]] && KL=0.2
    JOB_NAME="supe_opal_${DATASET%-v0}_c${CHUNK}"
    cmd=(sbatch --job-name="$JOB_NAME" --output="$LOG_DIR/${JOB_NAME}_%j.log" "$SBATCH_SCRIPT" "$DATASET" "$SEED" "$CHUNK" "$KL")
    echo "${cmd[@]}"
    [[ "$DRY_RUN" == "1" ]] || "${cmd[@]}"
    n=$((n + 1))
done
echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n jobs"
