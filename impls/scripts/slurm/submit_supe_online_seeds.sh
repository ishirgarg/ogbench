#!/usr/bin/env bash
# Submit SUPE online runs (scripts/slurm/run_supe_online_seed.sbatch) over cells x seeds x arms to the
# rnn.ist.berkeley.edu Slurm cluster. Default: the plain SUPE baseline (arm "none") on every cell that
# has both a finished continuous OPAL run (supe_cells.sh `supe_opal_run`) and its dataset on the NAS,
# 5 seeds -> one sbatch job per (cell, seed, arm[, alpha]).
#
# Arms (MODES): none | distill | distill-to-rlpd  -- the latter two are SUPE + our distilled
# empowerment bonus at each BONUS_SCALE in ALPHAS (and, with ANNEAL_FRAC set, annealed).
#
# This ONLY submits jobs. Run from the rnn login node, from THIS worktree's impls/:
#   cd /nas/ucb/ishirgarg/ogbench/.claude/worktrees/supe-baseline-integration-4d17b9/impls
#   bash scripts/slurm/submit_supe_online_seeds.sh
# Overrides:
#   DRY_RUN=1                       print the sbatch commands (and each job's main_online flags), submit nothing
#   CELLS="amz_corner pmt_corner"   a subset of the cells in supe_cells.sh (default: all with an OPAL run)
#   SEEDS="0 1 2 3 4"
#   MODES="none distill-to-rlpd"    ALPHAS="10 30"   ANNEAL_FRAC=0.5 (annealed arms only, in addition to constant)
#   plus every env var run_supe_online_seed.sbatch reads (SUPE_OPAL_ROOT, EMP_ROOT, SAVE_ROOT, K, UTD_RATIO, ...).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/
source scripts/slurm/supe_cells.sh
# The checkout this script lives in: every sbatch job runs THIS code (the sbatch scripts only fall
# back to the original worktree path when REPO_DIR is unset).
export REPO_DIR=${REPO_DIR:-$(cd .. && pwd)}

SBATCH_SCRIPT=scripts/slurm/run_supe_online_seed.sbatch
LOG_DIR=logs/slurm/supe_online
mkdir -p "$LOG_DIR"
DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
MODES=${MODES:-none}
ALPHAS=${ALPHAS:-1.0}
ANNEAL_FRAC=${ANNEAL_FRAC:-}
MAIN_IMPLS=/nas/ucb/ishirgarg/ogbench/impls
SUPE_OPAL_ROOT=${SUPE_OPAL_ROOT:-$MAIN_IMPLS/ckpts/final/supe_opal}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
export SUPE_OPAL_ROOT

if [[ -n "${CELLS:-}" ]]; then
    read -r -a CELL_LIST <<< "$CELLS"
else
    mapfile -t CELL_LIST < <(printf '%s\n' "${!SUPE_CELLS[@]}" | sort)
fi

n=0
for CELL in "${CELL_LIST[@]}"; do
    supe_cell "$CELL" || exit 1
    if ! OPAL_RUN=$(supe_opal_run "$SUPE_OPAL_ROOT" "$CELL_DATASET" 2>/dev/null); then
        if [[ -n "${CELLS:-}" ]]; then
            echo "ERROR: cell $CELL: no finished continuous OPAL run for $CELL_DATASET under $SUPE_OPAL_ROOT" >&2; exit 1
        fi
        echo "skipping $CELL: no finished OPAL run for $CELL_DATASET under $SUPE_OPAL_ROOT" >&2
        continue
    fi
    [[ -f "$DATASET_DIR/$CELL_DATASET.npz" ]] || { echo "ERROR: $DATASET_DIR/$CELL_DATASET.npz is missing" >&2; exit 1; }
    for MODE in $MODES; do
        if [[ "$MODE" == "none" ]]; then
            ARMS=("none|1.0|")
        else
            [[ -n "$CELL_EMP_RUN" ]] || { echo "ERROR: cell $CELL has no estimator run; MODE=$MODE impossible" >&2; exit 1; }
            ARMS=()
            for A in $ALPHAS; do
                ARMS+=("$MODE|$A|")
                [[ -n "$ANNEAL_FRAC" ]] && ARMS+=("$MODE|$A|$ANNEAL_FRAC")
            done
        fi
        for ARM in "${ARMS[@]}"; do
            IFS='|' read -r ARM_MODE ARM_ALPHA ARM_ANNEAL <<< "$ARM"
            for SEED in $SEEDS; do
                JOB_NAME="supe_${CELL}_${ARM_MODE}${ARM_ALPHA}${ARM_ANNEAL:+_ann$ARM_ANNEAL}_s${SEED}"
                cmd=(sbatch --job-name="$JOB_NAME" --output="$LOG_DIR/${JOB_NAME}_%j.log"
                     "$SBATCH_SCRIPT" "$CELL" "$SEED" "$ARM_MODE" "$ARM_ALPHA" "$ARM_ANNEAL")
                echo "${cmd[@]}"
                if [[ "$DRY_RUN" == "1" ]]; then
                    DRY_RUN=1 bash "$SBATCH_SCRIPT" "$CELL" "$SEED" "$ARM_MODE" "$ARM_ALPHA" "$ARM_ANNEAL" | sed 's/^/    /'
                else
                    "${cmd[@]}"
                fi
                n=$((n + 1))
            done
        done
    done
done
echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n jobs"
