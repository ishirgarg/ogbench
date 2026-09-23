#!/usr/bin/env bash
# SUPE on cube-single-multigoal and antmaze-medium-corner-all-squares, end to end, on the
# rnn.ist.berkeley.edu Slurm cluster. Submits (never runs) two stages, chained with Slurm
# dependencies so ONE invocation queues everything:
#
#   stage 1  OPAL skill pretraining, once per dataset, both jobs at the same time
#              cube-single-play-v0        (kl 0.2)   -> the skills for cube_mg
#              antmaze-medium-navigate-v0 (kl 0.1)   -> the skills for amz_all
#            scripts/slurm/run_supe_opal_pretrain.sbatch, continuous OPAL, chunk 4, 1M steps,
#            saved under $SUPE_OPAL_ROOT/<dataset>/ (MAIN checkout ckpts/final/supe_opal by default).
#            A dataset that ALREADY has a finished run there is skipped (its run is reused).
#   stage 2  30 online runs, each --dependency=afterok:<its dataset's OPAL job> (or no dependency
#            when the OPAL run already exists):
#              plain SUPE                          2 envs x 5 seeds          = 10   tag supe_rlpd
#              SUPE + empowerment distill           distill-to-rlpd, NO annealing,
#                 cube_mg  bonus_scale in {10, 3}   2 x 5 seeds              = 10   tag supe_rlpd_edrlpd{10,3}
#                 amz_all  bonus_scale in {30, 10}  2 x 5 seeds              = 10   tag supe_rlpd_edrlpd{30,10}
#            scripts/slurm/run_supe_online_seed.sbatch with the launcher defaults: 1M env steps,
#            100 eval episodes every 25k, RLPD with the cell's dataset, the online_crl gradient
#            budget (one 1024-row step per env step), results in $SAVE_ROOT/<cell>/<tag>/.
#
# Slurm note: a stage-2 job whose OPAL job FAILS stays pending as DependencyNeverSatisfied; cancel
# it (scancel) and resubmit with STAGE=online once the OPAL run is fixed.
#
# Usage, from the login node, inside the checkout that holds this branch (any path; the scripts
# find it from their own location):
#   cd <checkout>/impls
#   bash scripts/slurm/submit_supe_pipeline_cube_mg_amz_all.sh              # both stages, chained
#   STAGE=opal   bash scripts/slurm/submit_supe_pipeline_cube_mg_amz_all.sh # only the 2 pretrains
#   STAGE=online bash scripts/slurm/submit_supe_pipeline_cube_mg_amz_all.sh # only the 30 online runs
#                                                                            # (OPAL runs must exist)
# Overrides: DRY_RUN=1 (print every sbatch command, submit nothing), SEEDS ("0 1 2 3 4"),
#   CELLS ("cube_mg amz_all"), SUPE_OPAL_ROOT, EMP_ROOT, SAVE_ROOT, CHUNK (4), plus any env var
#   run_supe_online_seed.sbatch reads (K must equal CHUNK).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/
source scripts/slurm/supe_cells.sh

# The checkout this script lives in: exported so every sbatch job runs THIS code (the sbatch scripts
# fall back to the original worktree path only when REPO_DIR is unset).
export REPO_DIR=${REPO_DIR:-$(cd .. && pwd)}
STAGE=${STAGE:-all}
DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
CHUNK=${CHUNK:-4}
export K=${K:-$CHUNK}
MAIN_IMPLS=/nas/ucb/ishirgarg/ogbench/impls
export SUPE_OPAL_ROOT=${SUPE_OPAL_ROOT:-$MAIN_IMPLS/ckpts/final/supe_opal}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
OPAL_SBATCH=scripts/slurm/run_supe_opal_pretrain.sbatch
ONLINE_SBATCH=scripts/slurm/run_supe_online_seed.sbatch
LOG_OPAL=logs/slurm/supe_opal
LOG_ONLINE=logs/slurm/supe_online
mkdir -p "$LOG_OPAL" "$LOG_ONLINE"

# cell -> bonus scales of the distill-to-rlpd arm
declare -A CELL_ALPHAS=([cube_mg]="10 3" [amz_all]="30 10")
read -r -a CELL_LIST <<< "${CELLS:-cube_mg amz_all}"
[[ "$STAGE" =~ ^(all|opal|online)$ ]] || { echo "ERROR: STAGE=$STAGE (all | opal | online)" >&2; exit 1; }
[[ "$K" == "$CHUNK" ]] || { echo "ERROR: K=$K must equal CHUNK=$CHUNK (the OPAL chunk_size)" >&2; exit 1; }

submit() {  # prints the sbatch command; submits unless DRY_RUN; echoes the job id (or a fake one)
    echo "  sbatch $*" >&2
    if [[ "$DRY_RUN" == "1" ]]; then echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi
}

# ── stage 1: OPAL pretraining, one job per distinct dataset ──
declare -A OPAL_JOB=()   # dataset -> job id ("" when an existing run is reused)
for CELL in "${CELL_LIST[@]}"; do
    supe_cell "$CELL" || exit 1
    [[ -v OPAL_JOB[$CELL_DATASET] ]] && continue
    [[ -f "$DATASET_DIR/$CELL_DATASET.npz" ]] || { echo "ERROR: $DATASET_DIR/$CELL_DATASET.npz is missing" >&2; exit 1; }
    if EXISTING=$(supe_opal_run "$SUPE_OPAL_ROOT" "$CELL_DATASET" 2>/dev/null); then
        echo "[pipeline] $CELL_DATASET: reusing the finished OPAL run $EXISTING"
        OPAL_JOB[$CELL_DATASET]=""
        continue
    fi
    if [[ "$STAGE" == "online" ]]; then
        echo "ERROR: STAGE=online but no finished OPAL run for $CELL_DATASET under $SUPE_OPAL_ROOT (run STAGE=opal first)" >&2
        exit 1
    fi
    KL=0.1
    [[ "$CELL_DATASET" == cube-* || "$CELL_DATASET" == scene-* ]] && KL=0.2
    JOB_NAME="supe_opal_${CELL_DATASET%-v0}_c${CHUNK}"
    echo "[pipeline] $CELL_DATASET: submitting OPAL pretraining (chunk $CHUNK, kl $KL)"
    OPAL_JOB[$CELL_DATASET]=$(submit --job-name="$JOB_NAME" --output="$LOG_OPAL/${JOB_NAME}_%j.log" \
        "$OPAL_SBATCH" "$CELL_DATASET" 0 "$CHUNK" "$KL")
done
[[ "$STAGE" == "opal" ]] && { echo "[pipeline] stage 1 submitted: ${!OPAL_JOB[*]} -> ${OPAL_JOB[*]}"; exit 0; }

# ── stage 2: online runs ──
n=0
for CELL in "${CELL_LIST[@]}"; do
    supe_cell "$CELL" || exit 1
    [[ -n "$CELL_EMP_RUN" ]] || { echo "ERROR: cell $CELL has no empowerment estimator run" >&2; exit 1; }
    DEP=()
    if [[ -n "${OPAL_JOB[$CELL_DATASET]}" ]]; then DEP=(--dependency=afterok:"${OPAL_JOB[$CELL_DATASET]}"); fi
    # arm = "mode|alpha"; plain SUPE first, then the distill-to-rlpd weights (no annealing).
    ARMS=("none|1.0")
    for A in ${CELL_ALPHAS[$CELL]:?no bonus scales defined for cell $CELL}; do ARMS+=("distill-to-rlpd|$A"); done
    for ARM in "${ARMS[@]}"; do
        IFS='|' read -r MODE ALPHA <<< "$ARM"
        for SEED in $SEEDS; do
            JOB_NAME="supe_${CELL}_${MODE}${ALPHA}_s${SEED}"
            submit "${DEP[@]}" --job-name="$JOB_NAME" --output="$LOG_ONLINE/${JOB_NAME}_%j.log" \
                "$ONLINE_SBATCH" "$CELL" "$SEED" "$MODE" "$ALPHA" "" > /dev/null
            n=$((n + 1))
        done
    done
done
echo "[pipeline] $( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n online jobs" \
     "(OPAL jobs: $(for d in "${!OPAL_JOB[@]}"; do printf '%s=%s ' "$d" "${OPAL_JOB[$d]:-existing}"; done))"
