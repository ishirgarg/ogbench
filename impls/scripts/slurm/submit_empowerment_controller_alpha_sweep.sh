#!/usr/bin/env bash
# Submit the OFFLINE empowerment_skill high-level controller AWR-alpha sweep to the
# rnn.ist.berkeley.edu Slurm cluster: 4 empowerment_final checkpoints x 4 alphas -> 16
# jobs, one sbatch job per (checkpoint, alpha).
#
# The alphas are the full {0.3, 1, 3, 10} grid from the Savio antsoccer-medium sweep
# (scripts/run_skill_bc_relabel_awr_sweep_antsoccer.sh). Unlike the DDS alpha sweep
# (submit_dds_controller_alpha_sweep.sh, which skips 3.0 because the local run already
# covers it), NONE of these four checkpoints has a controller trained yet, so all four
# alphas are swept here.
#
# The 4 checkpoints (ckpts/final/empowerment_final/<env>/sd000_*), all K=50 skills,
# 1M steps, from the 2026-09-01 Savio batch:
#   antsoccer-arena-navigate, antsoccer-arena-stitch,
#   pointmaze-teleport-navigate, pointmaze-teleport-stitch
#
# Each run trains only the gciql high level over the K=50 skill indices; the pretrained
# agent is loaded read-only and results land in
#   <SKILL_CKPT>/controller_awr_sweep/alpha<ALPHA>/OGBench/Debug/sd000_s_<jobid>.<ts>/
# so nothing under <SKILL_CKPT> is touched. See
# run_empowerment_controller_alpha_seed.sbatch for the exact flags.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn login
# node, from this NAS checkout:
#   bash scripts/slurm/submit_empowerment_controller_alpha_sweep.sh
# Overrides:
#   DRY_RUN=1                 print the sbatch commands without submitting
#   ALPHAS="0.3 1"            a different alpha grid
#   SEEDS="0 1"               more than the single seed 0
#   CKPT_DIRS="antsoccer-arena-stitch ..."   a subset of the four env dirs
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each checkpoint's OFFLINE dataset must already be in
#     /nas/ucb/ishirgarg/.ogbench/data (compute nodes have no internet egress).
#     All four were present as of 2026-09-06.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_empowerment_controller_alpha_seed.sbatch
LOG_DIR=logs/slurm/empowerment_controller_alpha
EMP_ROOT=${EMP_ROOT:-ckpts/final/empowerment_final}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
ALPHAS=${ALPHAS:-"0.3 1 3 10"}
SEEDS=${SEEDS:-"0"}

# Parallel arrays: checkpoint env dir -> short job-name tag.
ALL_CKPT_DIRS=(
    antsoccer-arena-navigate
    antsoccer-arena-stitch
    pointmaze-teleport-navigate
    pointmaze-teleport-stitch
)
ALL_CKPT_TAGS=(asoc_nav asoc_sti pmt_nav pmt_sti)

read -r -a WANTED <<< "${CKPT_DIRS:-${ALL_CKPT_DIRS[*]}}"
CKPT_DIRS=(); CKPT_TAGS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_CKPT_DIRS[@]}"; do
        if [[ "${ALL_CKPT_DIRS[$i]}" == "$w" ]]; then
            CKPT_DIRS+=("${ALL_CKPT_DIRS[$i]}")
            CKPT_TAGS+=("${ALL_CKPT_TAGS[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown checkpoint dir '$w' (known: ${ALL_CKPT_DIRS[*]})" >&2; exit 1; }
done

n_submitted=0
for c in "${!CKPT_DIRS[@]}"; do
    # Resolve by glob so a re-rsync with a different job id still works; each env dir must
    # hold exactly one sd000_* run. (antmaze-medium-{navigate,stitch} hold two -- a k=50 and
    # a k=15 run -- which is one reason they are not in this sweep.)
    matches=("$EMP_ROOT/${CKPT_DIRS[$c]}"/sd000_*/)
    if (( ${#matches[@]} != 1 )) || [[ ! -d "${matches[0]}" ]]; then
        echo "ERROR: expected exactly one sd000_* run under $EMP_ROOT/${CKPT_DIRS[$c]}, found ${#matches[@]}" >&2
        exit 1
    fi
    SKILL_CKPT=${matches[0]%/}
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    # Offline training reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    ENV_NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$ENV_NAME.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$ENV_NAME.npz is missing; these jobs will fail on a compute node." >&2
    fi

    TAG=${CKPT_TAGS[$c]}
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
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
