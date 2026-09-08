#!/usr/bin/env bash
# Submit the OFFLINE DDS high-level controller AWR-alpha sweep to the
# rnn.ist.berkeley.edu Slurm cluster: one sbatch job per (checkpoint, alpha).
#
# TRIMMED 2026-09-06 to the two jobs still wanted: pointmaze-teleport-navigate at
# alpha in {0.3, 1}. The full grid this script originally submitted was 5 checkpoints x
# {0.3, 1, 10} = 15 jobs; the lookup tables below still carry all five checkpoints, so
# the rest is one override away, e.g.
#   CKPT_DIRS="antmaze-medium-navigate antmaze-medium-stitch antsoccer-arena-navigate \
#              pointmaze-teleport-stitch" ALPHAS="0.3 1 10" bash <this script>
#
# The alphas come from the empowerment_skill controller grid
# (scripts/run_skill_bc_relabel_awr_sweep_antsoccer.sh, {0.3, 1, 3, 10}). alpha=3.0 is
# deliberately excluded -- it is agents/dds_controller.py's paper default and is already
# trained on all five checkpoints by scripts/run_dds_controller_final5_local.sh, under
# <SKILL_CKPT>/controller/. Adding it back is a matter of ALPHAS="0.3 1 3 10", but that
# would be a duplicate run in a different directory.
#
# The checkpoints (ckpts/final/dds/<env>/sd000_*) are all K=50, H=10, 1M steps -- exactly
# the set in run_dds_controller_final5_local.sh.
#
# Each run trains only the gciql high level; the pretrained agent is loaded read-only and
# results land in <SKILL_CKPT>/controller_awr_sweep/alpha<ALPHA>/OGBench/Debug/
# sd000_s_<jobid>.<ts>/, so nothing under <SKILL_CKPT> or <SKILL_CKPT>/controller/ is
# touched. See run_dds_controller_alpha_seed.sbatch for the exact flags.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn login
# node, from this NAS checkout:
#   bash scripts/slurm/submit_dds_controller_alpha_sweep.sh
# Overrides:
#   DRY_RUN=1                 print the sbatch commands without submitting
#   ALPHAS="0.3 1 10"         a different alpha grid
#   SEEDS="0 1"               more than the single seed 0
#   CKPT_DIRS="antmaze-medium-stitch ..."   a different subset of the five env dirs
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each checkpoint's OFFLINE dataset must already be in
#     /nas/ucb/ishirgarg/.ogbench/data (compute nodes have no internet egress).
#     All five were present as of 2026-09-05.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_dds_controller_alpha_seed.sbatch
LOG_DIR=logs/slurm/dds_controller_alpha
DDS_ROOT=${DDS_ROOT:-ckpts/final/dds}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
ALPHAS=${ALPHAS:-"0.3 1"}
SEEDS=${SEEDS:-"0"}

# Parallel arrays: checkpoint env dir -> short job-name tag.
ALL_CKPT_DIRS=(
    antmaze-medium-navigate
    antmaze-medium-stitch
    antsoccer-arena-navigate
    pointmaze-teleport-navigate
    pointmaze-teleport-stitch
)
ALL_CKPT_TAGS=(amz_nav amz_sti asoc_nav pmt_nav pmt_sti)

# Only this one by default (see the TRIMMED note in the header); the arrays above stay
# complete so CKPT_DIRS can select any of the five.
DEFAULT_CKPT_DIRS="pointmaze-teleport-navigate"

read -r -a WANTED <<< "${CKPT_DIRS:-$DEFAULT_CKPT_DIRS}"
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
    # hold exactly one sd000_* run.
    matches=("$DDS_ROOT/${CKPT_DIRS[$c]}"/sd000_*/)
    if (( ${#matches[@]} != 1 )) || [[ ! -d "${matches[0]}" ]]; then
        echo "ERROR: expected exactly one sd000_* run under $DDS_ROOT/${CKPT_DIRS[$c]}, found ${#matches[@]}" >&2
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
            JOB_NAME="ddsctrl_a${ALPHA}_${TAG}_s${SEED}"
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
