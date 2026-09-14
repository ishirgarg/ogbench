#!/usr/bin/env bash
# Submit the OFFLINE high-level controller AWR-alpha sweep over the 3 antmaze-medium-explore
# checkpoints from the 2026-09-08 batch (job ids 38624747-38624749) to the
# rnn.ist.berkeley.edu Slurm cluster: 3 checkpoints x 4 alphas -> 12 jobs, one sbatch job
# per (checkpoint, alpha).
#
# Checkpoints (all antmaze-medium-explore-v0, seed 0):
#   ckpts/final/empowerment_final/antmaze-medium-explore/sd000_s_38624747.0.20260908_014705
#   ckpts/final/empowerment_final/antmaze-medium-explore/sd000_s_38624749.0.20260908_014705
#   ckpts/final/dds/antmaze-medium-explore/sd000_s_38624748.0.20260908_014705
# The two empowerment_final checkpoints share one env directory (unlike every other env
# dir, which holds exactly one run), so this script hardcodes full checkpoint paths
# instead of globbing by env dir -- same pattern as
# submit_online_crl_skill_controller_rlpd_seeds.sh.
#
# The alphas are the full {0.3, 1, 3, 10} grid used by
# submit_empowerment_controller_alpha_sweep.sh / submit_dds_controller_alpha_sweep.sh
# (originally from scripts/run_skill_bc_relabel_awr_sweep_antsoccer.sh); none of these
# three checkpoints has a controller trained yet, so all four alphas are swept for all
# three checkpoints.
#
# The empowerment_final checkpoints use run_empowerment_controller_alpha_seed.sbatch
# (agents/skill_bc_relabel_controller.py:gciql); the dds checkpoint uses
# run_dds_controller_alpha_seed.sbatch (agents/dds_controller.py:gciql). Each run trains
# only the gciql high level; the pretrained agent is loaded read-only and results land in
# <SKILL_CKPT>/controller_awr_sweep/alpha<ALPHA>/OGBench/Debug/sd000_s_<jobid>.<ts>/, so
# nothing under <SKILL_CKPT> is touched.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn login
# node, from this NAS checkout:
#   bash scripts/slurm/submit_antmaze_explore_controller_alpha_sweep.sh
# Overrides:
#   DRY_RUN=1                 print the sbatch commands without submitting
#   ALPHAS="0.3 1"            a different alpha grid
#   SEEDS="0 1"               more than the single seed 0
#   CKPTS="dds emp47 emp49"   a subset of the three checkpoints (see tags below)
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * antmaze-medium-explore-v0.npz must already be in
#     /nas/ucb/ishirgarg/.ogbench/data (compute nodes have no internet egress).
#     Present as of 2026-09-13.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

EMP_ROOT=${EMP_ROOT:-ckpts/final/empowerment_final}
DDS_ROOT=${DDS_ROOT:-ckpts/final/dds}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
LOG_DIR=logs/slurm/antmaze_explore_controller_alpha
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
ALPHAS=${ALPHAS:-"0.3 1 3 10"}
SEEDS=${SEEDS:-"0"}

# Parallel arrays: tag -> sbatch script, checkpoint path.
ALL_TAGS=(dds emp47 emp49)
ALL_SBATCH_SCRIPTS=(
    scripts/slurm/run_dds_controller_alpha_seed.sbatch
    scripts/slurm/run_empowerment_controller_alpha_seed.sbatch
    scripts/slurm/run_empowerment_controller_alpha_seed.sbatch
)
ALL_CKPTS=(
    "$DDS_ROOT/antmaze-medium-explore/sd000_s_38624748.0.20260908_014705"
    "$EMP_ROOT/antmaze-medium-explore/sd000_s_38624747.0.20260908_014705"
    "$EMP_ROOT/antmaze-medium-explore/sd000_s_38624749.0.20260908_014705"
)

read -r -a WANTED <<< "${CKPTS:-${ALL_TAGS[*]}}"
TAGS=(); SBATCH_SCRIPTS=(); CKPTS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_TAGS[@]}"; do
        if [[ "${ALL_TAGS[$i]}" == "$w" ]]; then
            TAGS+=("${ALL_TAGS[$i]}")
            SBATCH_SCRIPTS+=("${ALL_SBATCH_SCRIPTS[$i]}")
            CKPTS+=("${ALL_CKPTS[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown checkpoint tag '$w' (known: ${ALL_TAGS[*]})" >&2; exit 1; }
done

# antmaze-medium-explore-v0 is the offline training env for all three checkpoints; warn
# early if it isn't on the NAS.
if [[ ! -f "$DATASET_DIR/antmaze-medium-explore-v0.npz" ]]; then
    echo "WARNING: $DATASET_DIR/antmaze-medium-explore-v0.npz is missing; these jobs will fail on a compute node." >&2
fi

n_submitted=0
for c in "${!TAGS[@]}"; do
    TAG=${TAGS[$c]}
    SBATCH_SCRIPT=${SBATCH_SCRIPTS[$c]}
    SKILL_CKPT=${CKPTS[$c]}
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    for ALPHA in $ALPHAS; do
        for SEED in $SEEDS; do
            JOB_NAME="antexctrl_a${ALPHA}_${TAG}_s${SEED}"
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
