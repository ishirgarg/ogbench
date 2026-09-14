#!/usr/bin/env bash
# Submit the ONLINE high-level skill controller (agents/online_crl_skill_controller.py)
# with RLPD over the 3 antmaze-medium-explore checkpoints from the 2026-09-08 batch
# (job ids 38624747-38624749) to the rnn.ist.berkeley.edu Slurm cluster: 5 seeds (0-4)
# per checkpoint, one sbatch job per (checkpoint, seed) -> 15 jobs.
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
# Online env: antmaze-medium-center-online-v0, the deterministic noise-free antmaze-medium
# online task set (registered horizon 1000, divisible by skill_commitment_k=10 as
# main_online.py requires), same env used by submit_dds_online_controller_seeds.sh and
# submit_online_crl_skill_controller_rlpd_seeds.sh for the other antmaze-medium
# checkpoints.
#
# RLPD is on: OFFLINE_DATASET is left unset, so each sbatch job defaults it to the
# checkpoint's own dataset (its flags.json env_name), which is antmaze-medium-explore-v0
# for all three checkpoints -- i.e. RLPD trains on the same explore data the skills were
# trained on. K=10 skill commitment, target_entropy_multiplier=0.5, 1M env steps -- see
# run_dds_online_controller_seed.sbatch / run_online_crl_skill_controller_seed.sbatch for
# the exact flags.
#
# The empowerment_final checkpoints use run_online_crl_skill_controller_seed.sbatch; the
# dds checkpoint uses run_dds_online_controller_seed.sbatch. Both invoke the same agent
# module and only differ in job-name/log defaults; results land in
# <SKILL_CKPT>/online_controller/rlpd/, so nothing under <SKILL_CKPT> itself is touched.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn login
# node, from this NAS checkout:
#   bash scripts/slurm/submit_antmaze_explore_online_controller_seeds.sh
# Overrides:
#   DRY_RUN=1                 print the sbatch commands without submitting
#   SEEDS="0 1"               a different seed set
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
LOG_DIR=logs/slurm/antmaze_explore_online_controller
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
ENV_NAME=antmaze-medium-center-online-v0

# Parallel arrays: tag -> sbatch script, checkpoint path.
ALL_TAGS=(dds emp47 emp49)
ALL_SBATCH_SCRIPTS=(
    scripts/slurm/run_dds_online_controller_seed.sbatch
    scripts/slurm/run_online_crl_skill_controller_seed.sbatch
    scripts/slurm/run_online_crl_skill_controller_seed.sbatch
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

# RLPD reads each checkpoint's own dataset (antmaze-medium-explore-v0 for all three);
# warn early if it isn't on the NAS.
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

    for SEED in $SEEDS; do
        JOB_NAME="antexctrl_${TAG}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        # Positional args: SKILL_CKPT ENV_NAME SEED [OFFLINE_DATASET] [EPISODE_LENGTH].
        # Both left blank: OFFLINE_DATASET defaults to the checkpoint's own dataset
        # (RLPD on the explore data) and EPISODE_LENGTH defaults to the env's registered
        # horizon (1000).
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
             "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "" "")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
