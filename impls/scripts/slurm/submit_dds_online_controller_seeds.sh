#!/usr/bin/env bash
# Submit the online CRL skill controller over the 5 final **DDS** checkpoints to
# the rnn.ist.berkeley.edu Slurm cluster: 5 seeds (0-4) per checkpoint, one
# sbatch job per (checkpoint, seed) -> 25 jobs.
#
# Same setup as the empowerment_skill controller sweep
# (submit_online_crl_skill_controller_rlpd_seeds.sh): RLPD on, offline dataset =
# each checkpoint's OWN dataset (the data its skills were trained on, read from
# flags.json by the sbatch script), K=10, target_entropy_multiplier=0.5, 1M env
# steps. See run_dds_online_controller_seed.sbatch for the exact flags.
#
# Online env per checkpoint (the deterministic, noise-free `*-center-` task sets;
# see ogbench/locomaze/__init__.py):
#   antmaze-medium-{navigate,stitch}   -> antmaze-medium-center-online-v0     (horizon 1000)
#   antsoccer-arena-navigate           -> antsoccer-arena-center-online-v0    (horizon 500, overridden)
#   pointmaze-teleport-{navigate,stitch}-> pointmaze-teleport-center-online-v0 (horizon 1000)
# All three horizons are divisible by K=10, which main_online.py requires for
# macro rollouts.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the
# rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_dds_online_controller_seeds.sh
# Add DRY_RUN=1 to print the sbatch commands without submitting.
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each checkpoint's offline dataset must be present in
#     /nas/ucb/ishirgarg/.ogbench/data (compute nodes have no internet egress).
#     pointmaze-teleport-stitch-v0.npz was NOT there as of 2026-09-04.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_dds_online_controller_seed.sbatch
LOG_DIR=logs/slurm/dds_online_controller
DDS_ROOT=${DDS_ROOT:-ckpts/final/dds}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}

# Parallel arrays: checkpoint env dir, short tag, online env, episode-length override.
ALL_CKPT_DIRS=(
    antmaze-medium-navigate
    antmaze-medium-stitch
    antsoccer-arena-navigate
    pointmaze-teleport-navigate
    pointmaze-teleport-stitch
)
ALL_CKPT_TAGS=(amz_nav amz_sti asoc_nav pmt_nav pmt_sti)
ALL_ONLINE_ENVS=(
    antmaze-medium-center-online-v0
    antmaze-medium-center-online-v0
    antsoccer-arena-center-online-v0
    pointmaze-teleport-center-online-v0
    pointmaze-teleport-center-online-v0
)
ALL_EPISODE_LENGTHS=("" "" 500 "" "")

# CKPT_DIRS selects a subset by env-dir name, e.g. to submit the four complete
# checkpoints while the antmaze-medium-navigate rsync is still short:
#   CKPT_DIRS="antmaze-medium-stitch antsoccer-arena-navigate ..." bash <this script>
read -r -a WANTED <<< "${CKPT_DIRS:-${ALL_CKPT_DIRS[*]}}"
CKPT_DIRS=(); CKPT_TAGS=(); ONLINE_ENVS=(); EPISODE_LENGTHS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_CKPT_DIRS[@]}"; do
        if [[ "${ALL_CKPT_DIRS[$i]}" == "$w" ]]; then
            CKPT_DIRS+=("${ALL_CKPT_DIRS[$i]}")
            CKPT_TAGS+=("${ALL_CKPT_TAGS[$i]}")
            ONLINE_ENVS+=("${ALL_ONLINE_ENVS[$i]}")
            EPISODE_LENGTHS+=("${ALL_EPISODE_LENGTHS[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown checkpoint dir '$w' (known: ${ALL_CKPT_DIRS[*]})" >&2; exit 1; }
done

n_submitted=0
for c in "${!CKPT_DIRS[@]}"; do
    # Resolve by glob so a re-rsync with a different job id still works.
    matches=("$DDS_ROOT/${CKPT_DIRS[$c]}"/sd000_*/)
    if (( ${#matches[@]} != 1 )) || [[ ! -d "${matches[0]}" ]]; then
        echo "ERROR: expected exactly one sd000_* run under $DDS_ROOT/${CKPT_DIRS[$c]}" >&2
        exit 1
    fi
    SKILL_CKPT=${matches[0]%/}
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    # RLPD reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    OFFLINE_DATASET=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi

    TAG=${CKPT_TAGS[$c]}
    ENV_NAME=${ONLINE_ENVS[$c]}
    EPISODE_LENGTH=${EPISODE_LENGTHS[$c]}
    for SEED in $SEEDS; do
        JOB_NAME="ddsctrl_${TAG}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
             "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "" "$EPISODE_LENGTH")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
