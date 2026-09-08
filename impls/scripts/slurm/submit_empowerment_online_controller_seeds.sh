#!/usr/bin/env bash
# Submit the online CRL skill controller over the 4 **empowerment_skill** checkpoints in
# ckpts/final/empowerment_final to the rnn.ist.berkeley.edu Slurm cluster: 5 seeds (0-4)
# per checkpoint, one sbatch job per (checkpoint, seed) -> 20 jobs.
#
# Same setup as the DDS controller sweep (submit_dds_online_controller_seeds.sh) and the
# antmaze empowerment sweep (submit_online_crl_skill_controller_rlpd_seeds.sh): RLPD on,
# offline dataset = each checkpoint's OWN dataset (the data its low-level skills were
# trained on, read from flags.json by the sbatch script), K=10 skill commitment,
# target_entropy_multiplier=0.5, 1M env steps. See
# run_online_crl_skill_controller_seed.sbatch for the exact flags.
#
# The 4 checkpoints (ckpts/final/empowerment_final/<env>/sd000_*), all K=50 skills, 1M
# steps, from the 2026-09-01 Savio batch -- exactly the set in
# submit_empowerment_controller_alpha_sweep.sh, so the offline and online high levels are
# trained on the same frozen skills.
#
# Online env per checkpoint (the deterministic, noise-free `*-center-` task sets; see
# ogbench/locomaze/__init__.py):
#   antsoccer-arena-{navigate,stitch}    -> antsoccer-arena-center-online-v0    (horizon 500, overridden)
#   pointmaze-teleport-{navigate,stitch} -> pointmaze-teleport-center-online-v0 (horizon 1000)
# antsoccer-arena-center-v0 is *registered* at 1000, but every online run in this repo
# uses 500 for it (scripts/run_online_crl.sh's per-env table, and the DDS sweep), so it is
# passed explicitly here too. Both horizons are divisible by K=10, which main_online.py
# requires for macro rollouts.
#
# --time is raised to 16 h on the sbatch command line (the script's own directive is 8 h,
# sized for the shorter antmaze runs), matching the DDS online sweep.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn login
# node, from this NAS checkout:
#   bash scripts/slurm/submit_empowerment_online_controller_seeds.sh
# Overrides:
#   DRY_RUN=1                 print the sbatch commands without submitting
#   SEEDS="0 1"               a different seed set
#   CKPT_DIRS="antsoccer-arena-stitch ..."   a subset of the four env dirs
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each checkpoint's offline dataset must be present in
#     /nas/ucb/ishirgarg/.ogbench/data (compute nodes have no internet egress).
#     All four were present as of 2026-09-06.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_skill_controller_seed.sbatch
LOG_DIR=logs/slurm/empowerment_online_controller
EMP_ROOT=${EMP_ROOT:-ckpts/final/empowerment_final}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
TIME_LIMIT=${TIME_LIMIT:-16:00:00}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}

# Parallel arrays: checkpoint env dir, short tag, online env, episode-length override.
ALL_CKPT_DIRS=(
    antsoccer-arena-navigate
    antsoccer-arena-stitch
    pointmaze-teleport-navigate
    pointmaze-teleport-stitch
)
ALL_CKPT_TAGS=(asoc_nav asoc_sti pmt_nav pmt_sti)
ALL_ONLINE_ENVS=(
    antsoccer-arena-center-online-v0
    antsoccer-arena-center-online-v0
    pointmaze-teleport-center-online-v0
    pointmaze-teleport-center-online-v0
)
ALL_EPISODE_LENGTHS=(500 500 "" "")

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
    # Resolve by glob so a re-rsync with a different job id still works; each env dir must
    # hold exactly one sd000_* run. (antmaze-medium-{navigate,stitch} hold two -- a k=50 and
    # a k=15 run -- which is one reason they are not in this sweep; their k=50 seeds are
    # already covered by submit_online_crl_skill_controller_rlpd_seeds.sh.)
    matches=("$EMP_ROOT/${CKPT_DIRS[$c]}"/sd000_*/)
    if (( ${#matches[@]} != 1 )) || [[ ! -d "${matches[0]}" ]]; then
        echo "ERROR: expected exactly one sd000_* run under $EMP_ROOT/${CKPT_DIRS[$c]}, found ${#matches[@]}" >&2
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
        JOB_NAME="empctrl_${TAG}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$TIME_LIMIT"
             "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "" "$EPISODE_LENGTH")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
