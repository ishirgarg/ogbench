#!/usr/bin/env bash
# The alpha=0 baseline for the explore-bonus family: plain flat CRL + RLPD, no exploration
# bonus at all, on the same 3 cells / RLPD datasets / episode lengths / seeds as
# scripts/slurm/submit_flat_crl_explore_bonus_sweep.sh (216 jobs) and
# submit_flat_crl_explore_bonus_annealed_sweep.sh (144 jobs) -- so it's the "everything else
# matching" comparison line for both of those sweeps' plots (bonus_scale=0 needs no annealing
# either way, so one baseline arm serves both).
#
# 3 cells x 3 seeds = 9 jobs, one sbatch job per (cell, seed), through run_online_crl_seed.sbatch
# with every bonus-related env var left UNSET so its EMP_FLAG/ADD_EXPLORE logic never fires.
#
# NOTE: this is a fresh, standalone submitter, not `submit_flat_crl_explore_bonus_sweep.sh
# INCLUDE_BASELINE=1` -- that script's SCALES/MODES/REWARDS use `${VAR:-default}`, which falls
# back to the full default grid even when the caller passes an explicitly-empty override, so
# there was no way to get *only* the 9 baseline jobs out of it without resubmitting all 216.
#
# This ONLY submits jobs. Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_flat_crl_explore_bonus_baselines.sh
# Overrides: DRY_RUN=1, SEEDS="0 1 2" (default), GROUP_KEYS="pmt_nav cube_sgl"
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_seed.sbatch
LOG_DIR=logs/slurm/flat_crl_explore_bonus_sweep
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2"}

ALL_GROUP_KEYS=(asoc_ctr pmt_nav cube_sgl)
ALL_GROUP_ENVS=(
    antsoccer-arena-center-online-v0
    pointmaze-teleport-center-online-v0
    cube-single-center-online-v0
)
ALL_GROUP_OFFLINE=(
    antsoccer-arena-navigate-v0
    pointmaze-teleport-navigate-v0
    cube-single-play-v0
)
ALL_GROUP_EPISODE_LENGTH=(500 "" "")

read -r -a WANTED <<< "${GROUP_KEYS:-${ALL_GROUP_KEYS[*]}}"
GROUP_KEYS=(); GROUP_ENVS=(); GROUP_OFFLINE=(); GROUP_EPISODE_LENGTH=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_GROUP_KEYS[@]}"; do
        if [[ "${ALL_GROUP_KEYS[$i]}" == "$w" ]]; then
            GROUP_KEYS+=("${ALL_GROUP_KEYS[$i]}")
            GROUP_ENVS+=("${ALL_GROUP_ENVS[$i]}")
            GROUP_OFFLINE+=("${ALL_GROUP_OFFLINE[$i]}")
            GROUP_EPISODE_LENGTH+=("${ALL_GROUP_EPISODE_LENGTH[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown group key '$w' (known: ${ALL_GROUP_KEYS[*]})" >&2; exit 1; }
done

n_submitted=0
for g in "${!GROUP_KEYS[@]}"; do
    KEY=${GROUP_KEYS[$g]}
    ENV_NAME=${GROUP_ENVS[$g]}
    OFFLINE_DATASET=${GROUP_OFFLINE[$g]}
    EPISODE_LENGTH=${GROUP_EPISODE_LENGTH[$g]}

    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "ERROR: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; compute nodes have no internet egress." >&2
        exit 1
    fi

    for SEED in $SEEDS; do
        JOB_NAME="crlrb_${KEY}_baseline_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        cmd=(env EMP_CKPT_DIR= EMP_ENTROPY_TARGET= ADD_EXPLORE= BONUS_SCALE= EXPLORE_REWARD= BONUS_TIME_FRAC= \
             sbatch --job-name="$JOB_NAME" --output="$OUT" \
             "$SBATCH_SCRIPT" "$ENV_NAME" "$OFFLINE_DATASET" "$SEED" "$EPISODE_LENGTH")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
