#!/usr/bin/env bash
# Submit the flat online CRL baseline (agents/online_crl.py, RLPD ON) on the cube
# and pointmaze-teleport envs to the rnn.ist.berkeley.edu Slurm cluster:
# 5 seeds (0-4) per (env, offline_dataset) group, one sbatch job per pair -> 20 jobs.
#
# Companion of submit_online_crl_rlpd_seeds.sh (which covers the antmaze/antsoccer
# groups); it reuses the same per-run body, run_online_crl_seed.sbatch, so every
# baseline run across all envs shares one set of flags/intervals.
#
#   cube-single-center-online-v0        + cube-single-play-v0            -> seeds 0-4
#   cube-double-center-online-v0        + cube-double-play-v0            -> seeds 0-4
#   pointmaze-teleport-center-online-v0 + pointmaze-teleport-navigate-v0 -> seeds 0-4
#   pointmaze-teleport-center-online-v0 + pointmaze-teleport-stitch-v0   -> seeds 0-4
#
# The two pointmaze groups share one online env and differ only in the RLPD
# dataset, matching how the antmaze/antsoccer groups are split.
#
# Envs: the deterministic, noise-free `*-center-v0` registrations (one fixed start;
# 41 per-cell goals for pointmaze-teleport, one fixed pick-and-place task for each
# cube env). Episode length is left at each env's registered horizon -- 1000 for
# pointmaze-teleport, 200 for cube-single, 500 for cube-double.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn
# login node, from this NAS checkout:
#   bash scripts/slurm/submit_online_crl_rlpd_seeds_cube_pointmaze.sh
# Add DRY_RUN=1 to print the sbatch commands without submitting.
#
# Prerequisites:
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Every offline dataset below must be present in /nas/ucb/ishirgarg/.ogbench/data
#     (compute nodes have no internet egress). As of 2026-09-04 cube-single-play-v0
#     and pointmaze-teleport-stitch-v0 were NOT there -- the checks below will warn.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_seed.sbatch
LOG_DIR=logs/slurm/online_crl
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}

# Parallel arrays, one entry per (env, offline_dataset) group.
GROUP_ENVS=(
    cube-single-center-online-v0
    cube-double-center-online-v0
    pointmaze-teleport-center-online-v0
    pointmaze-teleport-center-online-v0
)
GROUP_OFFLINE=(
    cube-single-play-v0
    cube-double-play-v0
    pointmaze-teleport-navigate-v0
    pointmaze-teleport-stitch-v0
)
# Empty -> the env's registered horizon (200 / 500 / 1000 / 1000).
GROUP_EPISODE_LENGTH=("" "" "" "")

n_submitted=0
for g in "${!GROUP_ENVS[@]}"; do
    ENV_NAME=${GROUP_ENVS[$g]}
    OFFLINE_DATASET=${GROUP_OFFLINE[$g]}
    EPISODE_LENGTH=${GROUP_EPISODE_LENGTH[$g]}
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi
    for SEED in $SEEDS; do
        JOB_NAME="crl_${ENV_NAME%-online-v0}_$(basename "$OFFLINE_DATASET" -v0)_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
             "$SBATCH_SCRIPT" "$ENV_NAME" "$OFFLINE_DATASET" "$SEED" "$EPISODE_LENGTH")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
