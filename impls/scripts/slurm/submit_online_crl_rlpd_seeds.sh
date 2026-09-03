#!/usr/bin/env bash
# Submit online_crl (flat CRL) multi-seed RLPD sweeps to the rnn.ist.berkeley.edu
# Slurm cluster, one sbatch job per (env, offline_dataset, seed).
#
# Goal: 5 identical seeds (0-4) per (env, offline_dataset) combo, for confidence
# intervals. Two combos already have a seed-0 run finished locally (see
# logs/online_crl/*_rlpd_s0.log): antmaze-medium-center-online-v0 with
# antmaze-medium-navigate-v0, and antsoccer-arena-center-online-v0 with
# antsoccer-arena-navigate-v0. This script submits only what's missing:
#
#   antsoccer-arena-center-online-v0 + antsoccer-arena-navigate-v0  -> seeds 1-4 (4 runs)
#   antsoccer-arena-center-online-v0 + antsoccer-arena-stitch-v0    -> seeds 0-4 (5 runs)
#   antmaze-medium-center-online-v0  + antmaze-medium-navigate-v0   -> seeds 1-4 (4 runs)
#   antmaze-medium-center-online-v0  + antmaze-medium-stitch-v0     -> seeds 0-4 (5 runs)
#
# Total: 18 jobs. Every run uses agents/online_crl.py (flat CRL, no skill
# controller) with RLPD on (see run_online_crl_seed.sbatch for the exact flags,
# which match scripts/run_online_crl.sh's per-run body so seed 0 of the two
# "-4 more" groups is directly comparable to the already-finished local runs).
#
# This ONLY submits jobs -- it does not run any training itself. Run from the
# rnn login node (`ssh rnn`, then from this NAS checkout):
#   bash scripts/slurm/submit_online_crl_rlpd_seeds.sh
# Add DRY_RUN=1 to print the sbatch commands without submitting:
#   DRY_RUN=1 bash scripts/slurm/submit_online_crl_rlpd_seeds.sh
#
# Before running for real, make sure wandb credentials will be visible inside
# the Slurm jobs -- see the comment block in run_online_crl_seed.sbatch.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_seed.sbatch
LOG_DIR=logs/slurm/online_crl
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}

# Parallel arrays, one entry per (env, offline_dataset) group; SEEDS is a
# space-separated list of seeds still needed for that group.
GROUP_ENVS=(
    antsoccer-arena-center-online-v0
    antsoccer-arena-center-online-v0
    antmaze-medium-center-online-v0
    antmaze-medium-center-online-v0
)
GROUP_OFFLINE=(
    antsoccer-arena-navigate-v0
    antsoccer-arena-stitch-v0
    antmaze-medium-navigate-v0
    antmaze-medium-stitch-v0
)
GROUP_EPISODE_LENGTH=(500 500 "" "")
GROUP_SEEDS=(
    "1 2 3 4"
    "0 1 2 3 4"
    "1 2 3 4"
    "0 1 2 3 4"
)

n_submitted=0
for g in "${!GROUP_ENVS[@]}"; do
    ENV_NAME=${GROUP_ENVS[$g]}
    OFFLINE_DATASET=${GROUP_OFFLINE[$g]}
    EPISODE_LENGTH=${GROUP_EPISODE_LENGTH[$g]}
    for SEED in ${GROUP_SEEDS[$g]}; do
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
