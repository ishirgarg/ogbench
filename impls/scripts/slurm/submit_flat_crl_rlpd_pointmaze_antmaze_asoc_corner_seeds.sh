#!/usr/bin/env bash
# Submit online_crl (flat CRL, no skill controller) + RLPD multi-seed sweeps to the
# rnn.ist.berkeley.edu Slurm cluster, over 3 online envs x 2 RLPD dataset variants x
# 5 seeds, one sbatch job per (env, offline_dataset, seed):
#
#   pointmaze-teleport-sparse-online-v0     + {pointmaze-teleport-navigate-v0, pointmaze-teleport-stitch-v0}
#   antmaze-medium-corner-sparse-online-v0  + {antmaze-medium-navigate-v0,     antmaze-medium-stitch-v0}
#   antsoccer-arena-corner-online-v0        + {antsoccer-arena-navigate-v0,    antsoccer-arena-stitch-v0}
#
#   3 envs x 2 RLPD dataset variants x 5 seeds = 30 jobs.
#
# "navigate" / "stitch" here select which offline dataset RLPD mixes in (see
# utils/rlpd.py, agents/online_crl.py) -- they do NOT change the online env or its
# task.
#
# NOTE (2026-09-12): the pointmaze/antmaze envs were originally the older, denser
# `*-center-` task sets (41/22 goals) -- WRONG, caught after a first 30-job sweep
# already ran to completion on them. Fixed to use the NEW sparse task sets
# (pointmaze-teleport-sparse-online-v0: 4 goals from the fixed start;
# antmaze-medium-corner-sparse-online-v0: 3 goals from the bottom-left corner),
# registered 2026-09-12 in ogbench/locomaze/__init__.py. The antsoccer-arena-corner
# env was correct from the start and is unchanged. This mirrors
# submit_online_crl_rlpd_seeds.sh (which covers the OLD antmaze-medium-center-online-v0
# and antsoccer-arena-CENTER-online-v0), but is a full, fresh 5-seed sweep on
# different envs -- results here are a separate `exp/` tree, not a continuation of
# that script's runs (or of the earlier wrong center-env sweep, which is untouched
# on disk).
#
# Every job uses agents/online_crl.py (flat CRL) with RLPD on; see
# run_online_crl_seed.sbatch for the exact flags (1M env steps, same
# log/eval/save intervals as every other online sweep in this repo).
#
# Episode length: pointmaze-teleport-sparse and antmaze-medium-corner-sparse are
# registered at horizon 1000 (used as-is, no override -- flat CRL has no
# skill_commitment_k divisibility requirement, but 1000 matches every other
# sweep on these envs). antsoccer-arena-corner-online-v0 is registered at 1000 but,
# like every other antsoccer online env in this repo (see
# submit_antsoccer_corner_online_controller_seeds.sh), is run at 500 so results
# stay comparable to the rest of the antsoccer sweeps.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn
# login node, from this NAS checkout:
#   bash scripts/slurm/submit_flat_crl_rlpd_pointmaze_antmaze_asoc_corner_seeds.sh
# Overrides:
#   DRY_RUN=1              print the sbatch commands without submitting
#   SEEDS="0 1"            a different seed set (applies to every group)
#   GROUP_KEYS="pmt_nav amz_sti"   a subset of the six (env, variant) groups
#
# Prerequisites (checked below):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each offline dataset must be present in /nas/ucb/ishirgarg/.ogbench/data
#     (compute nodes have no internet egress). pointmaze-teleport-stitch-v0.npz
#     was NOT there as of 2026-09-04 (see submit_dds_online_controller_seeds.sh) --
#     verify before submitting for real.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_seed.sbatch
LOG_DIR=logs/slurm/flat_crl_rlpd_pointmaze_antmaze_asoc_corner
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}

# Parallel arrays, one entry per (env, offline_dataset) group.
ALL_GROUP_KEYS=(pmt_nav pmt_sti amz_nav amz_sti asoc_nav asoc_sti)
ALL_GROUP_ENVS=(
    pointmaze-teleport-sparse-online-v0
    pointmaze-teleport-sparse-online-v0
    antmaze-medium-corner-sparse-online-v0
    antmaze-medium-corner-sparse-online-v0
    antsoccer-arena-corner-online-v0
    antsoccer-arena-corner-online-v0
)
ALL_GROUP_OFFLINE=(
    pointmaze-teleport-navigate-v0
    pointmaze-teleport-stitch-v0
    antmaze-medium-navigate-v0
    antmaze-medium-stitch-v0
    antsoccer-arena-navigate-v0
    antsoccer-arena-stitch-v0
)
ALL_GROUP_EPISODE_LENGTH=("" "" "" "" 500 500)

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
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi

    echo "# $KEY -> env=$ENV_NAME rlpd dataset=$OFFLINE_DATASET episode_length=${EPISODE_LENGTH:-<registered>}"
    for SEED in $SEEDS; do
        JOB_NAME="crl_${KEY}_s${SEED}"
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
