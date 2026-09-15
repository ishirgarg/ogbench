#!/usr/bin/env bash
# Submit the empowerment-entropy-target lambda sweep for flat online CRL + RLPD to the
# rnn.ist.berkeley.edu Slurm cluster on TWO more cells: antsoccer-arena-CENTER (not the
# corner variant swept by submit_flat_crl_emp_lambda_sweep.sh) and cube-single-play.
# 2 envs x 3 lambdas x 5 seeds = 30 jobs, one sbatch job per (env, lambda, seed), all
# through run_online_crl_seed.sbatch. See agents/online_crl.py for what emp_lambda does.
#
# Cells (env -> RLPD dataset -> estimator checkpoint, ckpts/final/empowerment_final, K=50):
#   asoc_ctr  antsoccer-arena-center-online-v0  antsoccer-arena-navigate-v0  antsoccer-arena-navigate/sd000_s_38390672...
#   cube_sgl  cube-single-center-online-v0      cube-single-play-v0         cube-single-play/sd000_s_38624008.0.20260908_013305
#             (the checkpoint dir submit_composed_online_lowlr_sweep.sh also uses for this cell;
#              cube-single-play has 5 duplicate empowerment_final runs, all equivalent)
#
# Episode length: antsoccer-center at 500, matching every other antsoccer online sweep in
# this repo (EPISODE_LENGTH_DEFAULTS in run_online_crl.sh); cube-single-center at its
# registered horizon (200), no override.
#
# lambda-off baselines are NOT submitted here -- 5-seed plain flat-CRL + RLPD runs already
# exist on disk for both (env, dataset) pairs (exp/OGBench/Debug/, seeds 0-4, found
# 2026-09-15): antsoccer-arena-center-online-v0 + antsoccer-arena-navigate-v0, and
# cube-single-center-online-v0 + cube-single-play-v0. Use those as the baseline curve.
#
# This ONLY submits jobs. Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_flat_crl_emp_lambda_sweep_asoc_center_cube.sh
# Overrides:
#   DRY_RUN=1                 print the sbatch commands without submitting
#   SEEDS="0 1 2 3 4"         seed set (default "0 1 2 3 4")
#   LAMBDAS="1.0"             lambda set (default "0.3 1.0 3.0")
#   GROUP_KEYS="cube_sgl"     subset of the two cells
#   EMP_NUM_BINS / EMP_NUM_SPLUS_SAMPLES   forwarded to the sbatch (defaults 8 / 64)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_seed.sbatch
LOG_DIR=logs/slurm/flat_crl_emp_lambda_sweep_asoc_center_cube
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
LAMBDAS=${LAMBDAS:-"0.3 1.0 3.0"}

EMP_ROOT=ckpts/final/empowerment_final
ALL_GROUP_KEYS=(asoc_ctr cube_sgl)
ALL_GROUP_ENVS=(
    antsoccer-arena-center-online-v0
    cube-single-center-online-v0
)
ALL_GROUP_OFFLINE=(
    antsoccer-arena-navigate-v0
    cube-single-play-v0
)
ALL_GROUP_EMP_CKPT=(
    "$EMP_ROOT/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836"
    "$EMP_ROOT/cube-single-play/sd000_s_38624008.0.20260908_013305"
)
ALL_GROUP_EPISODE_LENGTH=(500 "")

read -r -a WANTED <<< "${GROUP_KEYS:-${ALL_GROUP_KEYS[*]}}"
GROUP_KEYS=(); GROUP_ENVS=(); GROUP_OFFLINE=(); GROUP_EMP_CKPT=(); GROUP_EPISODE_LENGTH=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_GROUP_KEYS[@]}"; do
        if [[ "${ALL_GROUP_KEYS[$i]}" == "$w" ]]; then
            GROUP_KEYS+=("${ALL_GROUP_KEYS[$i]}")
            GROUP_ENVS+=("${ALL_GROUP_ENVS[$i]}")
            GROUP_OFFLINE+=("${ALL_GROUP_OFFLINE[$i]}")
            GROUP_EMP_CKPT+=("${ALL_GROUP_EMP_CKPT[$i]}")
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
    EMP_CKPT=${GROUP_EMP_CKPT[$g]}
    EPISODE_LENGTH=${GROUP_EPISODE_LENGTH[$g]}

    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "ERROR: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; compute nodes have no internet egress." >&2
        exit 1
    fi
    if [[ ! -f "$EMP_CKPT/flags.json" ]]; then
        echo "ERROR: estimator checkpoint $EMP_CKPT has no flags.json." >&2
        exit 1
    fi

    echo "# $KEY -> env=$ENV_NAME rlpd dataset=$OFFLINE_DATASET estimator=$EMP_CKPT episode_length=${EPISODE_LENGTH:-<registered>}"
    for SEED in $SEEDS; do
        for LAMBDA in $LAMBDAS; do
            JOB_NAME="crlemp_${KEY}_l${LAMBDA}_s${SEED}"
            OUT="$LOG_DIR/${JOB_NAME}_%j.log"
            cmd=(env EMP_CKPT_DIR="$EMP_CKPT" EMP_LAMBDA="$LAMBDA"
                 ${EMP_NUM_BINS:+EMP_NUM_BINS="$EMP_NUM_BINS"} ${EMP_NUM_SPLUS_SAMPLES:+EMP_NUM_SPLUS_SAMPLES="$EMP_NUM_SPLUS_SAMPLES"}
                 sbatch --job-name="$JOB_NAME" --output="$OUT"
                 "$SBATCH_SCRIPT" "$ENV_NAME" "$OFFLINE_DATASET" "$SEED" "$EPISODE_LENGTH")
            echo "${cmd[@]}"
            if [[ "$DRY_RUN" != "1" ]]; then
                "${cmd[@]}"
            fi
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
