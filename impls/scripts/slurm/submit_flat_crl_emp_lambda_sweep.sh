#!/usr/bin/env bash
# Submit the empowerment-entropy-target lambda sweep for flat online CRL + RLPD to the
# rnn.ist.berkeley.edu Slurm cluster: 3 online envs x 4 lambdas x 4 seeds = 48 jobs, one
# sbatch job per (env, lambda, seed), all through run_online_crl_seed.sbatch.
#
# What is swept: agents/online_crl.py's `emp_lambda`, the nats of per-state entropy
# target per nat of (E(s) - E_mean), with E(s) from the frozen offline empowerment_skill
# estimator matched to each cell's RLPD dataset (see the module docstring of
# agents/online_crl.py). Everything else is the plain flat-CRL + RLPD recipe of
# run_online_crl_seed.sbatch (1M env steps, same intervals), so the lambda-off
# baselines are the runs of submit_flat_crl_rlpd_pointmaze_antmaze_asoc_corner_seeds.sh
# on the same (env, dataset) cells (pmt_sti, amz_nav, asoc_nav there). INCLUDE_BASELINE=1
# adds a fresh lambda-off run per (env, seed) to this submission (12 more jobs).
#
# Cells (env -> RLPD dataset -> estimator checkpoint, ckpts/final/empowerment_final, K=50):
#   pmt_sti  pointmaze-teleport-sparse-online-v0    pointmaze-teleport-stitch-v0   pointmaze-teleport-stitch/sd000_s_38390675...
#   amz_nav  antmaze-medium-corner-sparse-online-v0 antmaze-medium-navigate-v0     antmaze-medium-navigate/sd000_s_37866290..._k50_...
#   asoc_nav antsoccer-arena-corner-online-v0       antsoccer-arena-navigate-v0    antsoccer-arena-navigate/sd000_s_38390672...
#
# Episode length: antsoccer at 500 like every antsoccer online sweep in this repo; the
# mazes at their registered 1000.
#
# Per-job cost of the estimator: E(s) over the 1M-row RLPD dataset is computed once at
# start (~70 s on an A6000, cached under <ckpt>/empowerment_values/, keyed by seed) and
# online rows cost ~5 ms per 50-row update round.
#
# This ONLY submits jobs. Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_flat_crl_emp_lambda_sweep.sh
# Overrides:
#   DRY_RUN=1                 print the sbatch commands without submitting
#   SEEDS="0 1"               seed set (default "0 1 2 3")
#   LAMBDAS="1.0"             lambda set (default "0.3 1.0 3.0 10.0")
#   GROUP_KEYS="pmt_sti"      subset of the three cells
#   INCLUDE_BASELINE=1        also submit lambda-off (plain flat CRL + RLPD) per (env, seed)
#   EMP_NUM_BINS / EMP_NUM_SPLUS_SAMPLES   forwarded to the sbatch (defaults 8 / 64)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_seed.sbatch
LOG_DIR=logs/slurm/flat_crl_emp_lambda_sweep
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3"}
LAMBDAS=${LAMBDAS:-"0.3 1.0 3.0 10.0"}
INCLUDE_BASELINE=${INCLUDE_BASELINE:-0}

EMP_ROOT=ckpts/final/empowerment_final
ALL_GROUP_KEYS=(pmt_sti amz_nav asoc_nav)
ALL_GROUP_ENVS=(
    pointmaze-teleport-sparse-online-v0
    antmaze-medium-corner-sparse-online-v0
    antsoccer-arena-corner-online-v0
)
ALL_GROUP_OFFLINE=(
    pointmaze-teleport-stitch-v0
    antmaze-medium-navigate-v0
    antsoccer-arena-navigate-v0
)
ALL_GROUP_EMP_CKPT=(
    "$EMP_ROOT/pointmaze-teleport-stitch/sd000_s_38390675.0.20260901_154836"
    "$EMP_ROOT/antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001"
    "$EMP_ROOT/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836"
)
ALL_GROUP_EPISODE_LENGTH=("" "" 500)

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
submit() {  # submit KEY ENV OFFLINE EPISODE_LENGTH SEED LAMBDA|off EMP_CKPT
    local KEY=$1 ENV_NAME=$2 OFFLINE_DATASET=$3 EPISODE_LENGTH=$4 SEED=$5 LAMBDA=$6 EMP_CKPT=$7
    local JOB_NAME="crlemp_${KEY}_l${LAMBDA}_s${SEED}"
    local OUT="$LOG_DIR/${JOB_NAME}_%j.log"
    # sbatch exports the caller's environment (--export=ALL default), so the EMP_* variables
    # reach run_online_crl_seed.sbatch; an empty EMP_CKPT_DIR means the plain agent.
    local env_vars=(EMP_CKPT_DIR="$EMP_CKPT" EMP_LAMBDA="$LAMBDA")
    if [[ "$LAMBDA" == "off" ]]; then env_vars=(EMP_CKPT_DIR= EMP_LAMBDA=); fi
    [[ -n "${EMP_NUM_BINS:-}" ]] && env_vars+=(EMP_NUM_BINS="$EMP_NUM_BINS")
    [[ -n "${EMP_NUM_SPLUS_SAMPLES:-}" ]] && env_vars+=(EMP_NUM_SPLUS_SAMPLES="$EMP_NUM_SPLUS_SAMPLES")
    local cmd=(env "${env_vars[@]}" sbatch --job-name="$JOB_NAME" --output="$OUT"
               "$SBATCH_SCRIPT" "$ENV_NAME" "$OFFLINE_DATASET" "$SEED" "$EPISODE_LENGTH")
    echo "${cmd[@]}"
    if [[ "$DRY_RUN" != "1" ]]; then
        "${cmd[@]}"
    fi
    n_submitted=$((n_submitted + 1))
}

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
            submit "$KEY" "$ENV_NAME" "$OFFLINE_DATASET" "$EPISODE_LENGTH" "$SEED" "$LAMBDA" "$EMP_CKPT"
        done
        if [[ "$INCLUDE_BASELINE" == "1" ]]; then
            submit "$KEY" "$ENV_NAME" "$OFFLINE_DATASET" "$EPISODE_LENGTH" "$SEED" off ""
        fi
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
