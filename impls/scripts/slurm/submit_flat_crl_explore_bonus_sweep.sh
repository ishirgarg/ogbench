#!/usr/bin/env bash
# Submit the exploration-REWARD-BONUS sweep for flat online CRL + RLPD to the
# rnn.ist.berkeley.edu Slurm cluster: 3 cells x 6 bonus scales x 2 RLPD modes x 2 rewards
# x 3 seeds = 216 jobs, one sbatch job per configuration, all through
# run_online_crl_seed.sbatch (1M env steps, the usual intervals).
#
# What is swept (agents/online_crl.py, "Exploration reward bonus" in its docstring):
#   BONUS_SCALE     0.001 0.003 0.01 0.03 0.1 0.3   actor weight on the bonus critic Q_x(s, a)
#   ADD_EXPLORE     reward-to-rlpd | reward          Q_x backed up on the RLPD rows too, or online rows only
#   EXPLORE_REWARD  empowerment | max_episodic_empowerment
#                   r_x = E(s') - E_mean, or the running max of E over the episode through s' minus E_mean
# The empowerment entropy target is OFF in every job (EMP_ENTROPY_TARGET=False: scalar alpha,
# constant target entropy), so the only difference from plain flat CRL + RLPD is the bonus.
# E(s) comes from the frozen empowerment_skill estimator matched to each cell's RLPD dataset,
# at EMP_NUM_SPLUS_SAMPLES=64 futures per skill and state.
#
# Cells (env -> RLPD dataset -> estimator checkpoint, ckpts/final/empowerment_final, K=50):
#   asoc_ctr  antsoccer-arena-center-online-v0      antsoccer-arena-navigate-v0      antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836
#   pmt_nav   pointmaze-teleport-center-online-v0   pointmaze-teleport-navigate-v0   pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836
#   cube_sgl  cube-single-center-online-v0          cube-single-play-v0              cube-single-play/sd000_s_38624008.0.20260908_013305
# Episode length: antsoccer at 500 like every antsoccer online sweep in this repo; pointmaze and
# cube at their registered horizons (1000 / 200).
#
# Baselines: plain flat CRL + RLPD on the same (env, dataset) cell. 5-seed runs already exist
# on disk for asoc_ctr and cube_sgl (exp/OGBench/Debug/, seeds 0-4); INCLUDE_BASELINE=1 adds a
# fresh bonus-off run per (cell, seed) to this submission (9 more jobs).
#
# Tags / save dirs: run_online_crl_seed.sbatch tags each run "rlpd_noent_rb<scale>" (reward),
# "rlpd_noent_rbrlpd<scale>" (reward-to-rlpd), with "max" inserted before the scale for the
# running-max reward, e.g. rlpd_noent_rbrlpdmax0.03. Logs land in logs/slurm/flat_crl_explore_bonus_sweep/.
#
# This ONLY submits jobs. Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_flat_crl_explore_bonus_sweep.sh
# Overrides:
#   DRY_RUN=1                     print the sbatch commands without submitting
#   SEEDS="0 1 2"                 seed set (default "0 1 2")
#   SCALES="0.01 0.1"             bonus_scale set (default "0.001 0.003 0.01 0.03 0.1 0.3")
#   MODES="reward-to-rlpd"        add_explore set (default "reward-to-rlpd reward")
#   REWARDS="empowerment"         explore_reward set (default "empowerment max_episodic_empowerment")
#   GROUP_KEYS="pmt_nav cube_sgl" subset of the three cells
#   INCLUDE_BASELINE=1            also submit a bonus-off (plain flat CRL + RLPD) run per (cell, seed)
#   EMP_NUM_SPLUS_SAMPLES=64      futures per skill and state for E(s) (default 64)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_seed.sbatch
LOG_DIR=logs/slurm/flat_crl_explore_bonus_sweep
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2"}
SCALES=${SCALES:-"0.001 0.003 0.01 0.03 0.1 0.3"}
MODES=${MODES:-"reward-to-rlpd reward"}
REWARDS=${REWARDS:-"empowerment max_episodic_empowerment"}
INCLUDE_BASELINE=${INCLUDE_BASELINE:-0}
EMP_NUM_SPLUS_SAMPLES=${EMP_NUM_SPLUS_SAMPLES:-64}

EMP_ROOT=ckpts/final/empowerment_final
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
ALL_GROUP_EMP_CKPT=(
    "$EMP_ROOT/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836"
    "$EMP_ROOT/pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836"
    "$EMP_ROOT/cube-single-play/sd000_s_38624008.0.20260908_013305"
)
ALL_GROUP_EPISODE_LENGTH=(500 "" "")

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

for m in $MODES; do
    [[ "$m" == "reward" || "$m" == "reward-to-rlpd" ]] || { echo "ERROR: MODES entry '$m' must be reward or reward-to-rlpd" >&2; exit 1; }
done
for r in $REWARDS; do
    [[ "$r" == "empowerment" || "$r" == "max_episodic_empowerment" ]] || { echo "ERROR: REWARDS entry '$r' unknown" >&2; exit 1; }
done

n_submitted=0
submit() {  # submit KEY ENV OFFLINE EPISODE_LENGTH SEED EMP_CKPT MODE|off REWARD SCALE
    local KEY=$1 ENV_NAME=$2 OFFLINE_DATASET=$3 EPISODE_LENGTH=$4 SEED=$5 EMP_CKPT=$6 MODE=$7 REWARD=$8 SCALE=$9
    local JOB_NAME env_vars
    # sbatch exports the caller's environment (--export=ALL default), so these reach
    # run_online_crl_seed.sbatch, which turns them into --agent.* flags and the run tag.
    if [[ "$MODE" == "off" ]]; then
        JOB_NAME="crlrb_${KEY}_off_s${SEED}"
        env_vars=(EMP_CKPT_DIR= EMP_ENTROPY_TARGET= ADD_EXPLORE= BONUS_SCALE= EXPLORE_REWARD=)
    else
        local mode_tag=rlpd; [[ "$MODE" == "reward" ]] && mode_tag=onl
        local reward_tag=emp; [[ "$REWARD" == "max_episodic_empowerment" ]] && reward_tag=maxemp
        JOB_NAME="crlrb_${KEY}_${mode_tag}_${reward_tag}_a${SCALE}_s${SEED}"
        env_vars=(
            EMP_CKPT_DIR="$EMP_CKPT"
            EMP_ENTROPY_TARGET=False
            EMP_NUM_SPLUS_SAMPLES="$EMP_NUM_SPLUS_SAMPLES"
            ADD_EXPLORE="$MODE"
            EXPLORE_REWARD="$REWARD"
            BONUS_SCALE="$SCALE"
        )
    fi
    local OUT="$LOG_DIR/${JOB_NAME}_%j.log"
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
        for MODE in $MODES; do
            for REWARD in $REWARDS; do
                for SCALE in $SCALES; do
                    submit "$KEY" "$ENV_NAME" "$OFFLINE_DATASET" "$EPISODE_LENGTH" "$SEED" "$EMP_CKPT" "$MODE" "$REWARD" "$SCALE"
                done
            done
        done
        if [[ "$INCLUDE_BASELINE" == "1" ]]; then
            submit "$KEY" "$ENV_NAME" "$OFFLINE_DATASET" "$EPISODE_LENGTH" "$SEED" "" off "" ""
        fi
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
