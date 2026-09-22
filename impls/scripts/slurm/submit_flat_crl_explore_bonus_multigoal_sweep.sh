#!/usr/bin/env bash
# Exploration-bonus sweep on the two MULTI-GOAL envs (one fixed start, many goals; see
# ogbench/manipspace/__init__.py `cube-single-multigoal-v0` (30 goals) and
# ogbench/locomaze/__init__.py `antsoccer-arena-multigoal-v0` (48 goals)). Same recipe as
# submit_flat_crl_explore_bonus_v2_sweep.sh (flat CRL + RLPD, add_explore=reward-to-rlpd,
# entropy target off, 100 eval episodes / 25k steps, 8 seeds), with two swept axes:
#   * bonus_scale in {0.3, 0.1}
#   * annealing on (BONUS_TIME_FRAC=0.5: bonus_scale -> 0 by 500k of 1M steps) vs off (constant)
# 2 cells x 2 rewards x 2 scales x 2 anneal x 8 seeds = 128 jobs, one sbatch job each, through
# run_online_crl_seed.sbatch. This script only SUBMITS; nothing runs until you invoke it.
#
# Cells (env -> RLPD dataset -> frozen estimator checkpoint, ckpts/final/empowerment_final, K=50):
#   asoc_mg      antsoccer-arena-multigoal-online-v0      antsoccer-arena-navigate-v0  antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836
#   cube_mg      cube-single-multigoal-online-v0          cube-single-play-v0          cube-single-play/sd000_s_38624008.0.20260908_013305
#   asoc_far_mg  antsoccer-arena-far-multigoal-online-v0  antsoccer-arena-navigate-v0  antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836
# asoc_far_mg is the same arena/start as asoc_mg (ant at (10, 10)) but the ball sits far away at
# (1, 1) with only 8 goals (+-1 around it), so the agent must cross the arena before it can push.
# The estimator/dataset pairs are the same ones every earlier sweep used for these two tasks:
# antsoccer uses the navigate ckpt + navigate dataset; cube has no "navigate" variant, so it uses
# its play ckpt + play dataset. (The multigoal envs share the arena/cube, observation and action
# spaces with those datasets.) Episode length: antsoccer 500 like every antsoccer online sweep,
# cube at its registered 200.
#
# NOT included: a bonus_scale=0 baseline on these envs. Submit one separately (or pass
# BASELINE=1 below) -- there are no existing baseline runs for the multigoal envs.
#
# Tags / save dirs: run_online_crl_seed.sbatch tags runs "rlpd_noent_rbrlpd[max]<scale>[_ann0.5]"
# (the constant arm has no _ann suffix). Job names: crlrbmg_<cell>_<emp|maxemp>_a<scale>_<ann|const>_s<seed>.
# Logs land in logs/slurm/flat_crl_explore_bonus_multigoal_sweep/.
#
# Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_flat_crl_explore_bonus_multigoal_sweep.sh
# Overrides:
#   DRY_RUN=1                      print the sbatch commands without submitting
#   SEEDS="0 1 2 3 4 5 6 7"        seed set (default 8 seeds)
#   SCALES="0.3 0.1"               bonus_scale set
#   ANNEAL="on off"                annealing arms (on = BONUS_TIME_FRAC, off = constant bonus)
#   BONUS_TIME_FRAC=0.5            fraction of total steps over which the bonus decays to 0 (arm "on")
#   REWARDS="empowerment max_episodic_empowerment"   explore_reward set
#   GROUP_KEYS="asoc_mg cube_mg asoc_far_mg"   subset of the three cells
#   BASELINE=1                     ALSO submit the no-bonus baseline (cells x seeds) after the sweep
#   BASELINE_ONLY=1                submit only the no-bonus baseline
#   EVAL_EPISODES=100 EVAL_INTERVAL=25000   eval cadence
#   SBATCH_TIME=08:00:00           per-job time limit
#   EMP_NUM_SPLUS_SAMPLES=64       futures per skill and state for E(s)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_seed.sbatch
LOG_DIR=logs/slurm/flat_crl_explore_bonus_multigoal_sweep
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
SCALES=${SCALES:-"0.3 0.1"}
ANNEAL=${ANNEAL:-"on off"}
REWARDS=${REWARDS:-"empowerment max_episodic_empowerment"}
BONUS_TIME_FRAC=${BONUS_TIME_FRAC:-0.5}
BASELINE=${BASELINE:-0}
BASELINE_ONLY=${BASELINE_ONLY:-0}
EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-25000}
SBATCH_TIME=${SBATCH_TIME:-08:00:00}
EMP_NUM_SPLUS_SAMPLES=${EMP_NUM_SPLUS_SAMPLES:-64}
MODE=reward-to-rlpd   # fixed, not swept

EMP_ROOT=ckpts/final/empowerment_final
ALL_GROUP_KEYS=(asoc_mg cube_mg asoc_far_mg)
ALL_GROUP_ENVS=(
    antsoccer-arena-multigoal-online-v0
    cube-single-multigoal-online-v0
    antsoccer-arena-far-multigoal-online-v0
)
ALL_GROUP_OFFLINE=(
    antsoccer-arena-navigate-v0
    cube-single-play-v0
    antsoccer-arena-navigate-v0
)
ALL_GROUP_EMP_CKPT=(
    "$EMP_ROOT/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836"
    "$EMP_ROOT/cube-single-play/sd000_s_38624008.0.20260908_013305"
    "$EMP_ROOT/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836"
)
ALL_GROUP_EPISODE_LENGTH=(500 "" 500)

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

for r in $REWARDS; do
    [[ "$r" == "empowerment" || "$r" == "max_episodic_empowerment" ]] || { echo "ERROR: REWARDS entry '$r' unknown" >&2; exit 1; }
done
for a in $ANNEAL; do
    [[ "$a" == "on" || "$a" == "off" ]] || { echo "ERROR: ANNEAL entry '$a' unknown (on | off)" >&2; exit 1; }
done

n_submitted=0
run_cmd() {  # run_cmd <cmd...>: echo, and execute unless DRY_RUN
    echo "$*"
    if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
    n_submitted=$((n_submitted + 1))
}

submit_bonus() {  # KEY ENV OFFLINE EPISODE_LENGTH SEED EMP_CKPT REWARD SCALE ANNEAL_ARM
    local KEY=$1 ENV_NAME=$2 OFFLINE_DATASET=$3 EPISODE_LENGTH=$4 SEED=$5 EMP_CKPT=$6 REWARD=$7 SCALE=$8 ARM=$9
    local reward_tag=emp; [[ "$REWARD" == "max_episodic_empowerment" ]] && reward_tag=maxemp
    local frac="" ann_tag=const
    if [[ "$ARM" == "on" ]]; then frac="$BONUS_TIME_FRAC"; ann_tag=ann; fi
    local JOB_NAME="crlrbmg_${KEY}_${reward_tag}_a${SCALE}_${ann_tag}_s${SEED}"
    local OUT="$LOG_DIR/${JOB_NAME}_%j.log"
    run_cmd env \
        EMP_CKPT_DIR="$EMP_CKPT" EMP_ENTROPY_TARGET=False EMP_NUM_SPLUS_SAMPLES="$EMP_NUM_SPLUS_SAMPLES" \
        ADD_EXPLORE="$MODE" EXPLORE_REWARD="$REWARD" BONUS_SCALE="$SCALE" BONUS_TIME_FRAC="$frac" \
        EVAL_EPISODES="$EVAL_EPISODES" EVAL_INTERVAL="$EVAL_INTERVAL" \
        sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$SBATCH_TIME" \
        "$SBATCH_SCRIPT" "$ENV_NAME" "$OFFLINE_DATASET" "$SEED" "$EPISODE_LENGTH"
}

submit_baseline() {  # KEY ENV OFFLINE EPISODE_LENGTH SEED
    local KEY=$1 ENV_NAME=$2 OFFLINE_DATASET=$3 EPISODE_LENGTH=$4 SEED=$5
    local JOB_NAME="crlrbmg_${KEY}_baseline_s${SEED}"
    local OUT="$LOG_DIR/${JOB_NAME}_%j.log"
    run_cmd env \
        EMP_CKPT_DIR= EMP_ENTROPY_TARGET= ADD_EXPLORE= BONUS_SCALE= EXPLORE_REWARD= BONUS_TIME_FRAC= \
        EVAL_EPISODES="$EVAL_EPISODES" EVAL_INTERVAL="$EVAL_INTERVAL" \
        sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$SBATCH_TIME" \
        "$SBATCH_SCRIPT" "$ENV_NAME" "$OFFLINE_DATASET" "$SEED" "$EPISODE_LENGTH"
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
    if [[ "$BASELINE_ONLY" != "1" ]]; then
        for SCALE in $SCALES; do
            for REWARD in $REWARDS; do
                for ARM in $ANNEAL; do
                    for SEED in $SEEDS; do
                        submit_bonus "$KEY" "$ENV_NAME" "$OFFLINE_DATASET" "$EPISODE_LENGTH" "$SEED" "$EMP_CKPT" "$REWARD" "$SCALE" "$ARM"
                    done
                done
            done
        done
    fi
    if [[ "$BASELINE" == "1" || "$BASELINE_ONLY" == "1" ]]; then
        for SEED in $SEEDS; do
            submit_baseline "$KEY" "$ENV_NAME" "$OFFLINE_DATASET" "$EPISODE_LENGTH" "$SEED"
        done
    fi
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
