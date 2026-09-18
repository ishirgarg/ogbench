#!/usr/bin/env bash
# Submit the SAME 144-job exploration-bonus grid as
# scripts/run_online_crl_explore_bonus_local_sweep.sh, but to the rnn.ist.berkeley.edu Slurm
# cluster instead of packed onto this machine's local GPUs (that local run was killed after
# ~5.5h with only 24/144 jobs ever started -- see logs/local_explore_bonus_sweep/). One sbatch
# job per configuration, through run_online_crl_seed.sbatch, each on its own dedicated GPU.
#
# Grid (agents/online_crl.py, "Exploration reward bonus" in its docstring):
#   BONUS_SCALE     1 0.3 0.1                          actor weight on the bonus critic Q_x(s, a)
#   ADD_EXPLORE     reward-to-rlpd | reward             Q_x backed up on the RLPD rows too, or online rows only
#   EXPLORE_REWARD  empowerment | max_episodic_empowerment
#                   r_x = E(s') - E_mean, or the running max of E over the episode through s' minus E_mean
#   SEEDS           0 1 2 3
# = 3 cells x 3 scales x 2 modes x 2 rewards x 4 seeds = 144 jobs.
#
# Differs from scripts/slurm/submit_flat_crl_explore_bonus_sweep.sh (the original 216-job
# sweep, jobs 1220980-1221195): higher bonus_scale band (1/0.3/0.1 vs that sweep's 0.001-0.3),
# 4 seeds not 3, and BONUS_TIME_FRAC is pinned to 0.5 for every job here -- the bonus linearly
# anneals to 0 by half of total_env_steps (agents/online_crl.py's explore_reward_time_frac,
# added 2026-09-16; the agent's OWN default is now "no annealing", so this script pins it
# explicitly rather than relying on that default). run_online_crl_seed.sbatch tags every run
# "..._ann0.5" for this. emp_entropy_target=False in every job, same as before (isolates the bonus).
#
# Cells (env -> RLPD dataset -> estimator checkpoint, ckpts/final/empowerment_final, K=50):
#   asoc_ctr  antsoccer-arena-center-online-v0      antsoccer-arena-navigate-v0      antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836
#   pmt_nav   pointmaze-teleport-center-online-v0   pointmaze-teleport-navigate-v0   pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836
#   cube_sgl  cube-single-center-online-v0          cube-single-play-v0              cube-single-play/sd000_s_38624008.0.20260908_013305
# Episode length: antsoccer at 500 like every antsoccer online sweep in this repo; pointmaze and
# cube at their registered horizons.
#
# Tags / save dirs: run_online_crl_seed.sbatch tags each run "rlpd_noent_rb<scale>_ann0.5"
# (reward), "rlpd_noent_rbrlpd<scale>_ann0.5" (reward-to-rlpd), with "max" inserted before the
# scale for the running-max reward. Logs land in logs/slurm/flat_crl_explore_bonus_annealed_sweep/.
#
# This ONLY submits jobs. Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_flat_crl_explore_bonus_annealed_sweep.sh
# Overrides:
#   DRY_RUN=1                     print the sbatch commands without submitting
#   SEEDS="0 1 2 3"                seed set (default "0 1 2 3")
#   SCALES="0.3 0.1"               bonus_scale set (default "1 0.3 0.1")
#   MODES="reward-to-rlpd"         add_explore set (default "reward-to-rlpd reward")
#   REWARDS="empowerment"          explore_reward set (default "empowerment max_episodic_empowerment")
#   GROUP_KEYS="pmt_nav cube_sgl"  subset of the three cells
#   BONUS_TIME_FRAC=0              disable annealing for this submission (default 0.5)
#   EMP_NUM_SPLUS_SAMPLES=64       futures per skill and state for E(s) (default 64)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_seed.sbatch
LOG_DIR=logs/slurm/flat_crl_explore_bonus_annealed_sweep
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3"}
SCALES=${SCALES:-"1 0.3 0.1"}
MODES=${MODES:-"reward-to-rlpd reward"}
REWARDS=${REWARDS:-"empowerment max_episodic_empowerment"}
BONUS_TIME_FRAC=${BONUS_TIME_FRAC:-0.5}
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
submit() {  # submit KEY ENV OFFLINE EPISODE_LENGTH SEED EMP_CKPT MODE REWARD SCALE
    local KEY=$1 ENV_NAME=$2 OFFLINE_DATASET=$3 EPISODE_LENGTH=$4 SEED=$5 EMP_CKPT=$6 MODE=$7 REWARD=$8 SCALE=$9
    local mode_tag=rlpd; [[ "$MODE" == "reward" ]] && mode_tag=onl
    local reward_tag=emp; [[ "$REWARD" == "max_episodic_empowerment" ]] && reward_tag=maxemp
    local JOB_NAME="crlrbann_${KEY}_${mode_tag}_${reward_tag}_a${SCALE}_s${SEED}"
    # sbatch exports the caller's environment (--export=ALL default), so these reach
    # run_online_crl_seed.sbatch, which turns them into --agent.* flags and the run tag.
    local env_vars=(
        EMP_CKPT_DIR="$EMP_CKPT"
        EMP_ENTROPY_TARGET=False
        EMP_NUM_SPLUS_SAMPLES="$EMP_NUM_SPLUS_SAMPLES"
        ADD_EXPLORE="$MODE"
        EXPLORE_REWARD="$REWARD"
        BONUS_SCALE="$SCALE"
        BONUS_TIME_FRAC="$BONUS_TIME_FRAC"
    )
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
    for SCALE in $SCALES; do
        for REWARD in $REWARDS; do
            for MODE in $MODES; do
                for SEED in $SEEDS; do
                    submit "$KEY" "$ENV_NAME" "$OFFLINE_DATASET" "$EPISODE_LENGTH" "$SEED" "$EMP_CKPT" "$MODE" "$REWARD" "$SCALE"
                done
            done
        done
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
