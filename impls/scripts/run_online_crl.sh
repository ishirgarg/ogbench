#!/usr/bin/env bash
# Online CRL baseline (flat goal-conditioned agent, no skill controller) on an
# OGBench env, learning from its own rollouts. Mirrors JaxGCRL's flat `crl`
# agent; see agents/online_crl.py and main_online.py.
#
# Launches one run per entry of ENVS on the matching GPU. Task pairs come from
# the env registration, so a custom online task set is just another env name.
#
# RLPD is on by default: every batch mixes rows from the OGBench dataset named
# by OFFLINE_DATASET (a single name for all ENVS, or per-env via the map below;
# set OFFLINE_DATASET=none to train from online data only).
#
# Empowerment entropy target (agents/online_crl.py, `emp_*` config): set
# EMP_CKPT_DIR to a frozen empowerment_skill run dir, or EMP_CKPT_DIR=auto for the
# per-env default in EMP_DEFAULTS below. EMP_LAMBDA (default 1.0), EMP_NUM_BINS (8)
# and EMP_NUM_SPLUS_SAMPLES (64) tune it; the tag gets an "_emp<lambda>" suffix.
# EMP_ENTROPY_TARGET=False keeps the estimator but drops the per-state target (scalar
# alpha; tag "_noent"). Exploration reward bonus (same file, `add_explore`): set
# ADD_EXPLORE=reward (Q_x fit on online rows) or reward-to-rlpd (RLPD rows too), with
# BONUS_SCALE (default 1.0) the actor weight on Q_x; needs an estimator (EMP_CKPT_DIR).
# EXPLORE_REWARD picks the reward: empowerment (default, E(s')) or max_episodic_empowerment
# (running max of E over the episode). BONUS_TIME_FRAC (agent default: unset, no annealing) linearly decays
# bonus_scale to 0 by that fraction of TOTAL_ENV_STEPS, held at 0 after; 0 disables the bonus
# from the start, 1.0 decays over the whole run. Tag suffix "_rb[max]<scale>[_ann<frac>]".
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=${MUJOCO_GL:-egl}

PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

read -r -a ENVS <<< "${ENVS:-antmaze-medium-navigate-v0}"
read -r -a GPUS <<< "${GPUS:-0}"
SEED=${SEED:-0}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
EPISODE_LENGTH=${EPISODE_LENGTH:-}   # empty -> per-env default below (antsoccer: 500, else env's registered horizon)
OFFLINE_DATASET=${OFFLINE_DATASET:-}  # empty -> per-env default below; "none" -> no RLPD

# Default offline dataset per online env (the env family's OGBench dataset).
declare -A OFFLINE_DEFAULTS=(
    [antmaze-medium-center-online-v0]=antmaze-medium-navigate-v0
    [antsoccer-arena-center-online-v0]=antsoccer-arena-navigate-v0
)
offline_dataset_for() {
    if [[ "$OFFLINE_DATASET" == "none" ]]; then echo ""; return; fi
    if [[ -n "$OFFLINE_DATASET" ]]; then echo "$OFFLINE_DATASET"; return; fi
    echo "${OFFLINE_DEFAULTS[$1]:-}"
}

# Empowerment estimator per online env (ckpts/final/empowerment_final, the 50-skill runs).
EMP_CKPT_DIR=${EMP_CKPT_DIR:-}   # empty -> off; "auto" -> EMP_DEFAULTS; else a run dir
EMP_LAMBDA=${EMP_LAMBDA:-1.0}
EMP_NUM_BINS=${EMP_NUM_BINS:-8}
EMP_NUM_SPLUS_SAMPLES=${EMP_NUM_SPLUS_SAMPLES:-64}
EMP_ENTROPY_TARGET=${EMP_ENTROPY_TARGET:-}   # empty -> agent default (True); "False" -> scalar alpha
ADD_EXPLORE=${ADD_EXPLORE:-}   # empty -> off; "reward" | "reward-to-rlpd"
BONUS_SCALE=${BONUS_SCALE:-1.0}
EXPLORE_REWARD=${EXPLORE_REWARD:-empowerment}   # "empowerment" | "max_episodic_empowerment" (tag "max")
BONUS_TIME_FRAC=${BONUS_TIME_FRAC:-}   # empty -> agent default (no annealing); else fraction of total_env_steps at
                                        # which bonus_scale has linearly decayed to 0 (tag "_ann<frac>")
declare -A EMP_DEFAULTS=(
    [antmaze-medium-center-online-v0]=ckpts/final/empowerment_final/antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001
    [antsoccer-arena-center-online-v0]=ckpts/final/empowerment_final/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836
    [pointmaze-teleport-center-online-v0]=ckpts/final/empowerment_final/pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836
)
emp_ckpt_for() {
    if [[ -z "$EMP_CKPT_DIR" ]]; then echo ""; return; fi
    if [[ "$EMP_CKPT_DIR" != "auto" ]]; then echo "$EMP_CKPT_DIR"; return; fi
    local d="${EMP_DEFAULTS[$1]:-}"
    if [[ -z "$d" ]]; then echo "no EMP_DEFAULTS entry for $1; pass EMP_CKPT_DIR=<run dir>" >&2; exit 1; fi
    echo "$d"
}

# Default episode horizon per online env (antsoccer is shorter than the antmaze's registered 1000).
declare -A EPISODE_LENGTH_DEFAULTS=(
    [antsoccer-arena-center-online-v0]=500
)
episode_length_for() {
    if [[ -n "$EPISODE_LENGTH" ]]; then echo "$EPISODE_LENGTH"; return; fi
    echo "${EPISODE_LENGTH_DEFAULTS[$1]:-}"
}

LOG_DIR=logs/online_crl
mkdir -p "$LOG_DIR"

pids=()
for i in "${!ENVS[@]}"; do
    ENV_NAME=${ENVS[$i]}
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    EP_LEN=$(episode_length_for "$ENV_NAME")
    EP_FLAG=()
    if [[ -n "$EP_LEN" ]]; then EP_FLAG=(--episode_length="$EP_LEN"); fi
    OFFLINE=$(offline_dataset_for "$ENV_NAME")
    RLPD_FLAG=()
    TAG=norlpd
    if [[ -n "$OFFLINE" ]]; then RLPD_FLAG=(--offline_dataset="$OFFLINE"); TAG=rlpd; fi
    # RLPD_FRAC_TIME (env var, optional): --rlpd_frac_time, the fraction of total_env_steps during
    # which offline data is mixed in; batches are online-only afterwards (main_online.py, default 1.0).
    # Only meaningful with RLPD on; the tag gets a "_ft<frac>" suffix so save dirs / logs do not collide.
    if [[ -n "${RLPD_FRAC_TIME:-}" && ${#RLPD_FLAG[@]} -gt 0 ]]; then
        RLPD_FLAG+=(--rlpd_frac_time="$RLPD_FRAC_TIME")
        TAG="${TAG}_ft${RLPD_FRAC_TIME}"
    fi
    EMP_CKPT=$(emp_ckpt_for "$ENV_NAME")
    EMP_FLAG=()
    if [[ -n "$EMP_CKPT" ]]; then
        EMP_FLAG=(
            --agent.emp_checkpoint_path="$EMP_CKPT"
            --agent.emp_lambda="$EMP_LAMBDA"
            --agent.emp_num_bins="$EMP_NUM_BINS"
            --agent.emp_num_splus_samples="$EMP_NUM_SPLUS_SAMPLES"
        )
        if [[ "$EMP_ENTROPY_TARGET" == "False" || "$EMP_ENTROPY_TARGET" == "false" || "$EMP_ENTROPY_TARGET" == "0" ]]; then
            EMP_FLAG+=(--agent.emp_entropy_target=False)
            TAG="${TAG}_noent"
        else
            TAG="${TAG}_emp${EMP_LAMBDA}"
        fi
    fi
    if [[ -n "$ADD_EXPLORE" ]]; then
        if [[ -z "$EMP_CKPT" ]]; then echo "ADD_EXPLORE=$ADD_EXPLORE needs EMP_CKPT_DIR (the empowerment reward's estimator)" >&2; exit 1; fi
        EMP_FLAG+=(--agent.add_explore="$ADD_EXPLORE" --agent.bonus_scale="$BONUS_SCALE" --agent.explore_reward="$EXPLORE_REWARD")
        if [[ -n "$BONUS_TIME_FRAC" ]]; then
            EMP_FLAG+=(--agent.explore_reward_time_frac="$BONUS_TIME_FRAC")
            TAG="${TAG}_ann${BONUS_TIME_FRAC}"
        fi
        RB_KIND=""
        if [[ "$EXPLORE_REWARD" == "max_episodic_empowerment" ]]; then RB_KIND=max; fi
        case "$ADD_EXPLORE" in
            reward) TAG="${TAG}_rb${RB_KIND}${BONUS_SCALE}" ;;
            reward-to-rlpd) TAG="${TAG}_rbrlpd${RB_KIND}${BONUS_SCALE}" ;;
            *) echo "unknown ADD_EXPLORE=$ADD_EXPLORE (reward | reward-to-rlpd)" >&2; exit 1 ;;
        esac
    fi
    LOG="$LOG_DIR/${ENV_NAME}_${TAG}_s${SEED}.log"
    echo "launching online_crl env=${ENV_NAME} offline=${OFFLINE:-none} emp=${EMP_CKPT:-off} add_explore=${ADD_EXPLORE:-off} on GPU ${GPU} -> ${LOG}"
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main_online.py \
        --env_name="$ENV_NAME" \
        --seed="$SEED" \
        --agent=agents/online_crl.py \
        --total_env_steps="$TOTAL_ENV_STEPS" \
        "${EP_FLAG[@]}" \
        "${RLPD_FLAG[@]}" \
        "${EMP_FLAG[@]}" \
        --log_interval=5000 \
        --eval_interval=20000 \
        --save_interval=100000 \
        --eval_episodes=20 \
        --video_episodes=0 \
        > "$LOG" 2>&1 &
    pids+=($!)
    # Stagger: exp_name has 1-second resolution, so same-second launches would share a run dir.
    sleep 5
done

echo "waiting on ${#pids[@]} jobs..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit $fail
