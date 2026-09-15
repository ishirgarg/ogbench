#!/usr/bin/env bash
# Submit the OFFLINE composed-policy low-level learning-rate sweep to the
# rnn.ist.berkeley.edu Slurm cluster.
#
# agents/composed_skill_policy.py drops the skill horizon and the hard freeze: pi_hi(k|s,g)
# and pi_lo(a|s,z_k) become one flat policy
#     pi(a|s,g) = sum_k pi_hi(k|s,g) pi_lo(a|s,z_k)
# improved by offline RL over PRIMITIVE actions, with the low level on its own learning rate.
#
# The sweep IS the experiment: LOW_LR=0 is not a filler cell, it reproduces the frozen
# low level exactly (Adam at lr 0 is a no-op) and is therefore the in-run control that every
# other cell is compared against. Everything else -- critic, high level, batch, goal
# sampling, seed -- is identical across the cells, which the existing *_controller runs
# cannot claim since they differ in agent as well as in horizon.
#
# Default grid: 3 empowerment_final checkpoints x {0, 1e-5, 3e-5, 1e-4} -> 12 jobs.
# Results land in
#   <SKILL_CKPT>/composed_skill_policy/lr<LOW_LR>/OGBench/Debug/sd<seed>_s_<jobid>.<ts>/
# so nothing under <SKILL_CKPT> is modified. See run_composed_skill_policy_seed.sbatch.
#
# This ONLY submits jobs. Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_composed_skill_policy_sweep.sh
# Overrides:
#   DRY_RUN=1                       print the sbatch commands without submitting
#   LOW_LRS="0 1e-4"                a different low-level lr grid
#   SEEDS="0 1"                     more than the single seed 0
#   SKILL_ROOT=ckpts/final/opal     sweep the opal (latent_type=discrete) checkpoints instead
#   CKPT_DIRS="antmaze-medium-navigate ..."   a subset of the env dirs
#   COMPOSED_ACTOR_LOSS=ddpgbc      the DDPG+BC branch instead of AWR
#   ALPHA=1.0                       AWR temperature / DDPG+BC BC coefficient
#   COMPOSED_BATCH_SIZE=256         subsample the composed actor loss (cost is K x per row)
#
# Prerequisites: wandb credentials (see run_online_crl_seed.sbatch) and each checkpoint's
# offline dataset already in /nas/ucb/ishirgarg/.ogbench/data (compute nodes have no egress).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_composed_skill_policy_seed.sbatch
LOG_DIR=logs/slurm/composed_skill_policy
SKILL_ROOT=${SKILL_ROOT:-ckpts/final/empowerment_final}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
LOW_LRS=${LOW_LRS:-"0 1e-5 3e-5 1e-4"}
SEEDS=${SEEDS:-"0"}
COMPOSED_ACTOR_LOSS=${COMPOSED_ACTOR_LOSS:-awr}
ALPHA=${ALPHA:-3.0}

# Parallel arrays: checkpoint env dir -> short job-name tag.
ALL_CKPT_DIRS=(
    antmaze-medium-navigate
    antsoccer-arena-navigate
    pointmaze-teleport-navigate
)
ALL_CKPT_TAGS=(amaze_nav asoc_nav pmt_nav)

read -r -a WANTED <<< "${CKPT_DIRS:-${ALL_CKPT_DIRS[*]}}"
CKPT_DIRS=(); CKPT_TAGS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_CKPT_DIRS[@]}"; do
        if [[ "${ALL_CKPT_DIRS[$i]}" == "$w" ]]; then
            CKPT_DIRS+=("${ALL_CKPT_DIRS[$i]}"); CKPT_TAGS+=("${ALL_CKPT_TAGS[$i]}"); found=1; break
        fi
    done
    (( found )) || { echo "ERROR: unknown checkpoint dir '$w' (known: ${ALL_CKPT_DIRS[*]})" >&2; exit 1; }
done

n_submitted=0
for c in "${!CKPT_DIRS[@]}"; do
    # antmaze-medium-{navigate,stitch} hold TWO empowerment runs (k=50 and k=15), so take the
    # newest match rather than insisting on exactly one, and print which one was chosen.
    mapfile -t matches < <(ls -d "$SKILL_ROOT/${CKPT_DIRS[$c]}"/sd000_*/ 2>/dev/null | sort)
    if (( ${#matches[@]} == 0 )); then
        echo "ERROR: no sd000_* run under $SKILL_ROOT/${CKPT_DIRS[$c]}" >&2; exit 1
    fi
    SKILL_CKPT=${matches[-1]%/}
    (( ${#matches[@]} > 1 )) && echo "NOTE: ${#matches[@]} runs under ${CKPT_DIRS[$c]}; using $(basename "$SKILL_CKPT")" >&2
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    ENV_NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$ENV_NAME.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$ENV_NAME.npz is missing; these jobs will fail on a compute node." >&2
    fi

    TAG=${CKPT_TAGS[$c]}
    for LOW_LR in $LOW_LRS; do
        for SEED in $SEEDS; do
            JOB_NAME="composed_lr${LOW_LR}_${TAG}_s${SEED}"
            OUT="$LOG_DIR/${JOB_NAME}_%j.log"
            cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
                 --export="ALL,LOW_LR=$LOW_LR,ALPHA=$ALPHA,COMPOSED_ACTOR_LOSS=$COMPOSED_ACTOR_LOSS,SAVE_SUBDIR=composed_skill_policy/lr${LOW_LR},COMPOSED_BATCH_SIZE=${COMPOSED_BATCH_SIZE:-},DIFFUSION_LOGPROB_SCALE=${DIFFUSION_LOGPROB_SCALE:-1.0}"
                 "$SBATCH_SCRIPT" "$SKILL_CKPT" "$SEED")
            echo "${cmd[@]}"
            if [[ "$DRY_RUN" != "1" ]]; then
                "${cmd[@]}"
            fi
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
