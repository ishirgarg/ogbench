#!/usr/bin/env bash
# Submit online_crl_skill_controller (high-level skill controller) multi-seed
# RLPD sweeps to the rnn.ist.berkeley.edu Slurm cluster, one sbatch job per
# (skill checkpoint, seed).
#
# Goal: 5 identical seeds (0-4) per checkpoint, for confidence intervals. Seed 0
# already finished locally for both checkpoints (see
# ckpts/empowerment_final/antmaze-medium-{navigate,stitch}/.../online_controller/rlpd/),
# using the entropy-clamp fix verified 2026-09-02 (target_entropy_multiplier=0.5,
# clamped to <= target_entropy_cap_frac * log(num_skills)). This script submits
# only what's missing: seeds 1-4 for each of the two checkpoints (8 jobs total),
# on antmaze-medium-center-online-v0 with RLPD on (offline dataset = each
# checkpoint's own dataset -- antmaze-medium-navigate-v0 / antmaze-medium-stitch-v0).
#
# This ONLY submits jobs -- it does not run any training itself. Run from the
# rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_online_crl_skill_controller_rlpd_seeds.sh
# Add DRY_RUN=1 to print the sbatch commands without submitting.
#
# Before running for real, make sure wandb credentials are set up -- see the
# comment block in run_online_crl_skill_controller_seed.sbatch.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_skill_controller_seed.sbatch
LOG_DIR=logs/slurm/online_crl_skill_controller
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}

ENV_NAME=antmaze-medium-center-online-v0
EMP_ROOT=ckpts/empowerment_final
CKPT_NAV="$EMP_ROOT/antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001"
CKPT_STI="$EMP_ROOT/antmaze-medium-stitch/sd000_s_37866313.0.20260821_030454_k50_s0.01_bc0.001"

CKPTS=("$CKPT_NAV" "$CKPT_STI")
CKPT_TAGS=(navigate stitch)
SEEDS="1 2 3 4"

n_submitted=0
for c in "${!CKPTS[@]}"; do
    SKILL_CKPT=${CKPTS[$c]}
    TAG=${CKPT_TAGS[$c]}
    if [[ ! -f "$SKILL_CKPT/flags.json" ]]; then
        echo "ERROR: missing $SKILL_CKPT/flags.json" >&2
        exit 1
    fi
    for SEED in $SEEDS; do
        JOB_NAME="ctrl_${TAG}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
             "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ENV_NAME" "$SEED")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
