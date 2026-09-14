#!/usr/bin/env bash
# Submit the ONLINE CRL skill controller over the five **cube-single-play**
# empowerment_final checkpoints to the rnn.ist.berkeley.edu Slurm cluster: 5 seeds (0-4)
# per checkpoint, one sbatch job per (checkpoint, seed) -> 25 jobs.
#
# Companion of submit_dds_online_controller_cube_seeds.sh (the DDS cube checkpoints) and
# submit_empowerment_online_controller_seeds.sh (the antsoccer / pointmaze empowerment
# checkpoints); it reuses the very same per-run body,
# run_online_crl_skill_controller_seed.sbatch, so all online-controller runs share one set
# of flags: RLPD on with the checkpoint's OWN offline dataset, K=10,
# target_entropy_frac=0.9, 1M env steps, eval every 20k. Seeds are directly
# comparable to the DDS cube-single online sweep so the two skill families can be compared.
#
# Unlike the other empowerment_final env dirs (one sd000_* run each), cube-single-play
# holds FIVE separate sd000_* runs (5 seeds from the 2026-09-04/08 batches) -- so here the
# checkpoint loop is over every sd000_* subdirectory of one env dir, not over several env
# dirs with one checkpoint each. See submit_empowerment_controller_alpha_sweep_cube.sh for
# the same pattern applied to the offline alpha sweep.
#
# Online env (the deterministic, noise-free `*-center-` task set; see
# ogbench/manipspace/__init__.py):
#   cube-single-play -> cube-single-center-online-v0   (horizon 200, one pick-and-place)
# 200 is divisible by K=10, which main_online.py requires for macro rollouts, and K=10 is
# also each checkpoint's `sequence_length`, which online_crl_skill_controller.py's window
# labeller insists on. Episode length is left at the registered horizon, so no override.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn login
# node, from this NAS checkout:
#   bash scripts/slurm/submit_empowerment_online_controller_cube_seeds.sh
# Overrides:
#   DRY_RUN=1                print the sbatch commands without submitting
#   SEEDS="0 1"              a different seed set
#   CKPT_DIRS="sd000_s_38624009.0.20260908_013305"   a subset of the five checkpoint dirs
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * cube-single-play-v0.npz in /nas/ucb/ishirgarg/.ogbench/data (compute nodes have no
#     internet egress). Present as of 2026-09-10.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_crl_skill_controller_seed.sbatch
LOG_DIR=logs/slurm/empowerment_online_controller_cube
EMP_ROOT=${EMP_ROOT:-ckpts/final/empowerment_final}
CUBE_ENV_DIR=${CUBE_ENV_DIR:-cube-single-play}
ONLINE_ENV=cube-single-center-online-v0
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}

ALL_MATCHES=("$EMP_ROOT/$CUBE_ENV_DIR"/sd000_*/)
(( ${#ALL_MATCHES[@]} > 0 )) && [[ -d "${ALL_MATCHES[0]}" ]] || {
    echo "ERROR: no sd000_* runs found under $EMP_ROOT/$CUBE_ENV_DIR" >&2; exit 1
}
ALL_CKPT_DIRS=()
for m in "${ALL_MATCHES[@]}"; do ALL_CKPT_DIRS+=("$(basename "${m%/}")"); done

read -r -a WANTED <<< "${CKPT_DIRS:-${ALL_CKPT_DIRS[*]}}"
CKPT_DIRS=()
for w in "${WANTED[@]}"; do
    found=0
    for d in "${ALL_CKPT_DIRS[@]}"; do
        if [[ "$d" == "$w" ]]; then CKPT_DIRS+=("$d"); found=1; break; fi
    done
    (( found )) || { echo "ERROR: unknown checkpoint dir '$w' (known: ${ALL_CKPT_DIRS[*]})" >&2; exit 1; }
done

n_submitted=0
for c in "${!CKPT_DIRS[@]}"; do
    SKILL_CKPT="$EMP_ROOT/$CUBE_ENV_DIR/${CKPT_DIRS[$c]}"
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    # RLPD reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    OFFLINE_DATASET=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi

    # Short tag from the trailing slurm-job-id in the checkpoint dir name, e.g.
    # sd000_s_38624009.0.20260908_013305 -> c38624009.
    TAG="c$(echo "${CKPT_DIRS[$c]}" | sed -E 's/^sd000_s_([0-9]+).*/\1/')"
    for SEED in $SEEDS; do
        JOB_NAME="empctrl_cube_${TAG}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        # The cube offline dataset is ~300 MB and the DDS-style window-labelling pass over
        # it is the memory peak, so raise the sbatch script's 16 GB default and keep the
        # 16 GB A4000 nodes out (same reasoning as submit_dds_online_controller_cube_seeds.sh).
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
             --mem=32gb --exclude=ppo.ist.berkeley.edu,vae.ist.berkeley.edu
             "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ONLINE_ENV" "$SEED" "" "")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
