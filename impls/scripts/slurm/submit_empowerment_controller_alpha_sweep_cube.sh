#!/usr/bin/env bash
# Submit the OFFLINE empowerment_skill high-level controller AWR-alpha sweep for the
# **cube-single-play** empowerment_final checkpoints to the rnn.ist.berkeley.edu Slurm
# cluster: 5 checkpoints x 4 alphas {0.3, 1, 3, 10} = 20 jobs, one sbatch job each.
#
# Companion of submit_empowerment_controller_alpha_sweep.sh (the antsoccer / pointmaze
# checkpoints) and submit_dds_controller_alpha_sweep_cube.sh (the DDS cube checkpoints);
# it reuses the same per-run body, run_empowerment_controller_alpha_seed.sbatch, so every
# flag matches those sweeps: gciql high level over the K=50 skill codebook, 1M steps,
# expectile 0.9 / chunk_horizon 10, 50 eval episodes, no --run_group.
#
# Unlike the other empowerment_final env dirs (one sd000_* run each), cube-single-play
# holds FIVE separate sd000_* runs (5 seeds from the 2026-09-04/08 batches) -- so here the
# checkpoint loop is over every sd000_* subdirectory of one env dir, not over five env dirs.
#
# Env is the OFFLINE OGBench task each checkpoint was pretrained on, read from its own
# flags.json: cube-single-play-v0 (5 tasks).
#
# Results land in <SKILL_CKPT>/controller_awr_sweep/alpha<ALPHA>/OGBench/Debug/
# sd000_s_<jobid>.<ts>/, so no pretrained params_*.pkl is ever touched.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn login
# node, from this NAS checkout:
#   bash scripts/slurm/submit_empowerment_controller_alpha_sweep_cube.sh
# Overrides:
#   DRY_RUN=1                print the sbatch commands without submitting
#   ALPHAS="0.3 1"           a different alpha grid
#   SEEDS="0 1"              more than the single seed 0
#   CKPT_DIRS="sd000_s_38624009.0.20260908_013305"   a subset of the five checkpoint dirs
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * cube-single-play-v0.npz in /nas/ucb/ishirgarg/.ogbench/data (compute nodes have no
#     internet egress). Present as of 2026-09-10.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_empowerment_controller_alpha_seed.sbatch
LOG_DIR=logs/slurm/empowerment_controller_alpha_cube
EMP_ROOT=${EMP_ROOT:-ckpts/final/empowerment_final}
CUBE_ENV_DIR=${CUBE_ENV_DIR:-cube-single-play}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
ALPHAS=${ALPHAS:-"0.3 1 3 10"}
SEEDS=${SEEDS:-"0"}

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

    # Offline training reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    ENV_NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$ENV_NAME.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$ENV_NAME.npz is missing; these jobs will fail on a compute node." >&2
    fi

    # Short tag from the trailing slurm-job-id in the checkpoint dir name, e.g.
    # sd000_s_38624009.0.20260908_013305 -> c38624009.
    TAG="c$(echo "${CKPT_DIRS[$c]}" | sed -E 's/^sd000_s_([0-9]+).*/\1/')"
    for ALPHA in $ALPHAS; do
        for SEED in $SEEDS; do
            JOB_NAME="empctrl_a${ALPHA}_${TAG}_s${SEED}"
            OUT="$LOG_DIR/${JOB_NAME}_%j.log"
            cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
                 "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ALPHA" "$SEED")
            echo "${cmd[@]}"
            if [[ "$DRY_RUN" != "1" ]]; then
                "${cmd[@]}"
            fi
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
