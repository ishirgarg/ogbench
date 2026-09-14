#!/usr/bin/env bash
# Submit the OFFLINE DDS high-level controller AWR-alpha sweep for the two **cube**
# checkpoints in ckpts/final/dds to the rnn.ist.berkeley.edu Slurm cluster:
# 2 checkpoints x 4 alphas {0.3, 1, 3, 10} = 8 jobs, one sbatch job each.
#
# Companion of submit_dds_controller_alpha_sweep.sh (the antmaze / antsoccer /
# pointmaze checkpoints); it reuses the same per-run body,
# run_dds_controller_alpha_seed.sbatch, so every flag other than --agent.base.alpha
# matches the DDS paper stage: gciql high level over the K=50 codebook, 1M steps,
# expectile 0.7 / lr 1e-4 / tau 0.005, chunk_horizon 10, 50 eval episodes, no --run_group.
#
# Envs are the OFFLINE OGBench cube tasks each checkpoint was pretrained on, read from
# its flags.json: cube-single-play-v0 (5 tasks) and cube-double-play-v0 (5 tasks).
# Both checkpoints have sequence_length 10, which dds_controller.create asserts equals
# the default chunk_horizon, so no override is needed.
#
# alpha=3.0 IS included here, unlike the other sweep: it is dds_controller.py's paper
# default, and the other five checkpoints already had it trained locally under
# <SKILL_CKPT>/controller/, but the two cube checkpoints have no controller/ run yet,
# so all four grid points are new. The grid {0.3, 1, 3, 10} is the one from the
# empowerment_skill controller sweep (scripts/run_skill_bc_relabel_awr_sweep_antsoccer.sh).
#
# Layout note: the two cube runs are stored FLAT (flags.json / params_1000000.pkl
# directly under the env dir, no sd000_* subdirectory). The resolver below accepts either.
#
# Results land in <SKILL_CKPT>/controller_awr_sweep/alpha<ALPHA>/OGBench/Debug/
# sd000_s_<jobid>.<ts>/, so the pretrained params_*.pkl is never touched.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn login
# node, from this NAS checkout:
#   bash scripts/slurm/submit_dds_controller_alpha_sweep_cube.sh
# Overrides:
#   DRY_RUN=1                     print the sbatch commands without submitting
#   ALPHAS="0.3 1"                a different alpha grid
#   SEEDS="0 1"                   more than the single seed 0
#   CKPT_DIRS="cube-double-play"  a subset of the two env dirs
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * cube-{single,double}-play-v0.npz in /nas/ucb/ishirgarg/.ogbench/data (compute nodes
#     have no internet egress). Both were present as of 2026-09-10.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_dds_controller_alpha_seed.sbatch
LOG_DIR=logs/slurm/dds_controller_alpha
DDS_ROOT=${DDS_ROOT:-ckpts/final/dds}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
ALPHAS=${ALPHAS:-"0.3 1 3 10"}
SEEDS=${SEEDS:-"0"}

# Parallel arrays: checkpoint env dir -> short job-name tag.
ALL_CKPT_DIRS=(cube-single-play cube-double-play)
ALL_CKPT_TAGS=(cube_sgl cube_dbl)

read -r -a WANTED <<< "${CKPT_DIRS:-${ALL_CKPT_DIRS[*]}}"
CKPT_DIRS=(); CKPT_TAGS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_CKPT_DIRS[@]}"; do
        if [[ "${ALL_CKPT_DIRS[$i]}" == "$w" ]]; then
            CKPT_DIRS+=("${ALL_CKPT_DIRS[$i]}")
            CKPT_TAGS+=("${ALL_CKPT_TAGS[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown checkpoint dir '$w' (known: ${ALL_CKPT_DIRS[*]})" >&2; exit 1; }
done

# Accept both the flat layout (cube) and the sd000_* layout (the other five).
resolve_ckpt() {
    local root_dir=$1
    if [[ -f "$root_dir/flags.json" ]]; then
        echo "$root_dir"
        return
    fi
    local matches=("$root_dir"/sd000_*/)
    if (( ${#matches[@]} != 1 )) || [[ ! -d "${matches[0]}" ]]; then
        echo "ERROR: no flags.json in $root_dir and not exactly one sd000_* run under it" >&2
        exit 1
    fi
    echo "${matches[0]%/}"
}

n_submitted=0
for c in "${!CKPT_DIRS[@]}"; do
    SKILL_CKPT=$(resolve_ckpt "$DDS_ROOT/${CKPT_DIRS[$c]}")
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    # Offline training reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    ENV_NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$ENV_NAME.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$ENV_NAME.npz is missing; these jobs will fail on a compute node." >&2
    fi

    TAG=${CKPT_TAGS[$c]}
    for ALPHA in $ALPHAS; do
        for SEED in $SEEDS; do
            JOB_NAME="ddsctrl_a${ALPHA}_${TAG}_s${SEED}"
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
