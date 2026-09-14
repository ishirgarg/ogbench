#!/usr/bin/env bash
# Submit the ONLINE CRL skill controller over the two **cube** DDS checkpoints in
# ckpts/final/dds to the rnn.ist.berkeley.edu Slurm cluster: 5 seeds (0-4) per
# checkpoint, one sbatch job per (checkpoint, seed) -> 10 jobs.
#
# Companion of submit_dds_online_controller_seeds.sh (the antmaze / antsoccer /
# pointmaze checkpoints); it reuses the very same per-run body,
# run_dds_online_controller_seed.sbatch, so all DDS online-controller runs share
# one set of flags: RLPD on with the checkpoint's OWN offline dataset, K=10,
# target_entropy_multiplier=0.5, 1M env steps, eval every 20k.
#
# Online env per checkpoint (the deterministic, noise-free `*-center-` task sets;
# see ogbench/manipspace/__init__.py):
#   cube-single-play -> cube-single-center-online-v0   (horizon 200, one pick-and-place)
#   cube-double-play -> cube-double-center-online-v0   (horizon 500, two pick-and-places)
# Both horizons are divisible by K=10, which main_online.py requires for macro
# rollouts, and K=10 is also each checkpoint's `sequence_length` -- the value
# online_crl_skill_controller.py's DDS window labeller insists on. Episode length
# is left at the registered horizon, so no override is passed.
#
# Layout note: unlike the other five ckpts/final/dds entries, the two cube runs are
# stored FLAT (flags.json / params_1000000.pkl directly under the env dir, no
# sd000_* subdirectory). The resolver below accepts either shape.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn
# login node, from this NAS checkout:
#   bash scripts/slurm/submit_dds_online_controller_cube_seeds.sh
# Overrides:
#   DRY_RUN=1                    print the sbatch commands without submitting
#   SEEDS="0 1"                  a different seed set
#   CKPT_DIRS="cube-single-play" a subset of the two env dirs
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * cube-{single,double}-play-v0.npz in /nas/ucb/ishirgarg/.ogbench/data (compute
#     nodes have no internet egress). Both were present as of 2026-09-10.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_dds_online_controller_seed.sbatch
LOG_DIR=logs/slurm/dds_online_controller
DDS_ROOT=${DDS_ROOT:-ckpts/final/dds}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}

# Parallel arrays: checkpoint env dir, short tag, online env.
ALL_CKPT_DIRS=(cube-single-play cube-double-play)
ALL_CKPT_TAGS=(cube_sgl cube_dbl)
ALL_ONLINE_ENVS=(cube-single-center-online-v0 cube-double-center-online-v0)

read -r -a WANTED <<< "${CKPT_DIRS:-${ALL_CKPT_DIRS[*]}}"
CKPT_DIRS=(); CKPT_TAGS=(); ONLINE_ENVS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_CKPT_DIRS[@]}"; do
        if [[ "${ALL_CKPT_DIRS[$i]}" == "$w" ]]; then
            CKPT_DIRS+=("${ALL_CKPT_DIRS[$i]}")
            CKPT_TAGS+=("${ALL_CKPT_TAGS[$i]}")
            ONLINE_ENVS+=("${ALL_ONLINE_ENVS[$i]}")
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

    # RLPD reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    OFFLINE_DATASET=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi

    TAG=${CKPT_TAGS[$c]}
    ENV_NAME=${ONLINE_ENVS[$c]}
    for SEED in $SEEDS; do
        JOB_NAME="ddsctrl_${TAG}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        # The cube offline datasets are ~300 MB and the DDS window-labelling pass over
        # them is the memory peak, so raise the sbatch script's 16 GB default and keep
        # the 16 GB A4000 nodes out (same reasoning as run_dds_controller_alpha_seed.sbatch).
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
             --mem=32gb --exclude=ppo.ist.berkeley.edu,vae.ist.berkeley.edu
             "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "" "")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
