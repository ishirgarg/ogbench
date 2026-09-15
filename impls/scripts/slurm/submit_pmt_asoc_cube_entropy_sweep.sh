#!/usr/bin/env bash
# Sweep the online skill controller's (non-legacy) target_entropy_frac knob
# (agents/online_crl_skill_controller.py: target_entropy = frac * log(num_skills),
# use_legacy_entropy left at its default False) over {0.25, 0.5, 0.75}, for BOTH
# frozen skill families (dds, empowerment), on the three "navigate" checkpoints
# for PointMaze-Teleport, AntSoccer-Arena, and Cube-Single. One sbatch job per
# (family, env, entropy_frac, seed):
#
#   2 skill families (dds, empowerment)
#   x 3 envs         (pointmaze-teleport-navigate, antsoccer-arena-navigate, cube-single-play)
#   x 3 entropy_frac (0.25, 0.5, 0.75)
#   x 5 seeds        (0-4)
#   = 90 jobs total.
#
# This reuses the same two per-run sbatch bodies as every other online-controller
# sweep in this repo -- run_dds_online_controller_seed.sbatch and
# run_online_crl_skill_controller_seed.sbatch -- via their new (2026-09-13)
# optional 7th positional arg, TARGET_ENTROPY_FRAC (default 0.9, so every
# pre-existing caller of those two scripts is unaffected by this change). All
# other flags match those scripts' fixed defaults: RLPD on with each
# checkpoint's OWN offline dataset, skill_commitment_k=10, 1M env steps.
#
# Checkpoints (all under ckpts/final/):
#   dds/pointmaze-teleport-navigate           -- single sd000_* run
#   dds/antsoccer-arena-navigate               -- single sd000_* run
#   dds/cube-single-play                       -- single sd000_* run (DDS has only one
#                                                  cube-single checkpoint)
#   empowerment_final/pointmaze-teleport-navigate -- single sd000_* run
#   empowerment_final/antsoccer-arena-navigate    -- single sd000_* run
#   empowerment_final/cube-single-play/sd000_s_38624008.0.20260908_013305
#       -- explicit checkpoint, NOT auto-resolved: this env dir holds FIVE
#          separate sd000_* runs (5 seeds from the 2026-09-04/08 batches); this
#          specific one was chosen by the user (2026-09-13) for this sweep.
#
# Online env per env family (the deterministic, noise-free task sets; see
# ogbench/locomaze/__init__.py and ogbench/manipspace/__init__.py):
#   pointmaze-teleport -> pointmaze-teleport-sparse-online-v0  (horizon 1000, no override;
#                         same env used by submit_pointmaze_antmaze_skill_controller_seeds.sh)
#   antsoccer-arena     -> antsoccer-arena-center-online-v0    (registered at 1000, but
#                         every online run in this repo overrides it to 500 -- passed
#                         explicitly here too)
#   cube-single         -> cube-single-center-online-v0        (horizon 200, no override)
# All three horizons are divisible by skill_commitment_k=10, which main_online.py
# requires for macro rollouts.
#
# Results are saved under
#   <SKILL_CKPT>/online_controller_entropy_sweep/frac<FRAC>/rlpd/
# -- a tree separate from every prior sweep's <SKILL_CKPT>/online_controller*/,
# so nothing existing is touched.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the
# rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_pmt_asoc_cube_entropy_sweep.sh
# Overrides:
#   DRY_RUN=1                         print the sbatch commands without submitting
#   SEEDS="0 1"                       a different seed set
#   ENTROPY_FRACS="0.25 0.5"          a different entropy_frac set
#   CKPT_KEYS="dds_cube_sgl emp_cube_sgl"   a subset of the six (family, env) cells
#   TIME_LIMIT=16:00:00               sbatch --time override
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each checkpoint's offline dataset present in /nas/ucb/ishirgarg/.ogbench/data
#     (compute nodes have no internet egress).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

DDS_SBATCH=scripts/slurm/run_dds_online_controller_seed.sbatch
EMP_SBATCH=scripts/slurm/run_online_crl_skill_controller_seed.sbatch
LOG_DIR=logs/slurm/pmt_asoc_cube_entropy_sweep
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
TIME_LIMIT=${TIME_LIMIT:-16:00:00}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
ENTROPY_FRACS=${ENTROPY_FRACS:-"0.25 0.5 0.75"}

# Parallel arrays: key, checkpoint root (env dir to resolve, OR an exact leaf
# checkpoint dir for the ambiguous empowerment cube case), sbatch script,
# online env, episode-length override, extra sbatch flags (mem/exclude for cube).
ALL_KEYS=(dds_pmt_nav dds_asoc_nav dds_cube_sgl emp_pmt_nav emp_asoc_nav emp_cube_sgl)
ALL_CKPT_ROOTS=(
    "ckpts/final/dds/pointmaze-teleport-navigate"
    "ckpts/final/dds/antsoccer-arena-navigate"
    "ckpts/final/dds/cube-single-play"
    "ckpts/final/empowerment_final/pointmaze-teleport-navigate"
    "ckpts/final/empowerment_final/antsoccer-arena-navigate"
    "ckpts/final/empowerment_final/cube-single-play/sd000_s_38624008.0.20260908_013305"
)
ALL_SBATCH=("$DDS_SBATCH" "$DDS_SBATCH" "$DDS_SBATCH" "$EMP_SBATCH" "$EMP_SBATCH" "$EMP_SBATCH")
ALL_ONLINE_ENVS=(
    pointmaze-teleport-sparse-online-v0
    antsoccer-arena-center-online-v0
    cube-single-center-online-v0
    pointmaze-teleport-sparse-online-v0
    antsoccer-arena-center-online-v0
    cube-single-center-online-v0
)
ALL_EPISODE_LENGTHS=("" 500 "" "" 500 "")
# cube's DDS window-labelling / RLPD pass over its ~300MB dataset is the memory
# peak (same reasoning as submit_dds_online_controller_cube_seeds.sh); give
# those two cells more headroom and keep the 16GB A4000 nodes out.
ALL_EXTRA_SBATCH_FLAGS=(
    ""
    ""
    "--mem=32gb --exclude=ppo.ist.berkeley.edu,vae.ist.berkeley.edu"
    ""
    ""
    "--mem=32gb --exclude=ppo.ist.berkeley.edu,vae.ist.berkeley.edu"
)

read -r -a WANTED <<< "${CKPT_KEYS:-${ALL_KEYS[*]}}"
KEYS=(); CKPT_ROOTS=(); SBATCHES=(); ONLINE_ENVS=(); EPISODE_LENGTHS=(); EXTRA_SBATCH_FLAGS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_KEYS[@]}"; do
        if [[ "${ALL_KEYS[$i]}" == "$w" ]]; then
            KEYS+=("${ALL_KEYS[$i]}")
            CKPT_ROOTS+=("${ALL_CKPT_ROOTS[$i]}")
            SBATCHES+=("${ALL_SBATCH[$i]}")
            ONLINE_ENVS+=("${ALL_ONLINE_ENVS[$i]}")
            EPISODE_LENGTHS+=("${ALL_EPISODE_LENGTHS[$i]}")
            EXTRA_SBATCH_FLAGS+=("${ALL_EXTRA_SBATCH_FLAGS[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown key '$w' (known: ${ALL_KEYS[*]})" >&2; exit 1; }
done

# Accept: (a) an exact leaf checkpoint dir (has flags.json directly -- used for
# the ambiguous empowerment cube-single-play case above), or (b) an env dir
# holding exactly one sd000_* run (or, for a flat layout, flags.json directly).
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
for c in "${!KEYS[@]}"; do
    KEY=${KEYS[$c]}
    SKILL_CKPT=$(resolve_ckpt "${CKPT_ROOTS[$c]}")
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    # RLPD reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    OFFLINE_DATASET=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi

    ENV_NAME=${ONLINE_ENVS[$c]}
    EPISODE_LENGTH=${EPISODE_LENGTHS[$c]}
    read -r -a EXTRA_FLAGS <<< "${EXTRA_SBATCH_FLAGS[$c]}"
    echo "# $KEY -> $SKILL_CKPT (env=$ENV_NAME, rlpd dataset $OFFLINE_DATASET)"

    for FRAC in $ENTROPY_FRACS; do
        FRAC_TAG=${FRAC//./p}   # 0.25 -> 0p25, for job names / dirnames
        SAVE_SUBDIR="online_controller_entropy_sweep/frac${FRAC}"
        for SEED in $SEEDS; do
            JOB_NAME="entctrl_${KEY}_e${FRAC_TAG}_s${SEED}"
            OUT="$LOG_DIR/${JOB_NAME}_%j.log"
            cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$TIME_LIMIT"
                 "${EXTRA_FLAGS[@]}"
                 "${SBATCHES[$c]}" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "" "$EPISODE_LENGTH" "$SAVE_SUBDIR" "$FRAC")
            echo "${cmd[@]}"
            if [[ "$DRY_RUN" != "1" ]]; then
                "${cmd[@]}"
            fi
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
