#!/usr/bin/env bash
# Submit the online CRL skill controller over the 8 final **discrete OPAL**
# checkpoints (ckpts/final/opal, latent_type=discrete, K=50, chunk_size=10) to
# the rnn.ist.berkeley.edu Slurm cluster: 5 seeds (0-4) per checkpoint, one
# sbatch job per (checkpoint, seed) -> 40 jobs.
#
# Same setup as the DDS controller sweep (submit_dds_online_controller_seeds.sh
# + its cube companion): RLPD on, offline dataset = each checkpoint's OWN dataset
# (the data its skills were trained on, read from flags.json by the sbatch
# script), K=10 (= the checkpoints' chunk_size), target_entropy_frac=0.9, 1M env
# steps. The offline windows are labelled by the checkpoint's own clustering
# posterior p(z | tau) (agent.opal_label_mode=sample by default; OPAL_LABEL_MODE=mode
# for the argmax). See run_opal_online_controller_seed.sbatch for the exact flags.
#
# Online env per checkpoint (the deterministic, noise-free `*-center-` task sets;
# see ogbench/locomaze/__init__.py and ogbench/manipspace/__init__.py):
#   antmaze-medium-{navigate,stitch}     -> antmaze-medium-center-online-v0     (horizon 1000)
#   antsoccer-arena-{navigate,stitch}    -> antsoccer-arena-center-online-v0    (horizon 500, overridden)
#   pointmaze-teleport-{navigate,stitch} -> pointmaze-teleport-center-online-v0 (horizon 1000)
#   cube-single-play                     -> cube-single-center-online-v0        (horizon 200)
#   cube-double-play                     -> cube-double-center-online-v0        (horizon 500)
# All horizons are divisible by K=10, which main_online.py requires for macro rollouts.
#
# Checkpoint phase: utils/skill_checkpoint.py refuses an epoch inside the EM
# clustering stage (< cluster_steps=500000; the BC decoder is untrained there).
# As of 2026-09-14 ckpts/final/opal/cube-single-play only holds params_400000.pkl,
# so it is EXCLUDED from the default set below (pass CKPT_DIRS=cube-single-play to
# try it once a later checkpoint is in place; the sbatch will fail fast otherwise).
#
# This ONLY submits jobs -- it does not run any training itself. Run from the
# rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_opal_online_controller_seeds.sh
# Overrides:
#   DRY_RUN=1                        print the sbatch commands without submitting
#   SEEDS="0 1"                      a different seed set
#   CKPT_DIRS="antmaze-medium-stitch cube-double-play"   a subset of the env dirs
#   OPAL_LABEL_MODE=mode             argmax posterior labels instead of samples
#   SAVE_SUBDIR=online_controller_x  write results to <ckpt>/<SAVE_SUBDIR>/rlpd instead
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each checkpoint's offline dataset must be present in
#     /nas/ucb/ishirgarg/.ogbench/data (compute nodes have no internet egress).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_opal_online_controller_seed.sbatch
LOG_DIR=logs/slurm/opal_online_controller
OPAL_ROOT=${OPAL_ROOT:-ckpts/final/opal}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
OPAL_LABEL_MODE=${OPAL_LABEL_MODE:-sample}
SAVE_SUBDIR=${SAVE_SUBDIR:-online_controller}
TARGET_ENTROPY_FRAC=${TARGET_ENTROPY_FRAC:-0.9}

# Parallel arrays: checkpoint env dir, short tag, online env, episode-length override.
ALL_CKPT_DIRS=(
    antmaze-medium-navigate
    antmaze-medium-stitch
    antsoccer-arena-navigate
    antsoccer-arena-stitch
    pointmaze-teleport-navigate
    pointmaze-teleport-stitch
    cube-single-play
    cube-double-play
)
ALL_CKPT_TAGS=(amz_nav amz_sti asoc_nav asoc_sti pmt_nav pmt_sti cube_sgl cube_dbl)
ALL_ONLINE_ENVS=(
    antmaze-medium-center-online-v0
    antmaze-medium-center-online-v0
    antsoccer-arena-center-online-v0
    antsoccer-arena-center-online-v0
    pointmaze-teleport-center-online-v0
    pointmaze-teleport-center-online-v0
    cube-single-center-online-v0
    cube-double-center-online-v0
)
ALL_EPISODE_LENGTHS=("" "" 500 500 "" "" "" "")
# Default set: everything except cube-single-play (see the header).
DEFAULT_CKPT_DIRS="antmaze-medium-navigate antmaze-medium-stitch antsoccer-arena-navigate antsoccer-arena-stitch pointmaze-teleport-navigate pointmaze-teleport-stitch cube-double-play"

read -r -a WANTED <<< "${CKPT_DIRS:-$DEFAULT_CKPT_DIRS}"
CKPT_DIRS=(); CKPT_TAGS=(); ONLINE_ENVS=(); EPISODE_LENGTHS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_CKPT_DIRS[@]}"; do
        if [[ "${ALL_CKPT_DIRS[$i]}" == "$w" ]]; then
            CKPT_DIRS+=("${ALL_CKPT_DIRS[$i]}")
            CKPT_TAGS+=("${ALL_CKPT_TAGS[$i]}")
            ONLINE_ENVS+=("${ALL_ONLINE_ENVS[$i]}")
            EPISODE_LENGTHS+=("${ALL_EPISODE_LENGTHS[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown checkpoint dir '$w' (known: ${ALL_CKPT_DIRS[*]})" >&2; exit 1; }
done

# Accept both the flat layout (flags.json directly under the env dir) and sd000_*/.
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
    SKILL_CKPT=$(resolve_ckpt "$OPAL_ROOT/${CKPT_DIRS[$c]}")
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    # Discrete checkpoints only, past the EM clustering stage; fail before submitting.
    python - "$SKILL_CKPT" <<'PY' || exit 1
import glob, json, os, re, sys
d = sys.argv[1]
a = json.load(open(os.path.join(d, 'flags.json')))['agent']
if a.get('agent_name') != 'opal' or a.get('latent_type') != 'discrete':
    sys.exit(f"ERROR: {d} is not a discrete opal run (agent_name={a.get('agent_name')!r}, latent_type={a.get('latent_type')!r})")
epoch = max(int(re.search(r'params_(\d+)\.pkl$', os.path.basename(p)).group(1)) for p in glob.glob(os.path.join(d, 'params_*.pkl')))
if epoch < int(a.get('cluster_steps', 0)):
    sys.exit(f"ERROR: {d}: latest epoch {epoch} < cluster_steps={a['cluster_steps']} (decoder untrained)")
PY

    # RLPD reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    OFFLINE_DATASET=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi

    TAG=${CKPT_TAGS[$c]}
    ENV_NAME=${ONLINE_ENVS[$c]}
    EPISODE_LENGTH=${EPISODE_LENGTHS[$c]}
    for SEED in $SEEDS; do
        JOB_NAME="opalctrl_${TAG}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
             "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "" "$EPISODE_LENGTH"
             "$SAVE_SUBDIR" "$TARGET_ENTROPY_FRAC" "$OPAL_LABEL_MODE")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
