#!/usr/bin/env bash
# DDS counterpart of submit_loglik_threshold_sweep.sh: the same four (checkpoint, online env)
# cells and the same online-controller settings, but on the frozen **DDS** skill checkpoints
# under ckpts/final/dds, sweeping only RLPD on/off.
#
# No offline-window filtering here. `agent.offline_loglik_threshold` is an empowerment-only
# knob (the BC log-likelihood score exists only for that labeller; utils/rlpd.py raises for
# DDS), so this sweep is exactly the two endpoints:
#
#   rlpd      RLPD on, the checkpoint's own offline dataset mixed into every batch
#   norlpd    RLPD off (--offline_dataset not passed), online data only
#
#   4 cells x 2 configs x 3 seeds = 24 jobs.
#
# Cells (checkpoints under ckpts/final/dds/, online env, horizon) -- all four are K=50,
# sequence_length=10, epoch 1M, from the 2026-09-01 batch:
#   pointmaze-teleport-navigate -> pointmaze-teleport-sparse-online-v0     (1000, registered)
#   antsoccer-arena-navigate     -> antsoccer-arena-center-online-v0        (500, overridden)
#   antsoccer-arena-stitch       -> antsoccer-arena-center-online-v0        (500, overridden)
#   antmaze-medium-stitch        -> antmaze-medium-corner-sparse-online-v0  (1000, registered)
#
# skill_commitment_k=10 is not a tuning choice: the DDS offline-window labeller in
# online_crl_skill_controller.py requires k == the checkpoint's `sequence_length`, which is 10
# for all four. main_online.py also requires episode_horizon % k == 0, true for 1000 and 500.
# antsoccer-arena-center-online-v0 is registered at 1000 but every online run in this repo
# uses 500 (scripts/run_online_crl.sh's per-env table), so it is passed explicitly.
#
# target_entropy_frac=0.75 for every job, matching the empowerment sweep this is compared
# against (non-legacy formula, target_entropy = frac * log(num_skills) = 0.75 * log 50).
# Everything else is the shared online-controller default: 1M env steps, eval every 20k.
#
# Results go to
#   <SKILL_CKPT>/online_controller_rlpd_ablation/{rlpd,norlpd}/
# -- a tree separate from every prior sweep's <SKILL_CKPT>/online_controller*/, so nothing
# existing is touched.
#
# This ONLY submits jobs. Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_dds_rlpd_ablation.sh
# Overrides:
#   DRY_RUN=1                       print the sbatch commands without submitting
#   SEEDS="0 1"                     a different seed set
#   CONFIGS="rlpd"                  a subset of the two configurations
#   CKPT_KEYS="dds_amz_sti"         a subset of the four cells
#   TIME_LIMIT=24:00:00             sbatch --time override
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each checkpoint's offline dataset present in /nas/ucb/ishirgarg/.ogbench/data
#     (compute nodes have no internet egress).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

DDS_SBATCH=scripts/slurm/run_dds_online_controller_seed.sbatch
LOG_DIR=logs/slurm/dds_rlpd_ablation
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
TIME_LIMIT=${TIME_LIMIT:-16:00:00}
TARGET_ENTROPY_FRAC=${TARGET_ENTROPY_FRAC:-0.75}
mkdir -p "$LOG_DIR"

# sbatch exports the submitting environment (--export=ALL), so these two env-var knobs of
# run_dds_online_controller_seed.sbatch would silently leak into every job and make this
# ablation incomparable to the empowerment sweep (which predates both). Clear them here.
for v in RLPD_FRAC_TIME USE_TES TES_PATIENCE; do
    if [[ -n "${!v:-}" ]]; then
        echo "NOTE: clearing $v=${!v} for this sweep (it must match the empowerment runs)." >&2
        unset "$v"
    fi
done

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2"}
CONFIGS=${CONFIGS:-"rlpd norlpd"}

DDS_ROOT=ckpts/final/dds

# Parallel arrays: key, checkpoint env dir, online env, episode-length override.
ALL_KEYS=(dds_pmt_nav dds_asoc_nav dds_asoc_sti dds_amz_sti)
ALL_CKPT_ROOTS=(
    "$DDS_ROOT/pointmaze-teleport-navigate"
    "$DDS_ROOT/antsoccer-arena-navigate"
    "$DDS_ROOT/antsoccer-arena-stitch"
    "$DDS_ROOT/antmaze-medium-stitch"
)
ALL_ONLINE_ENVS=(
    pointmaze-teleport-sparse-online-v0
    antsoccer-arena-center-online-v0
    antsoccer-arena-center-online-v0
    antmaze-medium-corner-sparse-online-v0
)
ALL_EPISODE_LENGTHS=("" 500 500 "")

read -r -a WANTED <<< "${CKPT_KEYS:-${ALL_KEYS[*]}}"
KEYS=(); CKPT_ROOTS=(); ONLINE_ENVS=(); EPISODE_LENGTHS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_KEYS[@]}"; do
        if [[ "${ALL_KEYS[$i]}" == "$w" ]]; then
            KEYS+=("${ALL_KEYS[$i]}")
            CKPT_ROOTS+=("${ALL_CKPT_ROOTS[$i]}")
            ONLINE_ENVS+=("${ALL_ONLINE_ENVS[$i]}")
            EPISODE_LENGTHS+=("${ALL_EPISODE_LENGTHS[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown key '$w' (known: ${ALL_KEYS[*]})" >&2; exit 1; }
done

# Resolve an env dir to its single run (or, if several, the K=50 one) -- the same rule the
# empowerment submitters use. Also asserts sequence_length == 10, which the DDS labeller needs.
resolve_ckpt() {
    "$PYTHON" - "$1" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
runs = [root] if os.path.exists(os.path.join(root, 'flags.json')) else [
    r for r in sorted(glob.glob(os.path.join(root, 'sd000_*')))
    if os.path.exists(os.path.join(r, 'flags.json'))
]
if not runs:
    sys.exit(f'ERROR: no run with a flags.json under {root}')
if len(runs) > 1:
    runs = [r for r in runs
            if int(json.load(open(os.path.join(r, 'flags.json')))['agent']['num_skills']) == 50]
    if len(runs) != 1:
        sys.exit(f'ERROR: {root} has {len(runs)} candidate K=50 runs; pick one explicitly.')
cfg = json.load(open(os.path.join(runs[0], 'flags.json')))['agent']
if cfg.get('agent_name') != 'dds':
    sys.exit(f'ERROR: {runs[0]} is agent_name={cfg.get("agent_name")!r}, expected dds.')
if int(cfg['sequence_length']) != 10:
    sys.exit(f'ERROR: {runs[0]} has sequence_length={cfg["sequence_length"]}; the DDS offline-window '
             'labeller requires it to equal skill_commitment_k=10.')
print(runs[0])
PY
}

n_submitted=0
for c in "${!KEYS[@]}"; do
    KEY=${KEYS[$c]}
    SKILL_CKPT=$(resolve_ckpt "${CKPT_ROOTS[$c]}")
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    # RLPD reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    OFFLINE_DATASET=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; the rlpd jobs will fail on a compute node." >&2
    fi

    ENV_NAME=${ONLINE_ENVS[$c]}
    EPISODE_LENGTH=${EPISODE_LENGTHS[$c]}
    echo "# $KEY -> $SKILL_CKPT (env=$ENV_NAME, rlpd dataset $OFFLINE_DATASET)"

    for CONFIG in $CONFIGS; do
        case "$CONFIG" in
            rlpd)   DATASET_ARG="" ;;      # the checkpoint's own dataset, resolved by the sbatch
            norlpd) DATASET_ARG="none" ;;  # RLPD off
            *) echo "ERROR: unknown config '$CONFIG' (known: rlpd norlpd)" >&2; exit 1 ;;
        esac
        for SEED in $SEEDS; do
            JOB_NAME="ddsrlpd_${KEY}_${CONFIG}_s${SEED}"
            OUT="$LOG_DIR/${JOB_NAME}_%j.log"
            cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$TIME_LIMIT"
                 "$DDS_SBATCH" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "$DATASET_ARG" "$EPISODE_LENGTH"
                 "online_controller_rlpd_ablation" "$TARGET_ENTROPY_FRAC")
            echo "${cmd[@]}"
            if [[ "$DRY_RUN" != "1" ]]; then
                "${cmd[@]}"
            fi
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
