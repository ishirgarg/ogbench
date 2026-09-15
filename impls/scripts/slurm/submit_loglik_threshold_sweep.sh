#!/usr/bin/env bash
# Sweep the RLPD offline-data filter (agents/online_crl_skill_controller.py's
# `offline_loglik_threshold`, added 2026-09-13) for the online skill controller on
# four frozen empowerment checkpoints.
#
# The filter drops every offline window the frozen skill policy cannot reproduce,
# before that window ever becomes a row of the RLPD offline buffer. A window's score
# is the winning skill's BC log-likelihood per step and per action dimension
# (utils/rlpd.py; utils/datasets.py: SequenceDataset.relabel_chunk_skills), so the same
# threshold means the same thing in every env -- with the policy's const_std=True actor,
# score = -MSE_per_dim / 2 - 0.9189, i.e. -0.9189 is a perfect fit and -1.05 / -1.0 /
# -0.97 correspond to per-dimension squared errors of 0.262 / 0.162 / 0.102.
#
# Five configurations, three seeds each, four (checkpoint, online env) cells:
#
#   nofilter  RLPD on, no threshold        -- keep every offline window (the status quo)
#   norlpd    RLPD off (--offline_dataset  -- the "filter everything" end of the sweep;
#             not passed)                     no offline data reaches the learner at all
#   thr-1.05  RLPD on, threshold -1.05     -- loosest filter
#   thr-1.0   RLPD on, threshold -1.0
#   thr-0.97  RLPD on, threshold -0.97     -- tightest filter
#
#   4 cells x 5 configs x 3 seeds = 60 jobs.
#
# Cells (checkpoints under ckpts/final/empowerment_final/, online env, horizon):
#   pointmaze-teleport-navigate -> pointmaze-teleport-sparse-online-v0     (1000, registered)
#   antsoccer-arena-navigate     -> antsoccer-arena-center-online-v0        (500, overridden)
#   antsoccer-arena-stitch       -> antsoccer-arena-center-online-v0        (500, overridden)
#   antmaze-medium-stitch        -> antmaze-medium-corner-sparse-online-v0  (1000, registered)
#
# antsoccer-arena-center-online-v0 is registered at 1000 but every online run in this repo
# uses 500 (scripts/run_online_crl.sh's per-env table), so it is passed explicitly.
# All horizons are divisible by skill_commitment_k=10, which main_online.py requires.
#
# antmaze-medium-stitch holds TWO runs (a K=50 and a K=15); `pick_k50` below selects the
# K=50 one, the same rule submit_pointmaze_antmaze_skill_controller_seeds.sh and
# submit_antsoccer_corner_online_controller_seeds.sh already use, and the one that matches
# the other three cells (all K=50).
#
# target_entropy_frac=0.75 for every job (the non-legacy formula,
# target_entropy = frac * log(num_skills); use_legacy_entropy left at False). Everything
# else matches every other online-controller sweep in this repo: skill_commitment_k=10,
# 1M env steps, RLPD on each checkpoint's OWN offline dataset.
#
# Results go to
#   <SKILL_CKPT>/online_controller_loglik_sweep/<CONFIG>/{rlpd,norlpd}/
# -- a tree separate from every prior sweep's <SKILL_CKPT>/online_controller*/, so nothing
# existing is touched.
#
# This ONLY submits jobs. Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_loglik_threshold_sweep.sh
# Overrides:
#   DRY_RUN=1                          print the sbatch commands without submitting
#   SEEDS="0 1"                        a different seed set
#   CONFIGS="nofilter thr-1.0"         a subset of the five configurations
#   CKPT_KEYS="emp_amz_sti"            a subset of the four cells
#   TIME_LIMIT=24:00:00                sbatch --time override
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each checkpoint's offline dataset present in /nas/ucb/ishirgarg/.ogbench/data
#     (compute nodes have no internet egress).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

EMP_SBATCH=scripts/slurm/run_online_crl_skill_controller_seed.sbatch
LOG_DIR=logs/slurm/loglik_threshold_sweep
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
TIME_LIMIT=${TIME_LIMIT:-16:00:00}
TARGET_ENTROPY_FRAC=${TARGET_ENTROPY_FRAC:-0.75}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2"}
CONFIGS=${CONFIGS:-"nofilter norlpd thr-1.05 thr-1.0 thr-0.97"}

EMP_ROOT=ckpts/final/empowerment_final

# Parallel arrays: key, checkpoint env dir, online env, episode-length override.
ALL_KEYS=(emp_pmt_nav emp_asoc_nav emp_asoc_sti emp_amz_sti)
ALL_CKPT_ROOTS=(
    "$EMP_ROOT/pointmaze-teleport-navigate"
    "$EMP_ROOT/antsoccer-arena-navigate"
    "$EMP_ROOT/antsoccer-arena-stitch"
    "$EMP_ROOT/antmaze-medium-stitch"
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

# Pick the K=50 run out of an env dir (antmaze-medium-stitch holds a K=50 and a K=15);
# a dir with exactly one sd000_* run resolves to it whatever its K.
pick_k50() {
    "$PYTHON" - "$1" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
if os.path.exists(os.path.join(root, 'flags.json')):
    print(root); raise SystemExit
runs = sorted(glob.glob(os.path.join(root, 'sd000_*')))
runs = [r for r in runs if os.path.exists(os.path.join(r, 'flags.json'))]
if len(runs) == 1:
    print(runs[0]); raise SystemExit
k50 = [r for r in runs
       if int(json.load(open(os.path.join(r, 'flags.json')))['agent']['num_skills']) == 50]
if len(k50) != 1:
    sys.exit(f'ERROR: {root} has {len(runs)} runs and {len(k50)} with num_skills=50; pick one explicitly.')
print(k50[0])
PY
}

# Map a config name to (OFFLINE_DATASET override, LOGLIK_THRESHOLD). "" for OFFLINE_DATASET
# means "the checkpoint's own dataset", which the sbatch resolves from its flags.json.
config_args() {
    case "$1" in
        nofilter) echo "|" ;;         # RLPD on, no threshold
        norlpd)   echo "none|" ;;     # RLPD off
        # `thr-1.05` means threshold -1.05: strip only `thr`, so the sign stays.
        thr-*)    echo "|${1#thr}" ;;
        *) echo "ERROR: unknown config '$1'" >&2; exit 1 ;;
    esac
}

for c in $CONFIGS; do config_args "$c" > /dev/null; done   # fail fast on a typo

n_submitted=0
for c in "${!KEYS[@]}"; do
    KEY=${KEYS[$c]}
    SKILL_CKPT=$(pick_k50 "${CKPT_ROOTS[$c]}")
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    # RLPD reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    OFFLINE_DATASET=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi

    ENV_NAME=${ONLINE_ENVS[$c]}
    EPISODE_LENGTH=${EPISODE_LENGTHS[$c]}
    echo "# $KEY -> $SKILL_CKPT (env=$ENV_NAME, rlpd dataset $OFFLINE_DATASET)"

    for CONFIG in $CONFIGS; do
        IFS='|' read -r DATASET_ARG THRESHOLD_ARG <<< "$(config_args "$CONFIG")"
        SAVE_SUBDIR="online_controller_loglik_sweep/$CONFIG"
        for SEED in $SEEDS; do
            JOB_NAME="llthr_${KEY}_${CONFIG}_s${SEED}"
            OUT="$LOG_DIR/${JOB_NAME}_%j.log"
            cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$TIME_LIMIT"
                 "$EMP_SBATCH" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "$DATASET_ARG" "$EPISODE_LENGTH"
                 "$SAVE_SUBDIR" "$TARGET_ENTROPY_FRAC" "$THRESHOLD_ARG")
            echo "${cmd[@]}"
            if [[ "$DRY_RUN" != "1" ]]; then
                "${cmd[@]}"
            fi
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
