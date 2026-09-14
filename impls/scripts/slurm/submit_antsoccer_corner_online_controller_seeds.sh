#!/usr/bin/env bash
# Submit the online CRL skill controller on the NEW antsoccer corner task to the
# rnn.ist.berkeley.edu Slurm cluster: 5 seeds (0-4) per checkpoint, one sbatch job per
# (checkpoint, seed) -> 20 jobs.
#
# Online env: antsoccer-arena-corner-online-v0 (registered 2026-09-12 in
# ogbench/locomaze/__init__.py) -- ant (2,2), ball (5,5), goal (5,10), no init/goal noise,
# single task. Registered horizon is 1000; 500 is passed explicitly here, matching every
# other antsoccer online run in this repo (scripts/run_online_crl.sh's per-env table and
# the two finished center-env sweeps), so the corner numbers are comparable to them.
# 500 % skill_commitment_k(10) == 0, which main_online.py requires for macro rollouts.
#
# Checkpoints -- the antsoccer-arena K=50 skill runs, stitch and navigate, for BOTH skill
# families, so the four cells of (dds | empowerment_skill) x (navigate | stitch) are covered:
#
#   dds        navigate  ckpts/final/dds/antsoccer-arena-navigate/sd000_*
#   dds        stitch    ckpts/dds/antsoccer-arena-stitch/<the K=50 run>       <-- NOT in final/
#   empowerment navigate ckpts/final/empowerment_final/antsoccer-arena-navigate/sd000_*
#   empowerment stitch   ckpts/final/empowerment_final/antsoccer-arena-stitch/sd000_*
#
# ckpts/final/dds holds no antsoccer-arena-stitch run, so the DDS stitch cell is filled from
# ckpts/dds/antsoccer-arena-stitch, which holds TWO sd000_* runs (K=50 and K=15); the K=50
# one is selected by reading flags.json, never by glob order. It is an older batch
# (2026-07-21) than the final/ runs (2026-09-01) but is the same agent at K=50,
# sequence_length=10, 1M steps. Drop it with CKPT_KEYS if you want final/-only results.
#
# RLPD is ON for every job: --offline_dataset defaults, inside the sbatch script, to each
# checkpoint's OWN training dataset (its flags.json env_name), i.e.
# antsoccer-arena-{navigate,stitch}-v0. The corner env itself has no dataset.
#
# Results go to <SKILL_CKPT>/online_controller_corner/rlpd/ -- a SEPARATE tree from the
# finished center-env sweeps in <SKILL_CKPT>/online_controller/rlpd/, via the optional 6th
# sbatch argument added for this script. Nothing already on disk is overwritten.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn login node,
# from this NAS checkout:
#   bash scripts/slurm/submit_antsoccer_corner_online_controller_seeds.sh
# Overrides:
#   DRY_RUN=1                     print the sbatch commands without submitting
#   SEEDS="0 1"                   a different seed set
#   CKPT_KEYS="dds_nav emp_sti"   a subset of the four cells
#   EPISODE_LENGTH=1000           use the env's registered horizon instead of 500
#
# Prerequisites (checked below):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each checkpoint's offline dataset present in /nas/ucb/ishirgarg/.ogbench/data
#     (compute nodes have no internet egress).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

DDS_SBATCH=scripts/slurm/run_dds_online_controller_seed.sbatch
EMP_SBATCH=scripts/slurm/run_online_crl_skill_controller_seed.sbatch
LOG_DIR=logs/slurm/antsoccer_corner_online_controller
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
TIME_LIMIT=${TIME_LIMIT:-16:00:00}
ONLINE_ENV=${ONLINE_ENV:-antsoccer-arena-corner-online-v0}
EPISODE_LENGTH=${EPISODE_LENGTH:-500}
SAVE_SUBDIR=${SAVE_SUBDIR:-online_controller_corner}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}

# Parallel arrays: key, checkpoint search dir, sbatch script, job-name tag.
ALL_KEYS=(dds_nav dds_sti emp_nav emp_sti)
ALL_CKPT_GLOBS=(
    "ckpts/final/dds/antsoccer-arena-navigate"
    "ckpts/dds/antsoccer-arena-stitch"
    "ckpts/final/empowerment_final/antsoccer-arena-navigate"
    "ckpts/final/empowerment_final/antsoccer-arena-stitch"
)
ALL_SBATCH=("$DDS_SBATCH" "$DDS_SBATCH" "$EMP_SBATCH" "$EMP_SBATCH")

read -r -a WANTED <<< "${CKPT_KEYS:-${ALL_KEYS[*]}}"
KEYS=(); CKPT_GLOBS=(); SBATCHES=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_KEYS[@]}"; do
        if [[ "${ALL_KEYS[$i]}" == "$w" ]]; then
            KEYS+=("${ALL_KEYS[$i]}")
            CKPT_GLOBS+=("${ALL_CKPT_GLOBS[$i]}")
            SBATCHES+=("${ALL_SBATCH[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown key '$w' (known: ${ALL_KEYS[*]})" >&2; exit 1; }
done

# Pick the single K=50 run under a checkpoint env dir, reading flags.json rather than
# trusting glob order (ckpts/dds/antsoccer-arena-stitch holds a K=50 and a K=15 run).
pick_k50 () {
    "$PYTHON" - "$1" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
hits = []
for d in sorted(glob.glob(os.path.join(root, 'sd000_*'))):
    fp = os.path.join(d, 'flags.json')
    if not os.path.isfile(fp):
        continue
    agent = json.load(open(fp)).get('agent', {})
    if agent.get('num_skills') == 50:
        hits.append(d)
if len(hits) != 1:
    sys.exit(f'ERROR: expected exactly one K=50 sd000_* run under {root}, found {len(hits)}: {hits}')
print(hits[0])
PY
}

n_submitted=0
for c in "${!KEYS[@]}"; do
    KEY=${KEYS[$c]}
    SKILL_CKPT=$(pick_k50 "${CKPT_GLOBS[$c]}")
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

    # RLPD reads the checkpoint's own dataset; fail early if it isn't on the NAS.
    OFFLINE_DATASET=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "ERROR: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs would fail on a compute node." >&2
        exit 1
    fi

    echo "# $KEY -> $SKILL_CKPT (rlpd dataset $OFFLINE_DATASET)"
    for SEED in $SEEDS; do
        JOB_NAME="corner_${KEY}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$TIME_LIMIT"
             "${SBATCHES[$c]}" "$SKILL_CKPT" "$ONLINE_ENV" "$SEED" "" "$EPISODE_LENGTH" "$SAVE_SUBDIR")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
