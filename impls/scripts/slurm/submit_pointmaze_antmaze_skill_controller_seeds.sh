#!/usr/bin/env bash
# Submit the online skill controller (online_crl_skill_controller, over a frozen
# **empowerment_skill** OR **dds** checkpoint) for every (skill family, env, dataset
# variant) cell on PointMaze-Teleport and AntMaze-Medium, to the
# rnn.ist.berkeley.edu Slurm cluster: 5 seeds (0-4) per cell, one sbatch job per
# (cell, seed).
#
#   2 skill families (dds, empowerment_skill)
#   x 2 envs         (pointmaze-teleport, antmaze-medium)
#   x 2 variants     (navigate, stitch)
#   x 5 seeds
#   = 40 jobs total.
#
# (If you only want ONE skill family, or ONE env, filter with CKPT_KEYS -- see
# below. The pre-existing submit_dds_online_controller_seeds.sh and
# submit_empowerment_online_controller_seeds.sh scripts each cover only half of
# this 2x2x2 grid -- notably neither covers the empowerment_skill x antmaze-medium
# cell, since ckpts/final/empowerment_final/antmaze-medium-{navigate,stitch} each
# hold TWO sd000_* runs, a K=50 and a K=15, and picking the wrong one silently
# halves the skill count. This script always resolves checkpoints by reading
# flags.json's `num_skills` (via `pick_k50`, borrowed from
# submit_antsoccer_corner_online_controller_seeds.sh) rather than trusting glob
# order, so it is correct even where the other two scripts had to punt.)
#
# All runs use RLPD (agents/online_crl_skill_controller.py's --offline_dataset),
# with the dataset defaulting -- inside run_dds_online_controller_seed.sbatch /
# run_online_crl_skill_controller_seed.sbatch -- to each checkpoint's OWN training
# dataset (its flags.json env_name), i.e. exactly the data its K=50 skills were
# trained on. skill_commitment_k=10 and target_entropy_multiplier=0.5 are fixed by
# those sbatch scripts (not tunable here -- see their own comment blocks for why).
#
# Online env per env family: the NEW 4-goal / 3-goal sparse task sets (registered
# 2026-09-12 in ogbench/locomaze/__init__.py), NOT the older 41-/22-goal `*-center-`
# sets. Both are registered at horizon 1000, divisible by skill_commitment_k=10, so
# no --episode_length override is needed.
#   pointmaze-teleport -> pointmaze-teleport-sparse-online-v0    (start (20,8); goals:
#                         the 3 far corners + straight up past the teleporter)
#   antmaze-medium     -> antmaze-medium-corner-sparse-online-v0 (start bottom-left
#                         corner (0,0); goals: the other 3 corners)
#
# Results are saved under <SKILL_CKPT>/online_controller_sparse/rlpd/ -- a SEPARATE
# tree from <SKILL_CKPT>/online_controller/rlpd/, which (as of 2026-09-12) holds an
# earlier, WRONG run of this same sweep against the older `*-center-` envs. Nothing
# on disk from that run is overwritten; if you don't need it, clean it up separately.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn
# login node, from this NAS checkout:
#   bash scripts/slurm/submit_pointmaze_antmaze_skill_controller_seeds.sh
# Overrides:
#   DRY_RUN=1                        print the sbatch commands without submitting
#   SEEDS="0 1"                      a different seed set
#   CKPT_KEYS="dds_pmt_nav emp_amz_sti"   a subset of the eight (family, env, variant) cells
#   TIME_LIMIT=16:00:00              sbatch --time override (default matches the
#                                    DDS/empowerment controller sweeps' 16h)
#
# Prerequisites (checked below):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * Each checkpoint's offline dataset present in /nas/ucb/ishirgarg/.ogbench/data
#     (compute nodes have no internet egress). pointmaze-teleport-stitch-v0.npz was
#     NOT there as of 2026-09-04 (see submit_dds_online_controller_seeds.sh) --
#     verify before submitting for real.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

DDS_SBATCH=scripts/slurm/run_dds_online_controller_seed.sbatch
EMP_SBATCH=scripts/slurm/run_online_crl_skill_controller_seed.sbatch
LOG_DIR=logs/slurm/pointmaze_antmaze_skill_controller
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
TIME_LIMIT=${TIME_LIMIT:-16:00:00}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}

# Parallel arrays: key, checkpoint search dir, sbatch script, online env.
ALL_KEYS=(
    dds_pmt_nav dds_pmt_sti dds_amz_nav dds_amz_sti
    emp_pmt_nav emp_pmt_sti emp_amz_nav emp_amz_sti
)
ALL_CKPT_GLOBS=(
    "ckpts/final/dds/pointmaze-teleport-navigate"
    "ckpts/final/dds/pointmaze-teleport-stitch"
    "ckpts/final/dds/antmaze-medium-navigate"
    "ckpts/final/dds/antmaze-medium-stitch"
    "ckpts/final/empowerment_final/pointmaze-teleport-navigate"
    "ckpts/final/empowerment_final/pointmaze-teleport-stitch"
    "ckpts/final/empowerment_final/antmaze-medium-navigate"
    "ckpts/final/empowerment_final/antmaze-medium-stitch"
)
ALL_SBATCH=(
    "$DDS_SBATCH" "$DDS_SBATCH" "$DDS_SBATCH" "$DDS_SBATCH"
    "$EMP_SBATCH" "$EMP_SBATCH" "$EMP_SBATCH" "$EMP_SBATCH"
)
ALL_ONLINE_ENVS=(
    pointmaze-teleport-sparse-online-v0
    pointmaze-teleport-sparse-online-v0
    antmaze-medium-corner-sparse-online-v0
    antmaze-medium-corner-sparse-online-v0
    pointmaze-teleport-sparse-online-v0
    pointmaze-teleport-sparse-online-v0
    antmaze-medium-corner-sparse-online-v0
    antmaze-medium-corner-sparse-online-v0
)
SAVE_SUBDIR=${SAVE_SUBDIR:-online_controller_sparse}

read -r -a WANTED <<< "${CKPT_KEYS:-${ALL_KEYS[*]}}"
KEYS=(); CKPT_GLOBS=(); SBATCHES=(); ONLINE_ENVS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_KEYS[@]}"; do
        if [[ "${ALL_KEYS[$i]}" == "$w" ]]; then
            KEYS+=("${ALL_KEYS[$i]}")
            CKPT_GLOBS+=("${ALL_CKPT_GLOBS[$i]}")
            SBATCHES+=("${ALL_SBATCH[$i]}")
            ONLINE_ENVS+=("${ALL_ONLINE_ENVS[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown key '$w' (known: ${ALL_KEYS[*]})" >&2; exit 1; }
done

# Pick the single K=50 run under a checkpoint env dir, reading flags.json rather than
# trusting glob order (the empowerment antmaze-medium dirs each hold a K=50 and a K=15
# run; every other dir here holds exactly one run, which this also verifies is K=50).
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

    # RLPD reads the checkpoint's own dataset; warn early if it isn't on the NAS.
    OFFLINE_DATASET=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi

    ENV_NAME=${ONLINE_ENVS[$c]}
    echo "# $KEY -> $SKILL_CKPT (env=$ENV_NAME, rlpd dataset $OFFLINE_DATASET)"
    for SEED in $SEEDS; do
        JOB_NAME="pmamzctrl_${KEY}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$TIME_LIMIT"
             "${SBATCHES[$c]}" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "" "" "$SAVE_SUBDIR")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
