#!/usr/bin/env bash
# TES-SAC online skill-controller seeds: agents/online_crl_skill_controller.py with
# --agent.use_tes=True (Target Entropy Scheduled SAC, Xu et al. 2021, arXiv:2112.02852:
# the entropy target starts at TES_INIT_FRAC * log(num_skills) and is multiplied by 0.9
# whenever the batch policy entropy stabilises at it; tes_patience=500 gradient steps
# per drop, the paper's Table 1 for everything else). One sbatch job per
# (family, env, seed):
#
#   2 skill families (dds, empowerment; all six checkpoints have num_skills=50)
#   x 3 envs         (pointmaze-teleport-navigate, antsoccer-arena-navigate, cube-single-play)
#   x 5 seeds        (0-4)
#   = 30 jobs total.
#
# Same per-run sbatch bodies as every other online-controller sweep
# (run_dds_online_controller_seed.sbatch / run_online_crl_skill_controller_seed.sbatch),
# which read the USE_TES env var (added 2026-09-14) -- passed explicitly through
# sbatch --export below so it does not depend on the submitter's shell. All other
# flags are those scripts' fixed defaults: RLPD on with each checkpoint's OWN offline
# dataset, skill_commitment_k=10, 1M env steps.
#
# TES_INIT_FRAC defaults to 1.0 = the paper's own initialisation H_0 = log|A|
# (the constant-target runs in this repo use 0.9; with TES that number is only
# where the schedule starts).
#
# Checkpoints (all under ckpts/final/), same cells as submit_pmt_asoc_cube_entropy_sweep.sh:
#   dds/{pointmaze-teleport-navigate, antsoccer-arena-navigate, cube-single-play}  -- single sd000_* run each
#   empowerment_final/{pointmaze-teleport-navigate, antsoccer-arena-navigate}       -- single sd000_* run each
#   empowerment_final/cube-single-play/sd000_s_38624008.0.20260908_013305
#       -- explicit: this env dir holds FIVE sd000_* runs; the user chose this one
#          (2026-09-13, re-confirmed 2026-09-14).
#
# Online env per env family (deterministic, noise-free task sets):
#   pointmaze-teleport -> pointmaze-teleport-sparse-online-v0  (horizon 1000)
#   antsoccer-arena     -> antsoccer-arena-center-online-v0    (registered 1000, overridden to 500
#                         like every other online run in this repo)
#   cube-single         -> cube-single-center-online-v0        (horizon 200)
#
# Results land under <SKILL_CKPT>/online_controller_tes/rlpd/ -- a tree separate from
# every prior sweep's <SKILL_CKPT>/online_controller*/, so nothing existing is touched.
#
# This ONLY submits jobs. Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_pmt_asoc_cube_tes_seeds.sh
# Overrides:
#   DRY_RUN=1                         print the sbatch commands without submitting
#   SEEDS="0 1"                       a different seed set
#   CKPT_KEYS="dds_cube_sgl emp_cube_sgl"   a subset of the six (family, env) cells
#   TES_INIT_FRAC=0.9                 initial target = frac * log(num_skills)
#   TES_PATIENCE=1000                 override agent.tes_patience (default 500)
#   TIME_LIMIT=16:00:00               sbatch --time override
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

DDS_SBATCH=scripts/slurm/run_dds_online_controller_seed.sbatch
EMP_SBATCH=scripts/slurm/run_online_crl_skill_controller_seed.sbatch
LOG_DIR=logs/slurm/pmt_asoc_cube_tes
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
TIME_LIMIT=${TIME_LIMIT:-16:00:00}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TES_INIT_FRAC=${TES_INIT_FRAC:-1.0}
SAVE_SUBDIR=${SAVE_SUBDIR:-online_controller_tes}
EXPORT="ALL,USE_TES=1"
if [[ -n "${TES_PATIENCE:-}" ]]; then EXPORT="$EXPORT,TES_PATIENCE=$TES_PATIENCE"; fi

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
# cube's window-labelling / RLPD pass over its ~300MB dataset is the memory peak
# (as in submit_dds_online_controller_cube_seeds.sh): more headroom, no 16GB A4000s.
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

# (a) an exact leaf checkpoint dir (flags.json directly), or (b) an env dir holding
# exactly one sd000_* run.
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

    NUM_SKILLS=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['agent']['num_skills'])" "$SKILL_CKPT")
    [[ "$NUM_SKILLS" == "50" ]] || { echo "ERROR: $SKILL_CKPT has num_skills=$NUM_SKILLS, expected 50" >&2; exit 1; }

    OFFLINE_DATASET=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi

    ENV_NAME=${ONLINE_ENVS[$c]}
    EPISODE_LENGTH=${EPISODE_LENGTHS[$c]}
    read -r -a EXTRA_FLAGS <<< "${EXTRA_SBATCH_FLAGS[$c]}"
    echo "# $KEY -> $SKILL_CKPT (env=$ENV_NAME, rlpd dataset $OFFLINE_DATASET, num_skills=$NUM_SKILLS)"

    for SEED in $SEEDS; do
        JOB_NAME="tesctrl_${KEY}_s${SEED}"
        OUT="$LOG_DIR/${JOB_NAME}_%j.log"
        cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$TIME_LIMIT" --export="$EXPORT"
             "${EXTRA_FLAGS[@]}"
             "${SBATCHES[$c]}" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "" "$EPISODE_LENGTH" "$SAVE_SUBDIR" "$TES_INIT_FRAC")
        echo "${cmd[@]}"
        if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs (use_tes=1, init frac $TES_INIT_FRAC, save subdir $SAVE_SUBDIR)"
