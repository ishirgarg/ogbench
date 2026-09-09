#!/usr/bin/env bash
# Submit the OFFLINE empowerment_skill high-level controller CHUNK-HORIZON sweep to the
# rnn.ist.berkeley.edu Slurm cluster: 4 horizons x 4 alphas -> 16 jobs on the single
# antsoccer-arena-navigate checkpoint, one sbatch job per (horizon, alpha).
#
# Grid: H in {5, 15, 20, 50} x alpha in {0.3, 1, 3, 10}.
#   H=10 is DELIBERATELY ABSENT -- it already exists at all four alphas under
#   <SKILL_CKPT>/controller_awr_sweep/alpha<A>/ from submit_empowerment_controller_alpha_sweep.sh
#   (all four scored 0.000 overall success at 1M steps). Read that arm as the H=10 row
#   of this sweep; re-running it would only cost GPU hours.
#
# H is one number with three roles -- label window, option length, and the number of env
# steps the frozen low-level policy is executed for before the high level re-selects.
# --agent.chunk_horizon moves all three (sequence_length via a shared FieldReference,
# skill_horizon via its None default). See run_empowerment_controller_horizon_seed.sbatch
# for the full rationale and the plot_relabel_diagnosis.py measurements that motivate it.
#
# Everything else is held at the alpha sweep's values so the two are directly
# comparable: expectile 0.9, seed 0, 1M steps, 50 eval episodes, gciql high level over
# the K=50 skill indices, pretrained agent loaded read-only.
#
# Results land INSIDE the frozen run's own folder:
#   ckpts/final/empowerment_final/antsoccer-arena-navigate/sd000_*/
#       controller_horizon_sweep/h<H>_alpha<A>/OGBench/Debug/sd000_s_<jobid>.<ts>/
# so nothing under the checkpoint is overwritten and the H=10 arm stays where it is.
#
# This ONLY submits jobs -- it does not run any training itself. Run from the rnn login
# node, from this NAS checkout:
#   bash scripts/slurm/submit_empowerment_controller_horizon_sweep.sh
# Overrides:
#   DRY_RUN=1                 print the sbatch commands without submitting
#   HORIZONS="5 15"           a different horizon grid
#   ALPHAS="1"                a different alpha grid
#   SEEDS="0 1"               more than the single seed 0
#   CKPT_DIR=antsoccer-arena-stitch    a different env dir under ckpts/final/empowerment_final
#
# Prerequisites (checked below where possible):
#   * wandb credentials -- see the comment block in run_online_crl_seed.sbatch.
#   * The checkpoint's OFFLINE dataset must already be in /nas/ucb/ishirgarg/.ogbench/data
#     (compute nodes have no internet egress).
#
# Note on packing: each job asks for --gpus=1 and lets JAX grow memory on demand
# (XLA_PYTHON_CLIENT_PREALLOCATE=false), matching every other sbatch script here. How
# many of the 16 land on a card at once is the scheduler's decision, not this script's;
# to pack them by hand instead, run the same command lines with CUDA_VISIBLE_DEVICES set
# and no srun.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_empowerment_controller_horizon_seed.sbatch
LOG_DIR=logs/slurm/empowerment_controller_horizon
EMP_ROOT=${EMP_ROOT:-ckpts/final/empowerment_final}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
HORIZONS=${HORIZONS:-"5 15 20 50"}
ALPHAS=${ALPHAS:-"0.3 1 3 10"}
SEEDS=${SEEDS:-"0"}
CKPT_DIR=${CKPT_DIR:-antsoccer-arena-navigate}
TAG=${TAG:-asoc_nav}

# Resolve by glob so a re-rsync with a different job id still works; the env dir must hold
# exactly one sd000_* run. (antmaze-medium-* hold two -- a k=50 and a k=15 -- which is why
# they are not the default here.)
matches=("$EMP_ROOT/$CKPT_DIR"/sd000_*/)
if (( ${#matches[@]} != 1 )) || [[ ! -d "${matches[0]}" ]]; then
    echo "ERROR: expected exactly one sd000_* run under $EMP_ROOT/$CKPT_DIR, found ${#matches[@]}" >&2
    exit 1
fi
SKILL_CKPT=${matches[0]%/}
[[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }

# Offline training reads the checkpoint's own dataset; warn early if it isn't on the NAS.
ENV_NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
if [[ ! -f "$DATASET_DIR/$ENV_NAME.npz" ]]; then
    echo "WARNING: $DATASET_DIR/$ENV_NAME.npz is missing; these jobs will fail on a compute node." >&2
fi

echo "checkpoint: $SKILL_CKPT  ($ENV_NAME)"
echo "grid:       H in {$HORIZONS} x alpha in {$ALPHAS} x seed in {$SEEDS}"
echo "results ->  \$SKILL_CKPT/controller_horizon_sweep/h<H>_alpha<A>/"

n_submitted=0
for H in $HORIZONS; do
    if [[ "$H" == "10" ]]; then
        echo "NOTE: skipping H=10 -- already trained at all four alphas under" \
             "$SKILL_CKPT/controller_awr_sweep/. Pass HORIZONS with 10 removed, or" \
             "delete this guard if you really want a fresh H=10 run." >&2
        continue
    fi
    for ALPHA in $ALPHAS; do
        for SEED in $SEEDS; do
            JOB_NAME="empctrl_h${H}_a${ALPHA}_${TAG}_s${SEED}"
            OUT="$LOG_DIR/${JOB_NAME}_%j.log"
            cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT"
                 "$SBATCH_SCRIPT" "$SKILL_CKPT" "$H" "$ALPHA" "$SEED")
            echo "${cmd[@]}"
            if [[ "$DRY_RUN" != "1" ]]; then
                "${cmd[@]}"
            fi
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs"
