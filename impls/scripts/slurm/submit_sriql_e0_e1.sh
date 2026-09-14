#!/usr/bin/env bash
# Submit E0 + E1 of the SR-IQL proposal (impls/docs/grounded_option_qi_proposal.md, v3) for the
# two K=50 empowerment_final checkpoints antmaze-medium-navigate and antsoccer-arena-navigate,
# to the rnn.ist.berkeley.edu Slurm cluster. Per checkpoint:
#
#   E0  1 job   run_gciql_flat_seed.sbatch      flat gciql (upper reference) + Q-selector eval
#   E1  12 jobs run_sriql_seed.sbatch           kernel {gauss, hard} x alpha {0, 1, 10} x expectile {0.7, 0.9}
#
# so 26 jobs in total. Everything is saved INSIDE the skill checkpoint's run folder:
#   <SKILL_CKPT>/gciql_flat/...        <SKILL_CKPT>/sriql_sweep/<kernel>_a<alpha>_e<expectile>/...
#
# alpha is the weight of log c(z|s) (coverage) in the selector; Q values live on a ~[-100, 0]
# scale (gc_negative rewards, discount 0.99) and log c differences are a few nats, so {1, 10}
# are the values that can matter and 0 is the ablation.
#
# antmaze-medium-navigate holds two sd000_* runs (k=50 and k=15); the k=50 one is selected by
# its `_k50_` suffix. antsoccer-arena-navigate holds exactly one.
#
# Run from the rnn login node:   bash scripts/slurm/submit_sriql_e0_e1.sh
# Overrides: DRY_RUN=1 | KERNELS="gauss" | ALPHAS="0 1" | EXPECTILES="0.9" | SEEDS="0 1"
#            SKIP_E0=1 | SKIP_E1=1 | CKPT_DIRS="antsoccer-arena-navigate"
#            KERNEL_SIGMA=0.2  fixed Gaussian width (exported to the sbatch; run dir tag gauss0.2_*)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

LOG_DIR=logs/slurm/sriql
EMP_ROOT=${EMP_ROOT:-ckpts/final/empowerment_final}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SKIP_E0=${SKIP_E0:-0}
SKIP_E1=${SKIP_E1:-0}
KERNELS=${KERNELS:-"gauss hard"}
ALPHAS=${ALPHAS:-"0 1 10"}
EXPECTILES=${EXPECTILES:-"0.7 0.9"}
SEEDS=${SEEDS:-"0"}

ALL_CKPT_DIRS=(antmaze-medium-navigate antsoccer-arena-navigate)
ALL_CKPT_TAGS=(amaze_nav asoc_nav)
read -r -a WANTED <<< "${CKPT_DIRS:-${ALL_CKPT_DIRS[*]}}"

resolve_ckpt() {  # env dir -> the K=50 skill run dir
    local env_dir=$1
    local matches=("$EMP_ROOT/$env_dir"/sd000_*/)
    if (( ${#matches[@]} > 1 )); then
        local k50=()
        for m in "${matches[@]}"; do [[ "$m" == *_k50_* ]] && k50+=("$m"); done
        matches=("${k50[@]}")
    fi
    if (( ${#matches[@]} != 1 )) || [[ ! -d "${matches[0]}" ]]; then
        echo "ERROR: expected exactly one K=50 sd000_* run under $EMP_ROOT/$env_dir, found ${#matches[@]}" >&2
        return 1
    fi
    echo "${matches[0]%/}"
}

n=0
submit() {
    echo "$*"
    if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
    n=$((n + 1))
}

for w in "${WANTED[@]}"; do
    TAG=""
    for i in "${!ALL_CKPT_DIRS[@]}"; do [[ "${ALL_CKPT_DIRS[$i]}" == "$w" ]] && TAG=${ALL_CKPT_TAGS[$i]}; done
    [[ -n "$TAG" ]] || { echo "ERROR: unknown checkpoint dir '$w' (known: ${ALL_CKPT_DIRS[*]})" >&2; exit 1; }
    SKILL_CKPT=$(resolve_ckpt "$w")
    [[ -f "$SKILL_CKPT/flags.json" ]] || { echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; exit 1; }
    compgen -G "$SKILL_CKPT/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $SKILL_CKPT" >&2; exit 1; }
    ENV_NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    K=$(python -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['agent']['num_skills'])" "$SKILL_CKPT")
    [[ "$K" == "50" ]] || { echo "ERROR: $SKILL_CKPT has num_skills=$K, expected 50" >&2; exit 1; }
    if [[ ! -f "$DATASET_DIR/$ENV_NAME.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$ENV_NAME.npz is missing; these jobs will fail on a compute node." >&2
    fi
    echo "== $w -> $SKILL_CKPT ($ENV_NAME, K=$K)"

    for SEED in $SEEDS; do
        if [[ "$SKIP_E0" != "1" ]]; then
            JOB="gciqlflat_${TAG}_s${SEED}"
            submit sbatch --job-name="$JOB" --output="$LOG_DIR/${JOB}_%j.log" \
                scripts/slurm/run_gciql_flat_seed.sbatch "$SKILL_CKPT" "$SEED"
        fi
        if [[ "$SKIP_E1" != "1" ]]; then
            for KERNEL in $KERNELS; do for ALPHA in $ALPHAS; do for EXP in $EXPECTILES; do
                JOB="sriql_${KERNEL}${KERNEL_SIGMA:-}_a${ALPHA}_e${EXP}_${TAG}_s${SEED}"
                submit sbatch --export=ALL,KERNEL_SIGMA="${KERNEL_SIGMA:-}" --job-name="$JOB" --output="$LOG_DIR/${JOB}_%j.log" \
                    scripts/slurm/run_sriql_seed.sbatch "$SKILL_CKPT" "$KERNEL" "$ALPHA" "$EXP" "$SEED"
            done; done; done
        fi
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n jobs"
