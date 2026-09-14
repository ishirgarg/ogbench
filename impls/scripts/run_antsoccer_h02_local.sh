#!/usr/bin/env bash
# Local (non-Slurm) online skill-controller runs on THIS machine's GPUs with a LOW skill-entropy
# floor: target_entropy_frac=0.2 instead of the 0.9 default.
#
# Why: the default sets H_target = 0.9 * log K = 3.521 nats for K=50, i.e. 90% of the maximum
# possible categorical entropy, and the measured actor entropy sits
# exactly on it in every finished run -- the constraint is active and holds the policy near uniform.
# At 3.521 nats the largest share any one skill can take is 0.236, so a selected skill survives ~13
# env steps on average at skill_commitment_k=10, while the useful antsoccer skills need 90-200 steps
# of uninterrupted execution. frac=0.2 gives H_target = 0.782 nats, top-skill share up to ~0.89
# and a mean committed run of ~90 env steps. It is a FLOOR, not a setpoint: the dual is non-negative,
# so if the critic prefers more exploration the temperature decays and the constraint stops acting.
#
# Grid: 2 envs x 2 skill families x 2 seeds = 8 runs, navigate checkpoints only, both K=50, both in
# ckpts/final:
#   env   antsoccer-arena-center-online-v0   and   antsoccer-arena-corner-online-v0   (horizon 500)
#   ckpt  ckpts/final/dds/antsoccer-arena-navigate/sd000_*
#         ckpts/final/empowerment_final/antsoccer-arena-navigate/sd000_*
#   seeds 0 1
#
# Scheduling: one lane = one (env, ckpt) pair running its 2 seeds SEQUENTIALLY. At most NUM_GPUS
# lanes run concurrently, one lane per GPU, so at most 2 training processes exist at any time. The 4
# lanes are dealt round-robin over the GPUs, so each GPU runs 2 lanes back to back = 4 runs.
#
# Results go to <ckpt>/online_controller_h02/rlpd (center) and
# <ckpt>/online_controller_corner_h02/rlpd (corner) -- separate trees from the finished frac=0.9
# sweeps in online_controller/ and online_controller_corner/, so nothing on disk is overwritten.
#
# RLPD is on for every run, with the offline dataset defaulting to the checkpoint's own training data.
# Every other flag matches scripts/slurm/run_online_crl_skill_controller_seed.sbatch exactly, so these
# are comparable to the finished sweeps apart from the entropy floor.
#
# Usage:
#   bash scripts/run_antsoccer_h02_local.sh              # GPUs "0 1"
#   GPUS="3 4" bash scripts/run_antsoccer_h02_local.sh
#   DRY_RUN=1 bash scripts/run_antsoccer_h02_local.sh    # print the commands only
#   LANES="corner:dds corner:emp" bash scripts/run_antsoccer_h02_local.sh
# Launch it detached so an ssh drop cannot kill it:
#   setsid nohup bash scripts/run_antsoccer_h02_local.sh < /dev/null > logs/h02_driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."   # -> impls/

PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
GPUS=${GPUS:-"0 1"}
DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1"}
ENT_FRAC=${ENT_FRAC:-0.2}
EPISODE_LENGTH=${EPISODE_LENGTH:-500}
TOTAL_STEPS=${TOTAL_STEPS:-1000000}
LOG_DIR=${LOG_DIR:-logs/h02_local}
# ~17 GB per run; the cards are 48 GB. Set this explicitly rather than trusting the default, because
# these GPUs are shared and other users' usage swings by tens of GB.
MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}
mkdir -p "$LOG_DIR"

DDS_CKPT=$(ls -d ckpts/final/dds/antsoccer-arena-navigate/sd000_*/ 2>/dev/null | head -1)
EMP_CKPT=$(ls -d ckpts/final/empowerment_final/antsoccer-arena-navigate/sd000_*/ 2>/dev/null | head -1)
DDS_CKPT=${DDS_CKPT%/}
EMP_CKPT=${EMP_CKPT%/}

ALL_LANES=${LANES:-"center:dds center:emp corner:dds corner:emp"}

lane_env ()  { case "$1" in center) echo antsoccer-arena-center-online-v0;; corner) echo antsoccer-arena-corner-online-v0;; esac; }
lane_sub ()  { case "$1" in center) echo online_controller_h02;;            corner) echo online_controller_corner_h02;; esac; }
lane_ckpt () { case "$1" in dds) echo "$DDS_CKPT";; emp) echo "$EMP_CKPT";; esac; }

# ── Preflight ────────────────────────────────────────────────────────────────
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
fail=0
for c in "$DDS_CKPT" "$EMP_CKPT"; do
    [[ -n "$c" && -d "$c" ]] || { echo "ERROR: checkpoint dir not found" >&2; fail=1; continue; }
    [[ -f "$c/flags.json" ]] || { echo "ERROR: missing $c/flags.json" >&2; fail=1; }
    compgen -G "$c/params_*.pkl" > /dev/null || { echo "ERROR: no params_*.pkl in $c" >&2; fail=1; }
    K=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]+'/flags.json'))['agent'].get('num_skills'))" "$c")
    [[ "$K" == "50" ]] || { echo "ERROR: $c has num_skills=$K, expected 50" >&2; fail=1; }
    DS=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]+'/flags.json'))['env_name'])" "$c")
    [[ -f "$DATASET_DIR/$DS.npz" ]] || { echo "ERROR: RLPD dataset $DATASET_DIR/$DS.npz missing" >&2; fail=1; }
done
(( fail == 0 )) || exit 1

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=/nas/ucb/ishirgarg/.wandb_api_key
if [[ -r "$WANDB_KEY_FILE" ]]; then export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"; fi
export TMPDIR=${TMPDIR:-/nas/ttl=60d/ishirgarg/tmp}
mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OGBENCH_DATASET_DIR="$DATASET_DIR"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}

run_one () {
    local gpu=$1 lane=$2 seed=$3
    local kind=${lane%%:*} fam=${lane##*:}
    local env_name; env_name=$(lane_env "$kind")
    local sub;      sub=$(lane_sub "$kind")
    local ckpt;     ckpt=$(lane_ckpt "$fam")
    local save_dir="$ckpt/$sub/rlpd"
    local ds; ds=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]+'/flags.json'))['env_name'])" "$ckpt")
    local epoch; epoch=$("$PYTHON" -c "
import glob,os,re,sys
print(max(int(re.search(r'params_(\d+)\.pkl\$', os.path.basename(p)).group(1)) for p in glob.glob(os.path.join(sys.argv[1],'params_*.pkl'))))" "$ckpt")
    local log="$LOG_DIR/${kind}_${fam}_s${seed}.log"
    mkdir -p "$save_dir"

    echo "[gpu $gpu] START $kind/$fam seed=$seed env=$env_name ent_frac=$ENT_FRAC -> $log"
    if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
    # Per-process autotune cache: the shared NAS one is NOT concurrency-safe and kills jobs outright.
    CUDA_VISIBLE_DEVICES="$gpu" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
    XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_h02_${gpu}_${kind}_${fam}_${seed}" \
    "$PYTHON" -u main_online.py \
        --env_name="$env_name" \
        --seed="$seed" \
        --save_dir="$save_dir" \
        --agent=agents/online_crl_skill_controller.py \
        --agent.skill_checkpoint_path="$ckpt" \
        --agent.skill_restore_epoch="$epoch" \
        --agent.skill_commitment_k=10 \
        --agent.target_entropy_frac="$ENT_FRAC" \
        --total_env_steps="$TOTAL_STEPS" \
        --episode_length="$EPISODE_LENGTH" \
        --offline_dataset="$ds" \
        --log_interval=5000 \
        --eval_interval=20000 \
        --save_interval=100000 \
        --eval_episodes=20 \
        --video_episodes=0 \
        > "$log" 2>&1
    local rc=$?
    echo "[gpu $gpu] END   $kind/$fam seed=$seed rc=$rc"
    return $rc
}

# One worker per GPU; each drains its own lane list, seeds in order.
worker () {
    local gpu=$1; shift
    for lane in "$@"; do
        for seed in $SEEDS; do
            run_one "$gpu" "$lane" "$seed"
        done
    done
    echo "[gpu $gpu] all lanes done"
}

read -r -a GPU_ARR <<< "$GPUS"
read -r -a LANE_ARR <<< "$ALL_LANES"
n_gpu=${#GPU_ARR[@]}
declare -a assigned
for i in "${!LANE_ARR[@]}"; do
    g=$(( i % n_gpu ))
    assigned[$g]="${assigned[$g]:-} ${LANE_ARR[$i]}"
done

echo "GPUs: ${GPU_ARR[*]}   seeds: $SEEDS   ent_frac: $ENT_FRAC   total runs: $(( ${#LANE_ARR[@]} * $(wc -w <<< "$SEEDS") ))"
for g in "${!GPU_ARR[@]}"; do
    echo "  gpu ${GPU_ARR[$g]} lanes:${assigned[$g]}"
done

pids=()
for g in "${!GPU_ARR[@]}"; do
    # shellcheck disable=SC2086
    worker "${GPU_ARR[$g]}" ${assigned[$g]} &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "all workers finished (rc=$rc)"
exit $rc
