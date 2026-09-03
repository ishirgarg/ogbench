#!/usr/bin/env bash
# Local (non-Slurm) version of run_skill_bc_relabel_awr_sweep.sh.
#
# AWR-alpha sweep for the goal-conditioned skill controller
# (skill_bc_relabel_controller:gciql) on top of the two frozen k=50 antmaze
# checkpoints rsynced under impls/ckpts/empowerment_final. See the Slurm script
# for the rationale behind the {1, 3, 10} alpha grid and the pinned expectile.
#
# Sweep: 2 checkpoints x 3 alphas = 6 runs.
#   0-2 : antmaze-medium-navigate-v0  k=50  alpha in {1, 3, 10}
#   3-5 : antmaze-medium-stitch-v0    k=50  alpha in {1, 3, 10}
#
# Scheduling: runs are dealt round-robin over GPUS (default 3 GPUs), with at most
# PER_GPU concurrent runs on each GPU. XLA_PYTHON_CLIENT_MEM_FRACTION is set so
# PER_GPU processes fit on one card. Every run is launched immediately as long as
# it fits (6 runs / 3 GPUs = 2 per GPU by default); anything beyond the cap waits
# for a free slot on its GPU.
#
# Checkpoints go to
#   <SKILL_CKPT>/controller_awr_sweep/alpha<ALPHA>/OGBench/Debug/sd000_<ts>/
# so the pretrained params_*.pkl in each SKILL_CKPT is never touched.
#
# Run from impls/:  bash scripts/run_skill_bc_relabel_awr_sweep_local.sh
#   GPUS="0 1 2" PER_GPU=3 RUN_IDS="0 3"   to override the GPU set / cap / subset.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> impls/

PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
EMP_ROOT=${EMP_ROOT:-ckpts/empowerment_final}

read -r -a GPUS <<< "${GPUS:-0 1 2}"
PER_GPU=${PER_GPU:-3}
NGPU=${#GPUS[@]}

# ── Sweep definitions (parallel arrays) ──────────────────────────────────────
CKPT_A="$EMP_ROOT/antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001"
CKPT_B="$EMP_ROOT/antmaze-medium-stitch/sd000_s_37866313.0.20260821_030454_k50_s0.01_bc0.001"
SKILL_CKPTS=("$CKPT_A" "$CKPT_A" "$CKPT_A" "$CKPT_B" "$CKPT_B" "$CKPT_B")
ALPHAS=(1 3 10 1 3 10)
EXPECTILE=${EXPECTILE:-0.9}
SEED=${SEED:-0}
TRAIN_STEPS=${TRAIN_STEPS:-1000000}
read -r -a RUN_IDS <<< "${RUN_IDS:-0 1 2 3 4 5}"

# ── Environment ─────────────────────────────────────────────────────────────
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=egl
# PER_GPU JAX processes share a card: cap each one's preallocation.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-$(awk -v n="$PER_GPU" 'BEGIN{printf "%.2f", 0.9/n}')}
# Root fs (/tmp) runs at 100% on this box; keep ptxas scratch on the NAS.
export TMPDIR=${TMPDIR:-/nas/ucb/ishirgarg/tmp}
export JAX_COMPILATION_CACHE_DIR="$(pwd)/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$TMPDIR"

LOG_DIR=logs/skill_bc_relabel_awr_sweep
mkdir -p "$LOG_DIR"

# ── Per-run launcher ─────────────────────────────────────────────────────────
launch_run() {
    local IDX=$1 GPU=$2
    local SKILL_CKPT=${SKILL_CKPTS[$IDX]}
    local ALPHA=${ALPHAS[$IDX]}

    if [[ ! -f "$SKILL_CKPT/flags.json" ]]; then
        echo "ERROR: missing $SKILL_CKPT/flags.json" >&2; return 1
    fi
    local ENV_NAME SKILL_EPOCH
    ENV_NAME=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    SKILL_EPOCH=$($PYTHON -c "
import glob, os, re, sys
print(max(int(re.search(r'params_(\d+)\.pkl$', os.path.basename(p)).group(1))
          for p in glob.glob(os.path.join(sys.argv[1], 'params_*.pkl'))))
" "$SKILL_CKPT")
    if [[ -z "$ENV_NAME" || -z "$SKILL_EPOCH" ]]; then
        echo "could not read env_name / latest epoch from $SKILL_CKPT" >&2; return 1
    fi

    local SAVE_DIR="$SKILL_CKPT/controller_awr_sweep/alpha${ALPHA}"
    mkdir -p "$SAVE_DIR"
    local LOG="$LOG_DIR/idx${IDX}_${ENV_NAME}_alpha${ALPHA}_s${SEED}.log"

    echo "[gpu $GPU] IDX=$IDX  ENV=$ENV_NAME  ALPHA=$ALPHA  EXPECTILE=$EXPECTILE  SEED=$SEED"
    echo "          ckpt=$SKILL_CKPT  epoch=$SKILL_EPOCH"
    echo "          save_dir=$SAVE_DIR  log=$LOG"

    # `--agent.base.actor_loss` is deliberately not passed: _base_config pins it to
    # 'awr' because the option MDP's action space is discrete.
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main.py \
        --env_name="$ENV_NAME" \
        --save_dir="$SAVE_DIR" \
        --agent=agents/skill_bc_relabel_controller.py:gciql \
        --agent.skill_checkpoint_path="$SKILL_CKPT" \
        --agent.skill_restore_epoch="$SKILL_EPOCH" \
        --agent.base.expectile=$EXPECTILE \
        --agent.base.alpha=$ALPHA \
        --seed=$SEED \
        --train_steps=$TRAIN_STEPS \
        --log_interval=5000 \
        --eval_interval=100000 \
        --save_interval=100000 \
        --eval_episodes=50 \
        --video_episodes=0 \
        > "$LOG" 2>&1 &
    LAST_PID=$!
}

# ── Scheduler: round-robin over GPUs, at most PER_GPU live runs per GPU ──────
# One background "lane" per GPU runs its assigned indices, keeping up to PER_GPU
# of them alive at once. Lanes themselves run concurrently.
gpu_lane() {
    local GPU=$1; shift
    local -a queue=("$@")
    local -a live=()
    local fail=0
    for IDX in "${queue[@]}"; do
        while (( ${#live[@]} >= PER_GPU )); do
            local -a still=()
            for p in "${live[@]}"; do
                if kill -0 "$p" 2>/dev/null; then still+=("$p"); else wait "$p" || fail=1; fi
            done
            live=("${still[@]}")
            (( ${#live[@]} >= PER_GPU )) && sleep 30
        done
        launch_run "$IDX" "$GPU"
        live+=("$LAST_PID")
        # exp_name has 1-second resolution: stagger so no two runs share a dir name.
        sleep 5
    done
    for p in "${live[@]}"; do wait "$p" || fail=1; done
    return $fail
}

declare -A ASSIGN
for k in "${!RUN_IDS[@]}"; do
    IDX=${RUN_IDS[$k]}
    if (( IDX < 0 || IDX >= ${#SKILL_CKPTS[@]} )); then
        echo "ERROR: run id $IDX out of range 0..$(( ${#SKILL_CKPTS[@]} - 1 ))" >&2; exit 1
    fi
    GPU=${GPUS[$((k % NGPU))]}
    ASSIGN[$GPU]+="$IDX "
done

lane_pids=()
for GPU in "${GPUS[@]}"; do
    [[ -n "${ASSIGN[$GPU]:-}" ]] || continue
    read -r -a ids <<< "${ASSIGN[$GPU]}"
    echo "GPU $GPU lane: run ids ${ids[*]} (max $PER_GPU concurrent)"
    gpu_lane "$GPU" "${ids[@]}" &
    lane_pids+=($!)
    sleep 2   # keep lanes' first launches on different seconds
done

echo "waiting on ${#lane_pids[@]} GPU lanes (${#RUN_IDS[@]} runs)..."
fail=0
for p in "${lane_pids[@]}"; do wait "$p" || fail=1; done
echo "ALL DONE (fail=$fail)"
exit $fail
