#!/usr/bin/env bash
# Local (no Slurm) version of scripts/slurm/submit_supe_pipeline_cube_mg_amz_all.sh, as run on dgx6 from
# 2026-09-23: the same two stages and the same 30 online runs (cube_mg + amz_all; plain SUPE and SUPE +
# distill-to-rlpd at bonus_scale {10, 3} cube / {30, 10} antmaze; 5 seeds), directly on ONE GPU, using the code
# of the checkout this script lives in. Defaults = the dgx6 sweep that produced exp/supe_gc_crlutd_dgx6:
#   * goal-conditioned SUPE (GOAL_CONDITIONED=1) with the paper's RM / RND / minibatch / discount schedule
#     (PAPER_SCHEDULE=1), but the online CRL critic budget: UTD_RATIO=4 CRITIC_UPDATES=1 MINIBATCH=1024
#     (tag supe_gc_rlpd_paperaux_u4c1b1024). Unset those three for the paper's 20 x 256 critic steps.
#   * cells from scripts/slurm/supe_cells_dgx6.sh: supe_cells.sh with cube_mg on the dgx6 cube-single-play
#     estimator sd000_s_38579169.0.20260904_235556 (the rnn one, 38624008, is not on dgx6).
#   * results / checkpoints / logs under the MAIN checkout $MAIN (/home/ishir/ogbench/impls).
#
#   stage 1  both OPAL pretrains in parallel (cube-single-play-v0 kl 0.2, antmaze-medium-navigate-v0
#            kl 0.1; continuous, chunk 4, 1M steps -- the flags of run_supe_opal_pretrain.sbatch),
#            saved under $SUPE_OPAL_ROOT/<dataset>/. A dataset with a finished run there is reused.
#   stage 2  the 30 online runs through a pool of MAX_CONCURRENT (15) slots. A run becomes eligible
#            as soon as ITS dataset's OPAL job exits 0 (local stand-in for afterok); if that OPAL job
#            fails, its runs are dropped. Each run is scripts/slurm/run_supe_online_seed.sbatch executed
#            with bash (it runs the command directly when there is no SLURM_JOB_ID).
#            Order: plain SUPE first, then distill-to-rlpd. Distill runs whose estimator checkpoint is
#            missing are HELD (not failed) and start once it appears under $EMP_ROOT.
#
#   A finished run can hang at interpreter exit; one whose log shows its final `Saved to .../params_<TOTAL>.pkl`
#   is killed after HANG_GRACE (300) s and counted as done.
#
# Usage (from anywhere): GPU=4 setsid nohup bash impls/scripts/run_supe_pipeline_cube_mg_amz_all_local.sh > master.log 2>&1 < /dev/null &
# Overrides: REPO_DIR, GPU (0), MAX_CONCURRENT (15), SEEDS, CELLS, SUPE_OPAL_ROOT, EMP_ROOT, SAVE_ROOT,
#   LOG_DIR, CHUNK (4), POLL_SECS (60), plus any env var run_supe_online_seed.sbatch reads.
set -uo pipefail   # no -e: one run's failure must not kill the others

MAIN=${MAIN:-/home/ishir/ogbench/impls}   # results / checkpoints go in the main checkout (its code is untouched)
export REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}   # the checkout holding this script
# Goal-conditioned SUPE on the paper schedule (run_supe_online_seed.sbatch PAPER_SCHEDULE / GOAL_CONDITIONED).
export PAPER_SCHEDULE=${PAPER_SCHEDULE:-1} GOAL_CONDITIONED=${GOAL_CONDITIONED:-1}
# Critic budget = the online CRL runs (one 1024-row critic + actor step per env step); RM / RND stay on the paper
# schedule (once per macro step). Unset these three for the paper's 20 x 256 critic steps per env step.
export UTD_RATIO=${UTD_RATIO:-4} CRITIC_UPDATES=${CRITIC_UPDATES:-1} MINIBATCH=${MINIBATCH:-1024}
cd "$REPO_DIR/impls" || exit 1
# cube_mg -> the cube-single-play estimator on dgx6 (sd000_s_38579169), not supe_cells.sh's rnn run 38624008
export SUPE_CELLS_FILE=${SUPE_CELLS_FILE:-$REPO_DIR/impls/scripts/slurm/supe_cells_dgx6.sh}
source "$SUPE_CELLS_FILE"

export PYTHON=${PYTHON:-/home/ishir/miniconda3/envs/ogbench/bin/python}
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"   # this worktree's ogbench shadows the editable install
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export WANDB_KEY_FILE=${WANDB_KEY_FILE:-$HOME/.no_wandb_key_file}   # absent on purpose: wandb uses ~/.netrc
export TMPDIR=$HOME/tmp/supe_dgx6
export WANDB_DIR="$TMPDIR/wandb"
mkdir -p "$TMPDIR" "$WANDB_DIR"
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$MAIN/.jax_cache}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${MKL_NUM_THREADS:-3} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-3}
BASE_XLA_FLAGS=${XLA_FLAGS:-}

export SUPE_OPAL_ROOT=${SUPE_OPAL_ROOT:-$MAIN/ckpts/final/supe_opal}
export EMP_ROOT=${EMP_ROOT:-$MAIN/ckpts/final/empowerment}
export SAVE_ROOT=${SAVE_ROOT:-$MAIN/exp/supe_gc_crlutd_dgx6}
LOG_DIR=${LOG_DIR:-$MAIN/logs/supe_gc_crlutd_dgx6}
MAX_CONCURRENT=${MAX_CONCURRENT:-15}
POLL_SECS=${POLL_SECS:-60}
SEEDS=${SEEDS:-"0 1 2 3 4"}
CHUNK=${CHUNK:-4}
export K=${K:-$CHUNK}
mkdir -p "$LOG_DIR" "$SUPE_OPAL_ROOT"
declare -A CELL_ALPHAS=([cube_mg]="10 3" [amz_all]="30 10")
read -r -a CELL_LIST <<< "${CELLS:-cube_mg amz_all}"
[[ "$K" == "$CHUNK" ]] || { echo "ERROR: K=$K must equal CHUNK=$CHUNK" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }

log "repo=$REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD)) gpu=$CUDA_VISIBLE_DEVICES max_concurrent=$MAX_CONCURRENT"
log "save_root=$SAVE_ROOT opal_root=$SUPE_OPAL_ROOT emp_root=$EMP_ROOT logs=$LOG_DIR"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# ── stage 1: OPAL pretraining, one background job per distinct dataset ──
declare -A OPAL_PID=() OPAL_STATE=()   # dataset -> pid ; dataset -> running | ok | failed
for CELL in "${CELL_LIST[@]}"; do
    supe_cell "$CELL" || exit 1
    [[ -v OPAL_STATE[$CELL_DATASET] ]] && continue
    [[ -f "$OGBENCH_DATASET_DIR/$CELL_DATASET.npz" ]] || { echo "ERROR: $OGBENCH_DATASET_DIR/$CELL_DATASET.npz missing" >&2; exit 1; }
    # supe_opal_run accepts any params_*.pkl, so an interrupted run would pass: require the final checkpoint.
    if EXISTING=$(supe_opal_run "$SUPE_OPAL_ROOT" "$CELL_DATASET" 2>/dev/null) \
            && [[ -f "$EXISTING/params_${TRAIN_STEPS:-1000000}.pkl" ]]; then
        log "$CELL_DATASET: reusing the finished OPAL run $EXISTING"
        OPAL_STATE[$CELL_DATASET]=ok
        continue
    fi
    KL=0.1
    [[ "$CELL_DATASET" == cube-* || "$CELL_DATASET" == scene-* ]] && KL=0.2
    NAME="supe_opal_${CELL_DATASET%-v0}_c${CHUNK}"
    mkdir -p "$SUPE_OPAL_ROOT/$CELL_DATASET"
    log "$CELL_DATASET: starting OPAL pretraining (chunk $CHUNK, kl $KL) -> $LOG_DIR/$NAME.log"
    XLA_FLAGS="$BASE_XLA_FLAGS --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_$NAME" \
    "$PYTHON" -u main.py \
        --env_name="$CELL_DATASET" --seed=0 --save_dir="$SUPE_OPAL_ROOT/$CELL_DATASET" \
        --agent=agents/opal.py --agent.latent_type=continuous --agent.skill_dim=8 \
        --agent.kl_coef="$KL" --agent.chunk_size="$CHUNK" --agent.sequence_length="$CHUNK" \
        --agent.batch_size=256 --train_steps="${TRAIN_STEPS:-1000000}" \
        --log_interval=5000 --eval_interval=250000 --eval_episodes=5 \
        --save_interval=100000 --video_episodes=0 > "$LOG_DIR/$NAME.log" 2>&1 &
    OPAL_PID[$CELL_DATASET]=$!
    OPAL_STATE[$CELL_DATASET]=running
done

# ── stage 2 queue: "cell|dataset|mode|alpha|seed", plain SUPE first ──
QUEUE=()
for ARM_KIND in plain distill; do
    for CELL in "${CELL_LIST[@]}"; do
        supe_cell "$CELL" || exit 1
        if [[ "$ARM_KIND" == plain ]]; then ARMS=("none|1.0")
        else ARMS=(); for A in ${CELL_ALPHAS[$CELL]:?no bonus scales for $CELL}; do ARMS+=("distill-to-rlpd|$A"); done; fi
        for ARM in "${ARMS[@]}"; do
            for SEED in $SEEDS; do QUEUE+=("$CELL|$CELL_DATASET|$ARM|$SEED"); done
        done
    done
done
log "queued ${#QUEUE[@]} online runs"

declare -A RUN_PID=()   # name -> pid
declare -A FINAL_SEEN=()   # name -> epoch seconds its final checkpoint was first seen in the log
TOTAL=${TOTAL_ENV_STEPS:-1000000}
HANG_GRACE=${HANG_GRACE:-300}
declare -A HELD_WARNED=()
n_ok=0; n_fail=0
while :; do
    for DS in "${!OPAL_PID[@]}"; do
        pid=${OPAL_PID[$DS]}
        if ! kill -0 "$pid" 2>/dev/null; then
            if wait "$pid"; then OPAL_STATE[$DS]=ok; log "$DS: OPAL finished OK"
            else OPAL_STATE[$DS]=failed; log "$DS: OPAL FAILED (see $LOG_DIR) -- its online runs are dropped"; fi
            unset "OPAL_PID[$DS]"
        fi
    done
    # A finished run can hang at interpreter exit: once its final checkpoint (params_<step >= TOTAL>) is in its
    # log, give it HANG_GRACE seconds, then kill it and count it as done.
    for NAME in "${!RUN_PID[@]}"; do
        pid=${RUN_PID[$NAME]}
        kill -0 "$pid" 2>/dev/null || continue
        # Only main_online's own saves ("Saved to <run dir>/params_<step>.pkl"); the log also prints the OPAL
        # checkpoint path, which ends in params_1000000.pkl.
        last=$(grep -oE '^Saved to .*/params_[0-9]+\.pkl' "$LOG_DIR/$NAME.log" 2>/dev/null | tail -1 | grep -oE 'params_[0-9]+' | tr -dc 0-9)
        if [[ -n "$last" ]] && (( last >= TOTAL )); then
            [[ -v FINAL_SEEN[$NAME] ]] || FINAL_SEEN[$NAME]=$(date +%s)
            if (( $(date +%s) - FINAL_SEEN[$NAME] > HANG_GRACE )); then
                log "hung   $NAME after its final checkpoint: killing"
                pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null
                unset "RUN_PID[$NAME]"; n_ok=$((n_ok + 1)); log "done   $NAME"
            fi
        fi
    done
    for NAME in "${!RUN_PID[@]}"; do
        pid=${RUN_PID[$NAME]}
        if ! kill -0 "$pid" 2>/dev/null; then
            if wait "$pid"; then n_ok=$((n_ok + 1)); log "done   $NAME"
            else n_fail=$((n_fail + 1)); log "FAILED $NAME (see $LOG_DIR/$NAME.log)"; fi
            unset "RUN_PID[$NAME]"
        fi
    done
    REMAINING=(); held=0
    for J in "${QUEUE[@]}"; do
        IFS='|' read -r CELL DS MODE ALPHA SEED <<< "$J"
        case "${OPAL_STATE[$DS]}" in
            failed) log "drop   $CELL $MODE$ALPHA s$SEED (OPAL for $DS failed)"; continue ;;
            running) REMAINING+=("$J"); continue ;;
        esac
        if [[ "$MODE" != none ]]; then
            supe_cell "$CELL"
            if [[ ! -f "$EMP_ROOT/$CELL_EMP_RUN/flags.json" ]]; then
                REMAINING+=("$J"); held=$((held + 1))
                [[ -v HELD_WARNED[$CELL] ]] || { log "hold   $CELL distill runs: no estimator at $EMP_ROOT/$CELL_EMP_RUN"; HELD_WARNED[$CELL]=1; }
                continue
            fi
        fi
        if (( ${#RUN_PID[@]} >= MAX_CONCURRENT )); then REMAINING+=("$J"); continue; fi
        NAME="supe_${CELL}_${MODE}${ALPHA}_s${SEED}"
        XLA_FLAGS="$BASE_XLA_FLAGS" TMPDIR="$TMPDIR" \
            bash scripts/slurm/run_supe_online_seed.sbatch "$CELL" "$SEED" "$MODE" "$ALPHA" "" > "$LOG_DIR/$NAME.log" 2>&1 &
        RUN_PID[$NAME]=$!
        log "start  $NAME (pid $!, ${#RUN_PID[@]}/$MAX_CONCURRENT running)"
        sleep 5   # stagger startup (dataset load + jit)
    done
    QUEUE=("${REMAINING[@]}")
    if (( ${#RUN_PID[@]} == 0 && ${#OPAL_PID[@]} == 0 )) && (( ${#QUEUE[@]} == held )); then
        (( held > 0 )) && { log "stopping with $held distill runs held (estimator missing):"; printf '  %s\n' "${QUEUE[@]}"; }
        break
    fi
    sleep "$POLL_SECS"
done
log "pipeline finished: $n_ok online runs OK, $n_fail failed"
