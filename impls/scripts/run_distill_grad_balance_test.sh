#!/usr/bin/env bash
# Short LOCAL gradient-balance test for the distilled future-max empowerment bonus
# (agents/online_crl.py, add_explore=distill | distill-to-rlpd), all jobs packed on one GPU.
#
# Every run has agent.bonus_grad_diagnostics=True, which logs the actor-parameter gradient norm
# of the CRL term and of the bonus term AT bonus_scale = 1 (independent of the run's own
# bonus_scale) under training/bonus_grad/*; scale_equal_param = ||g_crl|| / ||g_bonus|| is the
# bonus_scale ("alpha") that equalises the two. SCALES="0" measures that ratio under the plain
# CRL policy (E' is trained but the actor never reads it); any other scale measures it closed loop.
#
# Cells (env / RLPD dataset / 50-skill estimator under ckpts/final/empowerment_final, main checkout):
#   asoc_ctr  antsoccer-arena-center-online-v0     antsoccer-arena-navigate-v0    (episode_length 500)
#   pmt_ctr   pointmaze-teleport-center-online-v0  pointmaze-teleport-stitch-v0
#   cube_sgl  cube-single-center-online-v0         cube-single-play-v0
#
# Usage (from impls/):  GPU=0 SCALES="0 1" MODES="distill distill-to-rlpd" bash scripts/run_distill_grad_balance_test.sh
# Per-cell scales: SCALES_asoc_ctr="30" SCALES_pmt_ctr="20" ... override SCALES for that cell.
# Results: $OUT_ROOT/OGBench/Debug/sd00<seed>_s_dgb-<cell>_<mode>_a<scale>.<timestamp>/{train,eval}.csv
# Summarise with: python scripts/summarize_distill_grad_balance.py
set -euo pipefail
cd "$(dirname "$0")/.."

MAIN=/nas/ucb/ishirgarg/ogbench/impls   # checkpoints + results live in the main checkout (outlive the worktree)
PYTHON=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python
GPU=${GPU:-0}
SEED=${SEED:-0}
SCALES=${SCALES:-"0 1"}
MODES=${MODES:-"distill distill-to-rlpd"}
CELLS=${CELLS:-"asoc_ctr pmt_ctr cube_sgl"}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-60000}
DISTILL_TARGET=${DISTILL_TARGET:-episode_max}   # episode_max (agent default) | future_max; use a separate OUT_ROOT per target
OUT_ROOT=${OUT_ROOT:-$MAIN/exp/distill_grad_test}
LOG_DIR=${LOG_DIR:-$MAIN/logs/distill_grad_test}
mkdir -p "$LOG_DIR"

export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}   # EGL cannot open a headless display on this box (cube needs dm_control)
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # mandatory for packing several runs on one GPU
export TMPDIR=/nas/ttl=60d/ishirgarg/tmp WANDB_DIR=/nas/ttl=60d/ishirgarg/tmp   # local root disk is full
mkdir -p "$TMPDIR"

EMP_ROOT=$MAIN/ckpts/final/empowerment_final
declare -A ENV=([asoc_ctr]=antsoccer-arena-center-online-v0 [pmt_ctr]=pointmaze-teleport-center-online-v0 [cube_sgl]=cube-single-center-online-v0)
declare -A OFFLINE=([asoc_ctr]=antsoccer-arena-navigate-v0 [pmt_ctr]=pointmaze-teleport-stitch-v0 [cube_sgl]=cube-single-play-v0)
declare -A EMP=(
    [asoc_ctr]=$EMP_ROOT/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836
    [pmt_ctr]=$EMP_ROOT/pointmaze-teleport-stitch/sd000_s_38390675.0.20260901_154836
    [cube_sgl]=$EMP_ROOT/cube-single-play/sd000_s_38624008.0.20260908_013305
)
declare -A EP_LEN=([asoc_ctr]=500 [pmt_ctr]="" [cube_sgl]="")

pids=()
for CELL in $CELLS; do
    CELL_SCALES_VAR="SCALES_${CELL}"
    CELL_SCALES=${!CELL_SCALES_VAR:-$SCALES}
    for MODE in $MODES; do
        for SCALE in $CELL_SCALES; do
            NAME="dgb-${CELL}_${MODE}_a${SCALE}"
            EP_FLAG=()
            if [[ -n "${EP_LEN[$CELL]}" ]]; then EP_FLAG=(--episode_length="${EP_LEN[$CELL]}"); fi
            echo "launching $NAME on GPU $GPU -> $LOG_DIR/$NAME.log"
            # SLURM_JOB_ID only names the run (get_exp_name), keeping same-second launches apart.
            SLURM_JOB_ID="$NAME" CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u main_online.py \
                --env_name="${ENV[$CELL]}" \
                --seed="$SEED" \
                --agent=agents/online_crl.py \
                --total_env_steps="$TOTAL_ENV_STEPS" \
                "${EP_FLAG[@]}" \
                --offline_dataset="${OFFLINE[$CELL]}" \
                --agent.emp_checkpoint_path="${EMP[$CELL]}" \
                --agent.emp_entropy_target=False \
                --agent.add_explore="$MODE" \
                --agent.distill_target="$DISTILL_TARGET" \
                --agent.bonus_scale="$SCALE" \
                --agent.bonus_grad_diagnostics=True \
                --log_interval=2500 \
                --eval_interval=20000 \
                --eval_episodes=20 \
                --video_episodes=0 \
                --save_interval=10000000 \
                --save_dir="$OUT_ROOT" \
                > "$LOG_DIR/$NAME.log" 2>&1 &
            pids+=($!)
            sleep 2
        done
    done
done

echo "waiting on ${#pids[@]} jobs..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit $fail
