#!/usr/bin/env bash
# One SUPE online run (agents/supe.py over a frozen continuous OPAL checkpoint, RLPD with the cell's
# offline dataset, learned reward/termination relabelling, RND bonus) on the rnn.ist.berkeley.edu Slurm
# cluster, laid out so it is directly comparable to our flat CRL + RLPD runs and the distilled
# empowerment-bonus sweeps: the same cells (scripts/slurm/supe_cells.sh), the same 1M env steps,
# 100 eval episodes every 25k env steps, and the same save layout $SAVE_ROOT/<cell>/<tag>/OGBench/Debug/<run>.
#
# Usage (normally via scripts/slurm/submit_supe_online_seeds.sh):
#   sbatch --job-name=<name> --output=<log> scripts/slurm/run_supe_online_seed.sbatch \
#       <CELL> <SEED> [ADD_EXPLORE=none|distill|distill-to-rlpd] [BONUS_SCALE=1.0] [ANNEAL_FRAC=""]
#
# ADD_EXPLORE=distill | distill-to-rlpd adds the distilled empowerment bonus (agents/online_crl.py's
# recipe, here on E'(s, u): the same estimator run, bonus_scale, distill_target and annealing as the
# flat sweeps) -- SUPE + our exploration term, one flag. Tags mirror scripts/run_online_crl.sh:
#   supe_rlpd               plain SUPE
#   supe_rlpd_ed10          + distill (online rows), bonus_scale 10
#   supe_rlpd_edrlpd30      + distill-to-rlpd, bonus_scale 30
#   supe_rlpd_ann0.5_ed10   annealed to 0 by 0.5 of the env steps (ANNEAL_FRAC)
#   ..._fut                 DISTILL_TARGET=future_max
#   supe_norlpd[...]        RLPD=0 (SUPE's "Online w/ Trajectory Skills" ablation: no offline data,
#                           USE_RND_OFFLINE irrelevant)
#   ..._nornd               USE_RND_OFFLINE=0 USE_RND_ONLINE=0
#   ..._k<K>                K (skill_commitment_k) differs from the default 4
#
# Env vars: SUPE_OPAL_ROOT (ckpts/final/supe_opal in the MAIN checkout), SUPE_OPAL_RUN (explicit run
#   dir), EMP_ROOT (ckpts/final/empowerment_final, distill arms), SAVE_ROOT, TOTAL_ENV_STEPS (1000000),
#   RLPD (1), K (4: must equal the OPAL chunk_size), UTD_RATIO (update() calls per macro row; unset ->
#   K, one per env step), CRITIC_UPDATES (1 critic step per call), MINIBATCH (1024 rows per critic step), NUM_MIN_QS /
#   DISCOUNT (cell defaults), REWARD_SHIFT (-1), LOW_TEMPERATURE (1.0), USE_RND_OFFLINE / USE_RND_ONLINE (1),
#   RND_COEFF (8.0), OFFLINE_RELABEL (min), DISTILL_TARGET (episode_max), EVAL_INTERVAL (25000),
#   EVAL_EPISODES (100), REPO_DIR (this worktree), PYTHON, DRY_RUN (=1: print the command, launch nothing).
#
# Gradient budget (default): the SAME as our flat online CRL + RLPD baseline (agents/online_crl.py:
# one gradient step of 1024 rows per env step) -- one update() call per env step (UTD_RATIO = K per
# macro row), 1 critic step + 1 actor step of 1024 rows each, so SUPE and CRL are compared at equal env
# steps and equal gradient steps. The paper's own schedule (80 critic steps of 256 rows per 4-step macro
# step) is UTD_RATIO=4 CRITIC_UPDATES=20 MINIBATCH=256; it is 20x the budget and NOT the default.
#
# --mem / --time as scripts/slurm/run_distill_bonus_corner_sweep.sbatch: the 1M-row offline buffer +
# the 250k-row macro replay + the update batches need more than 16 GB, and 1M env steps at this UTD
# do not finish in 8 h.
#SBATCH --job-name=supe_online
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --qos=default
set -euo pipefail

CELL=${1:?usage: run_supe_online_seed.sbatch CELL SEED [ADD_EXPLORE] [BONUS_SCALE] [ANNEAL_FRAC]}
SEED=${2:?usage: run_supe_online_seed.sbatch CELL SEED [ADD_EXPLORE] [BONUS_SCALE] [ANNEAL_FRAC]}
ADD_EXPLORE=${3:-none}
BONUS_SCALE=${4:-1.0}
ANNEAL_FRAC=${5:-}

REPO_DIR=${REPO_DIR:-/nas/ucb/ishirgarg/ogbench/.claude/worktrees/supe-baseline-integration-4d17b9}
IMPLS_DIR="$REPO_DIR/impls"
PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
MAIN_IMPLS=/nas/ucb/ishirgarg/ogbench/impls
SUPE_OPAL_ROOT=${SUPE_OPAL_ROOT:-$MAIN_IMPLS/ckpts/final/supe_opal}
EMP_ROOT=${EMP_ROOT:-$MAIN_IMPLS/ckpts/final/empowerment_final}
SAVE_ROOT=${SAVE_ROOT:-$MAIN_IMPLS/exp/supe_slurm}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
RLPD=${RLPD:-1}
K=${K:-4}
# PAPER_SCHEDULE=1: SUPE's own update schedule (train_finetuning_supe.py): 4 update() calls per macro step x
# 20 critic minibatches of 256 rows (exactly half online / half offline) + 1 actor step each, the reward model
# (20 x 128 online rows) and the RND predictor (the single newest transition) once per macro step, discount
# 0.995 (every OGBench command in the SUPE README) with the within-chunk reward discount at 0.99 (OPAL's). Explicit UTD_RATIO / CRITIC_UPDATES / MINIBATCH / DISCOUNT win.
# GOAL_CONDITIONED=1: actor / critic / reward model on concat(s, g) (multigoal envs; agents/supe.py docstring).
PAPER_SCHEDULE=${PAPER_SCHEDULE:-0}
GOAL_CONDITIONED=${GOAL_CONDITIONED:-0}
if [[ "$PAPER_SCHEDULE" == "1" ]]; then
    UTD_RATIO=${UTD_RATIO:-4}
    CRITIC_UPDATES=${CRITIC_UPDATES:-20}
    MINIBATCH=${MINIBATCH:-256}
    DISCOUNT=${DISCOUNT:-0.995}
fi
UTD_RATIO=${UTD_RATIO:-}
CRITIC_UPDATES=${CRITIC_UPDATES:-1}
MINIBATCH=${MINIBATCH:-1024}
LABEL_CHUNK_BYTES=${LABEL_CHUNK_BYTES:-4194304}   # offline-window labelling chunk: 64 MB peaks at ~13 GB of GPU memory
EXTRA_AGENT_FLAGS=${EXTRA_AGENT_FLAGS:-}
REWARD_SHIFT=${REWARD_SHIFT:--1.0}
LOW_TEMPERATURE=${LOW_TEMPERATURE:-1.0}
USE_RND_OFFLINE=${USE_RND_OFFLINE:-1}
USE_RND_ONLINE=${USE_RND_ONLINE:-1}
RND_COEFF=${RND_COEFF:-8.0}
OFFLINE_RELABEL=${OFFLINE_RELABEL:-min}
DISTILL_TARGET=${DISTILL_TARGET:-episode_max}
EVAL_INTERVAL=${EVAL_INTERVAL:-25000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
DRY_RUN=${DRY_RUN:-0}

# The worktree's ogbench/ must shadow the editable install (main checkout): the corner / far /
# multigoal envs are registered only on this branch.
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_ENTITY="ishirgarg-university-of-california-berkeley"
WANDB_KEY_FILE=${WANDB_KEY_FILE:-/nas/ucb/ishirgarg/.wandb_api_key}
if [[ -r "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
    echo "[supe_online] WARNING: $WANDB_KEY_FILE not readable; wandb logging will likely fail." >&2
fi
export TMPDIR=${TMPDIR:-/nas/ttl=60d/ishirgarg/tmp}
mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$MAIN_IMPLS/.jax_cache}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$TMPDIR/autotune_${SLURM_JOB_ID:-$$}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}

cd "$IMPLS_DIR"
source "${SUPE_CELLS_FILE:-scripts/slurm/supe_cells.sh}"   # SUPE_CELLS_FILE: a machine-local copy (other estimator runs)
supe_cell "$CELL" || exit 1
NUM_MIN_QS=${NUM_MIN_QS:-$CELL_NUM_MIN_QS}
DISCOUNT=${DISCOUNT:-$CELL_DISCOUNT}

# ── Preflight ──
[[ -f main_online.py && -f agents/supe.py ]] || { echo "FATAL: $IMPLS_DIR is not the SUPE worktree's impls/" >&2; exit 1; }
"$PYTHON" - "$REPO_DIR" "$CELL_ENV" <<'PY' || exit 1
import os, sys
import gymnasium, ogbench.locomaze, ogbench.manipspace  # noqa: F401
p = os.path.realpath(os.path.dirname(ogbench.__file__)); root = os.path.realpath(sys.argv[1])
if not p.startswith(root + os.sep):
    sys.exit(f'FATAL: import ogbench resolves to {p}, not inside REPO_DIR={root} (PYTHONPATH order)')
env_id = sys.argv[2].replace('-online-v0', '-v0')
if env_id not in gymnasium.registry:
    sys.exit(f'FATAL: {env_id} is not registered by the ogbench package on PYTHONPATH')
PY
[[ -f "$OGBENCH_DATASET_DIR/$CELL_DATASET.npz" ]] || { echo "FATAL: $OGBENCH_DATASET_DIR/$CELL_DATASET.npz missing" >&2; exit 1; }
OPAL_RUN=$(supe_opal_run "$SUPE_OPAL_ROOT" "$CELL_DATASET") || exit 1
"$PYTHON" - "$OPAL_RUN" "$K" "$CELL_DATASET" <<'PY' || exit 1
import json, sys
d, k, dataset = sys.argv[1], int(sys.argv[2]), sys.argv[3]
f = json.load(open(d + '/flags.json')); a = f['agent']
if a.get('agent_name') != 'opal' or a.get('latent_type') != 'continuous':
    sys.exit(f"FATAL: {d} is not a continuous opal run (agent_name={a.get('agent_name')!r}, latent_type={a.get('latent_type')!r})")
if int(a.get('chunk_size', -1)) != k:
    sys.exit(f"FATAL: {d} was trained with chunk_size={a.get('chunk_size')}, but K={k}; set K to match (or SUPE_OPAL_RUN)")
if f.get('env_name') != dataset:
    sys.exit(f"FATAL: {d} was trained on {f.get('env_name')}, cell expects {dataset}")
PY

# ── Flags + tag ──
TAG=supe_rlpd
[[ "$GOAL_CONDITIONED" == "1" ]] && TAG=supe_gc_rlpd
RLPD_FLAG=()
if [[ "$RLPD" == "1" ]]; then
    RLPD_FLAG=(--offline_dataset="$CELL_DATASET")
else
    TAG=${TAG/_rlpd/_norlpd}
fi
if [[ "$USE_RND_OFFLINE" != "1" && "$USE_RND_ONLINE" != "1" ]]; then TAG="${TAG}_nornd"; fi
if [[ "$K" != "4" ]]; then TAG="${TAG}_k${K}"; fi
if [[ "$PAPER_SCHEDULE" == "1" ]]; then TAG="${TAG}_paper"; fi
bool() { [[ "$1" == "1" ]] && echo True || echo False; }
AGENT_FLAGS=(
    --agent.skill_checkpoint_path="$OPAL_RUN"
    --agent.skill_commitment_k="$K"
    --agent.num_min_qs="$NUM_MIN_QS"
    --agent.discount="$DISCOUNT"
    --agent.critic_updates_per_update="$CRITIC_UPDATES"
    --agent.batch_size=$(( MINIBATCH * CRITIC_UPDATES ))
    --agent.reward_shift="$REWARD_SHIFT"
    --agent.low_temperature="$LOW_TEMPERATURE"
    --agent.use_rnd_offline="$(bool "$USE_RND_OFFLINE")"
    --agent.use_rnd_online="$(bool "$USE_RND_ONLINE")"
    --agent.rnd_coeff="$RND_COEFF"
    --agent.offline_relabel="$OFFLINE_RELABEL"
    --agent.label_chunk_bytes="$LABEL_CHUNK_BYTES"
    --agent.goal_conditioned="$(bool "$GOAL_CONDITIONED")"
    $EXTRA_AGENT_FLAGS
)
if [[ -n "$UTD_RATIO" ]]; then AGENT_FLAGS+=(--agent.utd_ratio="$UTD_RATIO"); fi
# gamma_low 0.99: SUPE sums a chunk's rewards (and ds_minr) with the OPAL agent's discount (opal_config 0.99),
# not the SAC discount (MetaPolicyActionWrapper / ChunkDataset get the OPAL agent).
if [[ "$PAPER_SCHEDULE" == "1" ]]; then AGENT_FLAGS+=(--agent.aux_schedule=paper --agent.minibatch_split=interleave --agent.gamma_low=0.99); fi
EMP_CKPT=""
case "$ADD_EXPLORE" in
    none|"") ;;
    distill|distill-to-rlpd)
        [[ -n "$CELL_EMP_RUN" ]] || { echo "FATAL: cell $CELL has no empowerment estimator run; distill arms unavailable" >&2; exit 1; }
        EMP_CKPT="$EMP_ROOT/$CELL_EMP_RUN"
        [[ -f "$EMP_CKPT/flags.json" ]] || { echo "FATAL: no flags.json at $EMP_CKPT" >&2; exit 1; }
        [[ -w "$EMP_CKPT" ]] || { echo "FATAL: $EMP_CKPT is not writable (the offline E cache is written under it)" >&2; exit 1; }
        AGENT_FLAGS+=(
            --agent.emp_checkpoint_path="$EMP_CKPT"
            --agent.emp_num_splus_samples=64
            --agent.add_explore="$ADD_EXPLORE"
            --agent.bonus_scale="$BONUS_SCALE"
            --agent.distill_target="$DISTILL_TARGET"
        )
        if [[ -n "$ANNEAL_FRAC" ]]; then
            AGENT_FLAGS+=(--agent.explore_reward_time_frac="$ANNEAL_FRAC")
            TAG="${TAG}_ann${ANNEAL_FRAC}"
        fi
        if [[ "$ADD_EXPLORE" == "distill" ]]; then TAG="${TAG}_ed${BONUS_SCALE}"; else TAG="${TAG}_edrlpd${BONUS_SCALE}"; fi
        if [[ "$DISTILL_TARGET" == "future_max" ]]; then TAG="${TAG}_fut"; fi ;;
    *) echo "FATAL: ADD_EXPLORE=$ADD_EXPLORE (none | distill | distill-to-rlpd)" >&2; exit 1 ;;
esac
EP_FLAG=()
if [[ -n "$CELL_EPISODE_LENGTH" ]]; then EP_FLAG=(--episode_length="$CELL_EPISODE_LENGTH"); fi
SAVE_DIR="$SAVE_ROOT/$CELL/$TAG"

echo "[supe_online] cell=$CELL env=$CELL_ENV dataset=$CELL_DATASET ep_len=${CELL_EPISODE_LENGTH:-<registered>} seed=$SEED tag=$TAG"
echo "[supe_online]   opal=$OPAL_RUN K=$K num_min_qs=$NUM_MIN_QS discount=$DISCOUNT updates/macro-row=${UTD_RATIO:-K} x $CRITIC_UPDATES critic steps x $MINIBATCH rows"
echo "[supe_online]   add_explore=$ADD_EXPLORE estimator=${EMP_CKPT:-<none>} save_dir=$SAVE_DIR job=${SLURM_JOB_ID:-<none>} node=$(hostname)"

CMD=("$PYTHON" -u main_online.py
    --env_name="$CELL_ENV"
    --seed="$SEED"
    --save_dir="$SAVE_DIR"
    --agent=agents/supe.py
    "${AGENT_FLAGS[@]}"
    "${RLPD_FLAG[@]}"
    "${EP_FLAG[@]}"
    --total_env_steps="$TOTAL_ENV_STEPS"
    --log_interval=5000
    --eval_interval="$EVAL_INTERVAL"
    --eval_episodes="$EVAL_EPISODES"
    --save_interval="$TOTAL_ENV_STEPS"
    --video_episodes=0)
if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: ${CMD[*]}"
    exit 0
fi
mkdir -p "$SAVE_DIR"
if command -v srun > /dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun "${CMD[@]}"
else
    "${CMD[@]}"
fi
