#!/usr/bin/env bash
# Online COMPOSED (non-frozen low level) skill-policy sweep over the low-level learning rate:
# agents/online_composed_skill_policy.py on top of the three 50-skill empowerment checkpoints
# under ckpts/final/empowerment_final. One sbatch job per (env, low_lr, seed):
#
#   3 envs     (antsoccer-arena-navigate, pointmaze-teleport-stitch, cube-single-play)
#   x 3 low_lr (3e-4, 1e-4, 3e-5)
#   x 5 seeds  (0-4)
#   = 45 jobs total.
#
# What is different from every *_controller sweep in this directory: the low level is NOT
# frozen. pi_hi(k|s,g) and pi_lo(a|s,z_k) are trained jointly as one flat policy
# pi(a|s,g) = sum_k pi_hi(k|s,g) pi_lo(a|s,z_k), the low level at its own LOW_LR (the swept
# axis, and part of the save path). There is NO skill horizon -- the skill is redrawn every
# env step -- so rows are env steps and RLPD mixes plain (s_t, a_t) offline rows with no
# window labelling at all. LOW_LR=0 would reproduce the frozen baseline at k=1; the three
# values here are 10x, 3x and 1x the agent's default 3e-5 (= lr/10).
#
# Fixed flags (user's choices, 2026-09-14):
#   RLPD on, each checkpoint's OWN offline dataset (its flags.json env_name) -- the launcher default.
#   target_entropy_frac = 0.5   (H_target = 0.5 * log(50) = 1.96 nats for the categorical high
#                                level, which is the only entropy term this agent has).
#   1M env steps, 5 seeds per cell.
#
# Schedule: nothing here overrides it. Every shared hyperparameter of
# agents/online_composed_skill_policy.py is LITERALLY the value in
# agents/online_crl_skill_controller.py -- unroll_length 50, utd_ratio 1, min_replay_size 1000,
# replay_size 50000, batch_size 1024, lr 3e-4, both hidden dims (512,512,512), latent_dim 512,
# layer_norm, discount 0.99, low_temperature 0, offline_ratio 0.5 -- with NO rescaling by the
# controller's skill_commitment_k (user's explicit instruction, 2026-09-15). A row is one policy
# DECISION in both agents, so utd_ratio=1 is one gradient step per decision and replay_size is
# 50000 decisions of history for both; each stored row is consequently sampled ~batch_size times
# in expectation in both, which is the sense in which the replay is matched.
#
# The consequence to be aware of when reading the curves: per ENV STEP (main_online.py's x-axis)
# the two are NOT equal, because a controller decision spans k=10 env steps --
#     updates/env step  0.1 (controller)  vs  1.0 (here)   -> 100k vs 1M updates at 1M env steps
#     first update at   10000 env steps   vs  1000
#     buffer holds      500000 env steps  vs  50000
# WALL CLOCK: 1M updates at a measured ~119 ms/update (K=50, B=1024, A6000 under heavy host
# contention) is ~33 h; the GPU-bound floor is ~28.6 ms/update, i.e. ~8 h. The truth on a quiet
# compute node is in between and is NOT known -- if jobs hit the limit, raise TIME_LIMIT. Runs
# checkpoint every 100k env steps and write eval.csv every 20k, so a truncated job still yields
# a usable partial curve.
#
# Checkpoints (all 50-skill empowerment_skill runs under ckpts/final/empowerment_final/):
#   antsoccer-arena-navigate   -- single sd000_* run
#   pointmaze-teleport-stitch  -- single sd000_* run
#   cube-single-play/sd000_s_38624008.0.20260908_013305
#       -- explicit: that env dir holds FIVE sd000_* runs; this is the one the user picked for
#          the TES and entropy sweeps (2026-09-13/14), so the cells stay comparable.
#
# Online env per env family (deterministic, noise-free task sets; same table as
# submit_pmt_asoc_cube_tes_seeds.sh):
#   antsoccer-arena     -> antsoccer-arena-center-online-v0   ("normal", NOT the corner env;
#                          registered at 1000 but overridden to 500 like every other online run here)
#   pointmaze-teleport  -> pointmaze-teleport-sparse-online-v0 (horizon 1000)
#   cube-single         -> cube-single-center-online-v0        (horizon 200)
#
# Results land under <SKILL_CKPT>/online_composed/rlpd_lowlr<LOW_LR>/, a tree separate from
# every prior sweep's <SKILL_CKPT>/online_controller*/, so nothing existing is touched.
#
# This ONLY submits jobs. Run from the rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_composed_online_lowlr_sweep.sh
# Overrides:
#   DRY_RUN=1                        print the sbatch commands without submitting
#   SEEDS="0 1"                      a different seed set
#   LOW_LRS="3e-4 0"                 a different low-lr set (0 == frozen-low-level ablation)
#   CKPT_KEYS="emp_cube_sgl"         a subset of the cells; emp_pmt_nav (pointmaze-teleport-NAVIGATE,
#                                    the ckpt the controller baselines at frac=0.5 exist for) is
#                                    defined but NOT in the default set
#   LOG_DIR=logs/slurm/foo           where the per-job .log files go
#   TARGET_ENTROPY_FRAC=0.9          H_target = frac * log(num_skills)
#   UTD_RATIO=0.1                    one update per 10 env steps instead of per env step, i.e.
#                                    the controller's gradient budget per ENV STEP (see above)
#   ACTOR_BATCH_SIZE=256             rows the K-term actor loss uses (unset -> the full batch)
#   GRAD_METHOD=reinforce            sampled-skill REINFORCE high level instead of the exact
#                                    K-term sum (agent docstring, Eqs. 6-7); results go to a
#                                    separate rlpd_lowlr<LR>_reinforce/ tree, so it can be run
#                                    as a second pass without touching the enumerate results
#   GRAD_METHOD=softmax              softmax(logits) fed to the low level as the skill vector
#                                    (agent docstring, Eq. 8); tree rlpd_lowlr<LR>_softmax/.
#                                    submit_composed_online_softmax_lowlr_sweep.sh wraps this.
#   REINFORCE_BASELINE=none          drop the leave-one-out batch baseline (GRAD_METHOD=reinforce)
#   LEARNED_ACTION_STD=1             (GRAD_METHOD=softmax) high-level action log-std head + SAC
#                                    action entropy; tree rlpd_lowlr<LR>_softmax_astd/
#   TIME_LIMIT=20:00:00              sbatch --time override
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

SBATCH_SCRIPT=scripts/slurm/run_online_composed_skill_policy_seed.sbatch
LOG_DIR=${LOG_DIR:-logs/slurm/composed_online_lowlr}
DATASET_DIR=${OGBENCH_DATASET_DIR:-/nas/ucb/ishirgarg/.ogbench/data}
PYTHON=${PYTHON:-/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python}
TIME_LIMIT=${TIME_LIMIT:-20:00:00}
mkdir -p "$LOG_DIR"

DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
LOW_LRS=${LOW_LRS:-"3e-4 1e-4 3e-5"}
TARGET_ENTROPY_FRAC=${TARGET_ENTROPY_FRAC:-0.5}
UTD_RATIO=${UTD_RATIO:-}   # empty -> the agent default (0.1), which matches the controller
SAVE_SUBDIR=${SAVE_SUBDIR:-online_composed}
GRAD_METHOD=${GRAD_METHOD:-enumerate}
EXPORT="ALL,GRAD_METHOD=$GRAD_METHOD"
if [[ -n "$UTD_RATIO" ]]; then EXPORT="$EXPORT,UTD_RATIO=$UTD_RATIO"; fi
if [[ -n "${REINFORCE_BASELINE:-}" ]]; then EXPORT="$EXPORT,REINFORCE_BASELINE=$REINFORCE_BASELINE"; fi
LEARNED_ACTION_STD=${LEARNED_ACTION_STD:-0}
EXPORT="$EXPORT,LEARNED_ACTION_STD=$LEARNED_ACTION_STD"
if [[ -n "${ACTOR_BATCH_SIZE:-}" ]]; then EXPORT="$EXPORT,ACTOR_BATCH_SIZE=$ACTOR_BATCH_SIZE"; fi

# The default cell set is the first three; emp_pmt_nav is opt-in via CKPT_KEYS (both pointmaze
# checkpoints run on the same sparse online env, they differ only in the pretraining dataset).
DEFAULT_KEYS="emp_asoc_nav emp_pmt_stitch emp_cube_sgl"
ALL_KEYS=(emp_asoc_nav emp_pmt_stitch emp_cube_sgl emp_pmt_nav)
ALL_CKPT_ROOTS=(
    "ckpts/final/empowerment_final/antsoccer-arena-navigate"
    "ckpts/final/empowerment_final/pointmaze-teleport-stitch"
    "ckpts/final/empowerment_final/cube-single-play/sd000_s_38624008.0.20260908_013305"
    "ckpts/final/empowerment_final/pointmaze-teleport-navigate"
)
ALL_ONLINE_ENVS=(
    antsoccer-arena-center-online-v0
    pointmaze-teleport-sparse-online-v0
    cube-single-center-online-v0
    pointmaze-teleport-sparse-online-v0
)
ALL_EPISODE_LENGTHS=(500 "" "" "")
# cube's RLPD pass over its ~300MB dataset is the memory peak (as in every cube sweep here):
# more headroom, and no 16GB A4000s.
ALL_EXTRA_SBATCH_FLAGS=(
    ""
    ""
    "--mem=32gb --exclude=ppo.ist.berkeley.edu,vae.ist.berkeley.edu"
    ""
)

read -r -a WANTED <<< "${CKPT_KEYS:-$DEFAULT_KEYS}"
KEYS=(); CKPT_ROOTS=(); ONLINE_ENVS=(); EPISODE_LENGTHS=(); EXTRA_SBATCH_FLAGS=()
for w in "${WANTED[@]}"; do
    found=0
    for i in "${!ALL_KEYS[@]}"; do
        if [[ "${ALL_KEYS[$i]}" == "$w" ]]; then
            KEYS+=("${ALL_KEYS[$i]}")
            CKPT_ROOTS+=("${ALL_CKPT_ROOTS[$i]}")
            ONLINE_ENVS+=("${ALL_ONLINE_ENVS[$i]}")
            EPISODE_LENGTHS+=("${ALL_EPISODE_LENGTHS[$i]}")
            EXTRA_SBATCH_FLAGS+=("${ALL_EXTRA_SBATCH_FLAGS[$i]}")
            found=1
            break
        fi
    done
    (( found )) || { echo "ERROR: unknown key '$w' (known: ${ALL_KEYS[*]})" >&2; exit 1; }
done

# (a) an exact leaf checkpoint dir (flags.json directly), or (b) an env dir holding exactly
# one sd000_* run.
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

    read -r AGENT_NAME NUM_SKILLS < <("$PYTHON" -c "
import json, sys
a = json.load(open(sys.argv[1] + '/flags.json'))['agent']
print(a['agent_name'], a['num_skills'])
" "$SKILL_CKPT")
    [[ "$AGENT_NAME" == "empowerment_skill" ]] || { echo "ERROR: $SKILL_CKPT is a $AGENT_NAME run, expected empowerment_skill" >&2; exit 1; }
    [[ "$NUM_SKILLS" == "50" ]] || { echo "ERROR: $SKILL_CKPT has num_skills=$NUM_SKILLS, expected 50" >&2; exit 1; }

    OFFLINE_DATASET=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1] + '/flags.json'))['env_name'])" "$SKILL_CKPT")
    if [[ ! -f "$DATASET_DIR/$OFFLINE_DATASET.npz" ]]; then
        echo "WARNING: $DATASET_DIR/$OFFLINE_DATASET.npz is missing; these jobs will fail on a compute node." >&2
    fi

    ENV_NAME=${ONLINE_ENVS[$c]}
    EPISODE_LENGTH=${EPISODE_LENGTHS[$c]}
    read -r -a EXTRA_FLAGS <<< "${EXTRA_SBATCH_FLAGS[$c]}"
    echo "# $KEY -> $SKILL_CKPT (env=$ENV_NAME, rlpd dataset $OFFLINE_DATASET, num_skills=$NUM_SKILLS)"

    for LOW_LR in $LOW_LRS; do
        for SEED in $SEEDS; do
            JOB_NAME="composed_${KEY}_lr${LOW_LR}_s${SEED}"
            OUT="$LOG_DIR/${JOB_NAME}_%j.log"
            cmd=(sbatch --job-name="$JOB_NAME" --output="$OUT" --time="$TIME_LIMIT" --export="$EXPORT"
                 "${EXTRA_FLAGS[@]}"
                 "$SBATCH_SCRIPT" "$SKILL_CKPT" "$ENV_NAME" "$SEED" "" "$EPISODE_LENGTH" "$SAVE_SUBDIR"
                 "$TARGET_ENTROPY_FRAC" "$LOW_LR")
            echo "${cmd[@]}"
            if [[ "$DRY_RUN" != "1" ]]; then
                "${cmd[@]}"
            fi
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo "$( [[ "$DRY_RUN" == "1" ]] && echo "would submit" || echo "submitted" ) $n_submitted jobs" \
     "(low_lrs: $LOW_LRS, seeds: $SEEDS, target_entropy_frac=$TARGET_ENTROPY_FRAC, utd_ratio=${UTD_RATIO:-<agent default 1.0, = controller>}," \
     "grad_method=$GRAD_METHOD, learned_action_std=$LEARNED_ACTION_STD," \
     "save subdir $SAVE_SUBDIR)"
