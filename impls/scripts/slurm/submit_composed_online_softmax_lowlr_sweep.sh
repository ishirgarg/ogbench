#!/usr/bin/env bash
# Online COMPOSED skill-policy sweep with composed_grad_method='softmax': the high level's
# softmax(logits) is fed to the low level AS THE SKILL VECTOR (no one-hot, no sampling; agent
# docstring Eq. 8), so one pathwise gradient reaches both levels at one low-level pass per row.
# The swept axis is the LOW-LEVEL learning rate. One sbatch job per (env, low_lr, seed):
#
#   3 envs     (antsoccer-arena-navigate, pointmaze-teleport-navigate, cube-single-play)
#   x 5 low_lr (0, 1e-5, 3e-5, 1e-4, 3e-4) -- 0 == frozen low level fed probability vectors
#   x 3 seeds  (0-2)
#   = 45 jobs total.
#
# Fixed flags (user's choices, 2026-09-15):
#   GRAD_METHOD=softmax; target_entropy_frac=0.5; RLPD on with each checkpoint's OWN dataset;
#   1M env steps; every other knob is the agent default (== the controller's literal values,
#   see submit_composed_online_lowlr_sweep.sh). skill_commitment_k is 1 (the agent asserts it).
#
# Exploration (user's choice, 2026-09-16): LEARNED_ACTION_STD=1 by default. p = softmax(logits)
# is never sampled, but the high level has an action log-std head and the composed action is
# online_crl's tanh-squashed Gaussian centred on the low level's action, with a second alpha
# tuned toward -0.5 * action_dim (agent docstring, Eq. 9). The categorical entropy target
# (TARGET_ENTROPY_FRAC=0.5) is then a regulariser on how soft p is, not the exploration knob.
# LEARNED_ACTION_STD=0 gives the fully deterministic collector (tree rlpd_lowlr<LR>_softmax/).
# Two things specific to 'softmax' worth knowing before reading the curves:
#   * The low level was pretrained on one-hots only, so the simplex interior is off-distribution
#     for it; low_lr=0 is therefore NOT the frozen-controller baseline (that is the enumerate
#     sweep's low_lr=0), it is "frozen low level driven by vectors it never saw".
#
# Checkpoints (all 50-skill empowerment_skill runs under ckpts/final/empowerment_final/):
#   antsoccer-arena-navigate   -- single sd000_* run       -> antsoccer-arena-center-online-v0 (ep len 500)
#   pointmaze-teleport-navigate-- single sd000_* run       -> pointmaze-teleport-sparse-online-v0
#   cube-single-play/sd000_s_38624008.0.20260908_013305    -> cube-single-center-online-v0
#       (the env dir holds five sd000_* runs; this is the one every prior cube sweep used)
#
# Results land under <SKILL_CKPT>/online_composed/rlpd_lowlr<LOW_LR>_softmax_astd/, a tree separate
# from the enumerate sweep's rlpd_lowlr<LOW_LR>/ (which is how the two are compared).
# Logs: logs/slurm/composed_online_softmax_lowlr/.
#
# This is a thin wrapper over submit_composed_online_lowlr_sweep.sh (same sbatch template,
# scripts/slurm/run_online_composed_skill_policy_seed.sbatch). It ONLY submits jobs. Run from the
# rnn login node, from this NAS checkout:
#   bash scripts/slurm/submit_composed_online_softmax_lowlr_sweep.sh
# Overrides (all forwarded): DRY_RUN=1, SEEDS, LOW_LRS, CKPT_KEYS, TARGET_ENTROPY_FRAC,
#   UTD_RATIO, TIME_LIMIT (default 36:00:00; the enumerate runs took 6.5-16 h and softmax is
#   cheaper per update).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> impls/

export GRAD_METHOD=softmax
export LEARNED_ACTION_STD=${LEARNED_ACTION_STD:-1}
export LOW_LRS=${LOW_LRS:-"0 1e-5 3e-5 1e-4 3e-4"}
export SEEDS=${SEEDS:-"0 1 2"}
export CKPT_KEYS=${CKPT_KEYS:-"emp_asoc_nav emp_pmt_nav emp_cube_sgl"}
export TARGET_ENTROPY_FRAC=${TARGET_ENTROPY_FRAC:-0.5}
export TIME_LIMIT=${TIME_LIMIT:-36:00:00}
export LOG_DIR=${LOG_DIR:-logs/slurm/composed_online_softmax_lowlr}
# Never inherit a REINFORCE knob from the calling shell.
unset REINFORCE_BASELINE

exec bash scripts/slurm/submit_composed_online_lowlr_sweep.sh
