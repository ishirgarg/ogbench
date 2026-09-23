#!/usr/bin/env bash
# dgx6 copy of supe_cells.sh: identical except cube_sgl / cube_mg use the cube-single-play estimator available on
# dgx6 (sd000_s_38579169.0.20260904_235556) instead of the rnn run sd000_s_38624008.0.20260908_013305.
# Selected with SUPE_CELLS_FILE (run_supe_online_seed.sbatch, run_supe_pipeline_cube_mg_amz_all_local.sh).
# The SUPE experiment cells: one line per online env, shared by run_supe_online_seed.sbatch
# and submit_supe_online_seeds.sh (source this file, then call `supe_cell <name>`).
#
# Every cell is one of the (online env, RLPD dataset, episode length, empowerment estimator)
# quadruples our flat-CRL / distilled-bonus sweeps already use, so a SUPE run lands on the
# same x-axis, the same offline data, the same eval protocol and -- for the "+ distill" arm --
# the same frozen estimator as those baselines:
#   * *-center       scripts/slurm/submit_flat_crl_explore_bonus_baselines.sh, submit_opal_online_controller_seeds.sh
#   * corner / far / multigoal / all-squares   scripts/slurm/run_distill_bonus_*_sweep.sbatch,
#                    scripts/run_distill_bonus_local_sweep_{newenvs,cornerenvs}.sh
# The OPAL skills a cell uses are the continuous run trained on the cell's dataset
# (scripts/slurm/submit_supe_opal_pretrain.sh), found under $SUPE_OPAL_ROOT/<dataset>/.
#
# Per-cell SUPE hyperparameters follow the paper's per-env commands (README of
# github.com/rail-berkeley/supe): num_min_qs=1 on the maze / antsoccer families, 2 on cube;
# discount 0.995 on cube and antsoccer, 0.99 on the mazes.

# name -> "online_env|offline_dataset|episode_length|estimator_run|num_min_qs|discount"
declare -A SUPE_CELLS=(
    # antmaze-medium (dataset antmaze-medium-navigate-v0), registered horizon 1000
    [amz_center]="antmaze-medium-center-online-v0|antmaze-medium-navigate-v0||antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001|1|0.99"
    [amz_corner]="antmaze-medium-corner-sparse-online-v0|antmaze-medium-navigate-v0||antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001|1|0.99"
    [amz_all]="antmaze-medium-corner-all-squares-online-v0|antmaze-medium-navigate-v0||antmaze-medium-navigate/sd000_s_37866290.0.20260821_030441_k50_s0.01_bc0.001|1|0.99"
    # pointmaze-teleport (dataset pointmaze-teleport-navigate-v0), registered horizon 1000
    [pmt_center]="pointmaze-teleport-center-online-v0|pointmaze-teleport-navigate-v0||pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836|1|0.99"
    [pmt_sparse]="pointmaze-teleport-sparse-online-v0|pointmaze-teleport-navigate-v0||pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836|1|0.99"
    [pmt_corner]="pointmaze-teleport-corner-sparse-online-v0|pointmaze-teleport-navigate-v0||pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836|1|0.99"
    [pmt_all]="pointmaze-teleport-corner-all-squares-online-v0|pointmaze-teleport-navigate-v0||pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836|1|0.99"
    # antsoccer-arena (dataset antsoccer-arena-navigate-v0), project-wide episode length 500
    [asoc_center]="antsoccer-arena-center-online-v0|antsoccer-arena-navigate-v0|500|antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836|1|0.995"
    [asoc_corner]="antsoccer-arena-corner-online-v0|antsoccer-arena-navigate-v0|500|antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836|1|0.995"
    [asoc_cmg]="antsoccer-arena-corner-multigoal-online-v0|antsoccer-arena-navigate-v0|500|antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836|1|0.995"
    [asoc_fmg]="antsoccer-arena-far-multigoal-online-v0|antsoccer-arena-navigate-v0|500|antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836|1|0.995"
    # cube (datasets cube-single-play-v0 / cube-double-play-v0), registered horizons 200 / 500
    [cube_sgl]="cube-single-center-online-v0|cube-single-play-v0||cube-single-play/sd000_s_38579169.0.20260904_235556|2|0.995"
    [cube_mg]="cube-single-multigoal-online-v0|cube-single-play-v0||cube-single-play/sd000_s_38579169.0.20260904_235556|2|0.995"
    [cube_dbl]="cube-double-center-online-v0|cube-double-play-v0|||2|0.995"   # no estimator run: distill arms unavailable
)

# Sets CELL_ENV, CELL_DATASET, CELL_EPISODE_LENGTH, CELL_EMP_RUN, CELL_NUM_MIN_QS, CELL_DISCOUNT.
supe_cell() {
    local spec=${SUPE_CELLS[$1]:-}
    if [[ -z "$spec" ]]; then
        echo "ERROR: unknown SUPE cell '$1' (known: ${!SUPE_CELLS[*]})" >&2
        return 1
    fi
    IFS='|' read -r CELL_ENV CELL_DATASET CELL_EPISODE_LENGTH CELL_EMP_RUN CELL_NUM_MIN_QS CELL_DISCOUNT <<< "$spec"
}

# Locate the continuous OPAL run for a dataset under $1 (SUPE_OPAL_ROOT): exactly one run dir holding
# flags.json + params_*.pkl, directly under <root>/<dataset>/ or in main.py's <root>/<dataset>/OGBench/<group>/.
# SUPE_OPAL_RUN (env var, optional) names the run dir explicitly and skips the search.
supe_opal_run() {
    local root=$1 dataset=$2
    if [[ -n "${SUPE_OPAL_RUN:-}" ]]; then
        [[ -f "$SUPE_OPAL_RUN/flags.json" ]] || { echo "ERROR: SUPE_OPAL_RUN=$SUPE_OPAL_RUN has no flags.json" >&2; return 1; }
        echo "${SUPE_OPAL_RUN%/}"
        return
    fi
    local dir=$root/$dataset
    [[ -d "$dir" ]] || { echo "ERROR: no OPAL runs for $dataset under $root (run submit_supe_opal_pretrain.sh first)" >&2; return 1; }
    local runs=()
    local f
    while IFS= read -r f; do
        compgen -G "$(dirname "$f")/params_*.pkl" > /dev/null && runs+=("$(dirname "$f")")
    done < <(find "$dir" -maxdepth 4 -name flags.json | sort)
    if (( ${#runs[@]} != 1 )); then
        echo "ERROR: expected exactly one finished OPAL run under $dir, found ${#runs[@]}: ${runs[*]:-<none>}" >&2
        echo "       (pass SUPE_OPAL_RUN=<run dir> to pick one)" >&2
        return 1
    fi
    echo "${runs[0]}"
}
