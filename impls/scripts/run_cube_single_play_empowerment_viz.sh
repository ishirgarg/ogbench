#!/usr/bin/env bash
# Generate 4 empowerment visualizations (gripper-xy map x2 openness, height-vs-aperture
# map, distance-to-cube curve, real-trajectory phase plot) for every bc_alpha seed
# variant under ckpts/final/empowerment_final/cube-single-play.
#
# agent.empowerment() streams over skills/samples internally (lax.scan + chunked
# lax.map, see agents/empowerment_skill.py) instead of a single big vmap, so this
# compiles in seconds rather than the ~1hr pathological ptxas compile a naive vmap
# over num_skills x num_splus_samples used to trigger.
set -uo pipefail

cd "$(dirname "$0")/.."   # -> impls/

PY=/nas/ucb/ishirgarg/miniconda3/envs/ogbench/bin/python

read -r -a GPUS <<< "${GPUS:-0 1 2}"
NGPU=${#GPUS[@]}

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}
export JAX_COMPILATION_CACHE_DIR="$(pwd)/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

CKPT_ROOT="ckpts/final/empowerment_final/cube-single-play"
CUBE_XY="0.425,0.0"
GRID_RES=30
NUM_SPLUS_SAMPLES=384
EMP_SAMPLE_CHUNK_SIZE=64

mkdir -p logs
RUN_DIRS=()
for d in "$CKPT_ROOT"/*/; do
  [ -f "${d}params_1000000.pkl" ] && RUN_DIRS+=("$d")
done
echo "Found ${#RUN_DIRS[@]} checkpoints: ${RUN_DIRS[*]}"

# Build the job list checkpoint-major: [ckpt1's 5 jobs, ckpt2's 5 jobs, ...]
CMDS=()
LOGS=()
LABELS=()
for run_dir in "${RUN_DIRS[@]}"; do
  tag=$(basename "$run_dir")

  CMDS+=("$PY -u plot_empowerment_map_cube_gripperxy.py --run_dir '$run_dir' --grid_res $GRID_RES --num_splus_samples $NUM_SPLUS_SAMPLES --emp_sample_chunk_size $EMP_SAMPLE_CHUNK_SIZE --cube_xys '$CUBE_XY' --openness 0.0")
  LOGS+=("logs/cube1p_gripperxy_closed_${tag}.log")
  LABELS+=("gripperxy_closed $tag")

  CMDS+=("$PY -u plot_empowerment_map_cube_gripperxy.py --run_dir '$run_dir' --grid_res $GRID_RES --num_splus_samples $NUM_SPLUS_SAMPLES --emp_sample_chunk_size $EMP_SAMPLE_CHUNK_SIZE --cube_xys '$CUBE_XY' --openness 1.0")
  LOGS+=("logs/cube1p_gripperxy_open_${tag}.log")
  LABELS+=("gripperxy_open $tag")

  CMDS+=("$PY -u plot_empowerment_map_cube_height_aperture.py --run_dir '$run_dir' --grid_res $GRID_RES --num_splus_samples $NUM_SPLUS_SAMPLES --emp_sample_chunk_size $EMP_SAMPLE_CHUNK_SIZE --cube_xy '$CUBE_XY'")
  LOGS+=("logs/cube1p_height_aperture_${tag}.log")
  LABELS+=("height_aperture $tag")

  CMDS+=("$PY -u plot_empowerment_curve_cube_distance.py --run_dir '$run_dir' --num_distances 30 --num_angles 16 --num_splus_samples $NUM_SPLUS_SAMPLES --emp_sample_chunk_size $EMP_SAMPLE_CHUNK_SIZE --cube_xy '$CUBE_XY'")
  LOGS+=("logs/cube1p_distance_curve_${tag}.log")
  LABELS+=("distance_curve $tag")

  CMDS+=("$PY -u plot_empowerment_trajectory_phases.py --run_dir '$run_dir' --num_episodes 1 --num_splus_samples $NUM_SPLUS_SAMPLES --emp_sample_chunk_size $EMP_SAMPLE_CHUNK_SIZE --emp_batch_size 256")
  LOGS+=("logs/cube1p_trajectory_phases_${tag}.log")
  LABELS+=("trajectory_phases $tag")
done

echo "Dispatching ${#CMDS[@]} jobs across GPUs: ${GPUS[*]}"

i=0
for idx in "${!CMDS[@]}"; do
  gpu=${GPUS[$((i % NGPU))]}
  echo "[gpu $gpu] ${LABELS[$idx]} -> ${LOGS[$idx]}"
  eval "CUDA_VISIBLE_DEVICES=$gpu ${CMDS[$idx]} > '${LOGS[$idx]}' 2>&1 &"
  i=$((i + 1))
  (( i % NGPU == 0 )) && wait
done
wait

fail=0
for idx in "${!CMDS[@]}"; do
  grep -q "Saved image:" "${LOGS[$idx]}" || { echo "FAILED: ${LABELS[$idx]} (see ${LOGS[$idx]})"; fail=1; }
done

echo "ALL DONE (cube-single-play empowerment viz, fail=$fail)."
exit $fail
