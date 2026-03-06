#!/bin/bash
#SBATCH --job-name=cuda-kernel-tests
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load cuda

BUILD_DIR="${BUILD_DIR:-build-slurm}"
SRC_DIR="$(pwd -P)"

mkdir -p logs
JOB_TAG="${SLURM_JOB_ID:-local}"
RUN_LOG_DIR="logs/${JOB_TAG}"
mkdir -p "${RUN_LOG_DIR}"

echo "Node: $(hostname)"
nvidia-smi || { echo "No GPU visible"; exit 1; }

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  CACHED_SRC="$(grep '^CMAKE_HOME_DIRECTORY:INTERNAL=' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2- || true)"
  if [[ "${CACHED_SRC}" != "${SRC_DIR}" ]]; then
    echo "Detected stale CMake cache from a different source dir:"
    echo "  cached: ${CACHED_SRC}"
    echo "  current: ${SRC_DIR}"
    echo "Removing ${BUILD_DIR} and reconfiguring..."
    rm -rf "${BUILD_DIR}"
  fi
fi

echo "Configuring CMake..."
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

echo "Building..."
cmake --build "${BUILD_DIR}" -j

echo "Running GEMM correctness + launch smoke tests..."
ctest --test-dir "${BUILD_DIR}" --output-on-failure \
  -R "^(kernels_correctness_vs_cublas|launch_gemm_.*)$"

echo "Running GEMM benchmarks..."
"${BUILD_DIR}/bench" --kernel all --repeats 100 --cpu-repeats 1 --warmup 20 --batch 8 \
  | tee "${RUN_LOG_DIR}/bench.txt"

echo "Running register-tiling focused benchmark..."
"${BUILD_DIR}/bench" --kernel register-tiling --repeats 100 --cpu-repeats 1 --warmup 20 --batch 8 \
  | tee "${RUN_LOG_DIR}/bench-register-tiling.txt"

echo "Running flash-attn benchmark..."
"${BUILD_DIR}/cuda_practice" \
  --kernel flash-attn \
  --batch 4 \
  --n 8 \
  --m 1024 \
  --k 64 \
  --repeats 100 \
  | tee "${RUN_LOG_DIR}/flash-attn.txt"

echo "Running rk4-heat3d benchmark..."
"${BUILD_DIR}/cuda_practice" \
  --kernel rk4-heat3d \
  --m 128 \
  --k 128 \
  --n 128 \
  --repeats 100 \
  | tee "${RUN_LOG_DIR}/rk4-heat3d.txt"

if command -v ncu >/dev/null 2>&1; then
  NCU_LOG_DIR="${RUN_LOG_DIR}/ncu"
  mkdir -p "${NCU_LOG_DIR}"
  NCU_COMMON_ARGS=(
    --target-processes all
    --set full
    --force-overwrite true
    --profile-from-start on
  )

  echo "Running Nsight Compute profiles..."

  # Unfiltered sanity pass: confirms ncu can see launches on this node.
  ncu "${NCU_COMMON_ARGS[@]}" \
      --export "${NCU_LOG_DIR}/sanity-unfiltered" \
      "${BUILD_DIR}/cuda_practice" --kernel naive --m 256 --k 256 --n 256 --repeats 5 \
      | tee "${NCU_LOG_DIR}/sanity-unfiltered.txt"

  ncu "${NCU_COMMON_ARGS[@]}" \
      --kernel-name-base demangled \
      --kernel-name "regex:.*naiveMul.*" \
      --export "${NCU_LOG_DIR}/naive" \
      "${BUILD_DIR}/bench" --kernel naive --repeats 20 --cpu-repeats 1 --warmup 5 --batch 8 \
      | tee "${NCU_LOG_DIR}/naive.txt"

  ncu "${NCU_COMMON_ARGS[@]}" \
      --kernel-name-base demangled \
      --kernel-name "regex:.*tilingMul.*" \
      --export "${NCU_LOG_DIR}/tiled" \
      "${BUILD_DIR}/bench" --kernel tiled --repeats 20 --cpu-repeats 1 --warmup 5 --batch 8 \
      | tee "${NCU_LOG_DIR}/tiled.txt"

  ncu "${NCU_COMMON_ARGS[@]}" \
      --kernel-name-base demangled \
      --kernel-name "regex:.*regSharedTilingMul.*" \
      --export "${NCU_LOG_DIR}/register-tiling" \
      "${BUILD_DIR}/bench" --kernel register-tiling --repeats 20 --cpu-repeats 1 --warmup 5 --batch 8 \
      | tee "${NCU_LOG_DIR}/register-tiling.txt"

  # These runtime paths are currently placeholders in launch/launch.cu and do not launch kernels.
  # Leave a marker log to avoid misleading "No kernels were profiled" warnings.
  echo "Skipping flash-attn ncu profile: cuda_practice --kernel flash-attn is not wired to kernel launch yet." \
      | tee "${NCU_LOG_DIR}/flash-attn.txt"
  echo "Skipping rk4-heat3d ncu profile: cuda_practice --kernel rk4-heat3d is not wired to kernel launch yet." \
      | tee "${NCU_LOG_DIR}/rk4-heat3d.txt"
else
  echo "ncu not found in PATH; skipping Nsight Compute profiling."
fi

echo "Done."
