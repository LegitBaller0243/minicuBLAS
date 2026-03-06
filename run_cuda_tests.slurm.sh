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
"${BUILD_DIR}/bench" --kernel all --repeats 100 --warmup 20 --batch 8 \
  | tee "logs/bench-${SLURM_JOB_ID:-local}.txt"


echo "Done."
