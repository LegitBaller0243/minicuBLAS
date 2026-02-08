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

mkdir -p logs build

echo "Node: $(hostname)"
nvidia-smi || { echo "No GPU visible"; exit 1; }

echo "Configuring CMake..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

echo "Building..."
cmake --build build -j

echo "Running correctness tests..."
ctest --test-dir build --output-on-failure

echo "Done."
