#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

# define blocksize 16

__global__ void naiveMul(float* A, float* B, float* C, int M, int K, int N);
__global__ void tilingMul(const float* A, const float* B, float* C, int N);
__global__ void batchNaiveMul(const float* A, const float* B, float* C, int M, int K, int N);
__global__ void batchTilingMul(const float* A, const float* B, float* C, int M, int K, int N);
