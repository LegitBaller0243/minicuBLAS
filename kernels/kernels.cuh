#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

# define blocksize 16

enum Flag : int {
    OP_N = 0,
    OP_T = 1
};

struct Transpose {
    Flag transA;
    Flag transB;
};


__device__ __forceinline__
float A_at(const float* A, int i, int K, int kk, int c, Flag transA) {
    return (transA == OP_T) 
        ? A[(kk + c) * K + i] 
        : A[i * K + kk + c];
}
__device__ __forceinline__
float B_at(const float* B, int j, int N, int kk, int r, Flag transB) {
    return (transB == OP_T) 
        ? B[j * N + kk + r]
        : B[(kk + r) * N + j];
}

__global__ void naiveMul(float* A, float* B, float* C, int M, int K, int N, float alpha, float beta);
__global__ void tilingMul(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta);
__global__ void transposeTilingMul(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta, Transpose trans);
__global__ void batchNaiveMul(const float* A, const float* B, float* C, int M, int K, int N);
__global__ void batchStridedMul(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta);
