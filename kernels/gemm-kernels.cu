#include "gemm-kernels.cuh"

//matmul kernel
// each thread exists within a block 
__global__ void naiveMul(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C,
        int M, int K, int N,
        float alpha) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        float sum = 0;
        for (int k = 0; k < K; k += 1) {
            sum += A[i * K + k] * B[k * N + j];
        }
        int idx = i * N + j;
        C[idx] = sum * alpha;
    }
}
//tiling kernel on a 1D grid
__global__ void tilingMul(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C,
        int M, int N, int K,
        float alpha) {

    int r = threadIdx.y;
    int c = threadIdx.x;

    int i = blockIdx.y * blocksize + r;
    int j = blockIdx.x * blocksize + c;


    __shared__ float A_block[blocksize][blocksize];
    __shared__ float B_block[blocksize][blocksize];

    float sum = 0.0f;
    for (int kk = 0; kk < K; kk += blocksize) {
        A_block[r][c] = (i < M && (kk + c) < K)? A[i * K + kk + c] : 0.0f;
        B_block[r][c] = (j < N && (r + kk) < K)? B[(kk + r) * N + j] : 0.0f;
        __syncthreads();
            
        for (int k = 0; k < blocksize; k ++) {
            sum += A_block[r][k] * B_block[k][c]; 
        }
        __syncthreads();
    }
    if (i < M && j < N) {
        int idx = i * N + j;
        C[idx] = sum * alpha;
    }
}

__global__ void transposeTilingMul(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C,
        int M, int N, int K,
        float alpha, Transpose trans) {

    int r = threadIdx.y;
    int c = threadIdx.x;

    int i = blockIdx.y * blocksize + r;
    int j = blockIdx.x * blocksize + c;


    __shared__ float A_block[blocksize][blocksize];
    __shared__ float B_block[blocksize][blocksize];

    float sum = 0.0f;
    for (int kk = 0; kk < K; kk += blocksize) {
        A_block[r][c] = (i < M && (kk + c) < K)? A_at(A, i, K, kk, c, trans.transA): 0.0f;
        B_block[r][c] = (j < N && (r + kk) < K)? B_at(B, j, N, kk, r, trans.transB) : 0.0f;
        __syncthreads();
            
        for (int k = 0; k < blocksize; k ++) {
            sum += A_block[r][k] * B_block[k][c]; 
        }
        __syncthreads();
    }
    if (i < M && j < N) {
        int idx = i * N + j;
        C[idx] = sum * alpha;
    }
}
//batched naive matMul kernel ()
__global__ void batchNaiveMul(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, int M, int K, int N) {
    //each thread takes care of C[b][i][j]
    int batch = blockIdx.x;


    int i = (blockIdx.z * blockDim.y) + threadIdx.y;
    int j = (blockIdx.y * blockDim.x) + threadIdx.x;
    
    if (i < M && j < N) {
        float sum = 0;
        for (int k = 0; k < K; k ++) {
            sum += A[batch * M * K + i * K + k] * B[batch * K * N + k * N + j];
        }
        C[batch * M * N + i * N + j] = sum;
    }
}


__global__ void batchTilingMul(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, int M, int K, int N, float alpha) {
    int batch = blockIdx.x;

    //C[b][i][j]


    int r = threadIdx.y;
    int c = threadIdx.x;

    int i = blockIdx.z * blockDim.y + r;
    int j = blockIdx.y * blockDim.x + c;

    __shared__ float A_block[blocksize][blocksize];
    __shared__ float B_block[blocksize][blocksize];
    float sum = 0;

    const float* Ab = A + batch * M * K;
    const float* Bb = B + batch * K * N;
    float* Cb = C + batch * M * N;

    // for each block load memory into data
    for (int k = 0; k < K; k += blocksize) {
        A_block[r][c] = (i < M && (k + c) < K)? Ab[i * K + k + c] : 0;
        B_block[r][c] = (j < N && (r + k) < K)? Bb[(k + r) * N + j] : 0;
        __syncthreads();

        for (int kk = 0; kk < blocksize; kk++) {
            sum += A_block[r][kk] * B_block[kk][c];
        }
        __syncthreads();
    }
    if (i < M && j < N) {
        int idx = i * N + j;
        Cb[idx] = sum * alpha;
    }
}
