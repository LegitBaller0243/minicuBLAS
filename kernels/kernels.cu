#include "kernels.cuh"

//matmul kernel
// each thread exists within a block and we use math to turn 1D grid to a 2D grid index of that block
__global__ void naiveMul(float* A, float* B, float* C, 
        int M, int K, int N, 
        float alpha, float beta) {

    int numX_blocks = (N + blocksize - 1) / blocksize;

    int blockRow = blockIdx.x / numX_blocks;
    int blockCol = blockIdx.x % numX_blocks;
 
    int x = blockRow * blockDim.x + threadIdx.x;
    int y = blockCol * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        float sum = 0;
        for (int k = 0; k < K; k += 1) {
            sum += A[x * K + k] * B[k * N + y];
        }
        int idx = x * N + y;
        if (beta == 0.0f) {
            C[idx] = sum * (alpha);
        } else {
            C[idx] = sum * alpha + beta * C[idx];
        }
    }
}
//tiling kernel on a 1D grid
__global__ void tilingMul(const float* A, const float* B, float* C, 
        int M, int N, int K, 
        float alpha, float beta) {
            
    int numTiles = (N + blocksize - 1)/ blocksize;
    int blockRow = blockIdx.x / numTiles;
    int blockCol = blockIdx.x % numTiles;

    int r = threadIdx.y;
    int c = threadIdx.x;

    int i = blockRow * blocksize + r;
    int j = blockCol * blocksize + c; 


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
        if (beta == 0.0f) {
            C[idx] = sum * (alpha);
        } else {
            C[idx] = sum * alpha + beta * C[idx];
        }
    }
}

__global__ void transposeTilingMul(const float* A, const float* B, float* C, 
        int M, int N, int K, 
        float alpha, float beta, Transpose trans) {
    int numTiles = (N + blocksize - 1)/ blocksize;
    int blockRow = blockIdx.x / numTiles;
    int blockCol = blockIdx.x % numTiles;

    int r = threadIdx.y;
    int c = threadIdx.x;

    int i = blockRow * blocksize + r;
    int j = blockCol * blocksize + c; 


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
        if (beta == 0.0f) {
            C[idx] = sum * (alpha);
        } else {
            C[idx] = sum * alpha + beta * C[idx];
        }
    }
}
//batched naive matMul kernel ()
__global__ void batchNaiveMul(const float* A, const float* B, float* C, int M, int K, int N) {
    //each thread takes care of C[b][i][j]
    int batch = blockIdx.x;


    //blockIdx.y will have m * n blocks
    int numTilesX = (N + blocksize - 1) / blocksize;
    int row = (blockIdx.y / numTilesX);
    int col = (blockIdx.y % numTilesX);

    int i = (row * blockDim.y) + threadIdx.y;
    int j = (col * blockDim.x) + threadIdx.x;
    
    if (i < M && j < N) {
        float sum = 0;
        for (int k = 0; k < K; k ++) {
            sum += A[batch * M * K + i * K + k] * B[batch * K * N + k * N + j];
        }
        C[batch * M * N + i * N + j] = sum;
    }
}


__global__ void batchStridedMul(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta) {
    int batch = blockIdx.x;

    //C[b][i][j]

    //which block we are in
    int numTiles = (N + blocksize - 1) / blocksize;
    int blockRow = blockIdx.y / numTiles;
    int blockCol = blockIdx.y % numTiles;


    int r = threadIdx.y;
    int c = threadIdx.x;

    int i = blockRow * blockDim.y + r;
    int j = blockCol * blockDim.x + c;

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
        if (beta == 0.0f) {
            Cb[idx] = sum * (alpha);
        } else {
            Cb[idx] = sum * alpha + beta * Cb[idx];
        }
    }
}

