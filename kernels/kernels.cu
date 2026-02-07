#include "kernels.cuh"

//matmul kernel
// each thread exists within a block and we use math to turn 1D grid to a 2D grid index of that block
__global__ void naiveMul(float* A, float* B, float* C, int M, int K, int N) {
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
        C[x * N + y] = sum;
    }
}
//tiling kernel on a 1D grid
__global__ void tilingMul(const float* A, const float* B, float* C, int N) {
    int numTiles = (N + blocksize - 1)/ blocksize;
    int blockRow = blockIdx.x / numTiles;
    int blockCol = blockIdx.x % numTiles;


    int block_start_x = blockRow * blocksize;
    int block_start_y = blockCol * blocksize;


    int r = threadIdx.y;
    int c = threadIdx.x;

    __shared__ float A_block[blocksize][blocksize];
    __shared__ float B_block[blocksize][blocksize];

    float sum = 0;
    int x = block_start_x + r;
    int y = block_start_y + c; 


    for (int kk = 0; kk < N; kk += blocksize) {
        A_block[r][c] = (x < N && (kk + c) < N) ? A[x * N + kk + c] : 0;
        B_block[r][c] = (y < N && (kk + r) < N) ? B[(kk + r) * N + y] : 0;
        __syncthreads();

        for (int k = 0; k < blocksize; k ++) {
            sum += A_block[r][k] * B_block[k][c]; 
        }
        __syncthreads();
    }
    if (x < N && y < N) C[x * N + y] = sum;
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


__global__ void batchTilingMul(const float* A, const float* B, float* C, int M, int K, int N) {
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
    // for each block load memory into data
    for (int k = 0; k < K; k += blocksize) {
        A_block[r][c] = (i < M && (k + c) < K)? A[batch * M * K + i * K + k + c] : 0;
        B_block[r][c] = (j < N && (r + k) < K)? B[batch * K * N + (k + r) * N + j] : 0;
        __syncthreads();

        for (int kk = 0; kk < blocksize; kk++) {
            sum += A_block[r][kk] * B_block[kk][c];
        }
        __syncthreads();
    }

    if (i < M && j < N) C[batch * M * N + i * N + j] = sum;


}
