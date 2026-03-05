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
__global__ void sharedTilingMul(const float* __restrict__ A, const float* __restrict__ B,
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
// Assumptions for 1-dimension mapping and vectorized loads in this kernel:
// 1) Launch shape is blockDim == (blocksize, blocksize), so tx in [0, blocksize).
// 2) K % 4 == 0 (A) and N % 4 == 0 (B) and we cudaMalloced (A, B)
__global__ void regSharedTilingMul(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C,
        int M, int N, int K,
        float alpha) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;

    int i = blockRow + ty * THREAD_TILE_Y;
    int j = blockCol + tx * THREAD_TILE_X;

    float accum[THREAD_TILE_Y][THREAD_TILE_X];
    __shared__ float A_block[BLOCK_M][BLOCK_K];
    __shared__ float B_block[BLOCK_K][BLOCK_N];

    for (int y = 0; y < THREAD_TILE_Y; y += 1) {
        for (int x = 0; x < THREAD_TILE_X; x += 1) {
            accum[y][x] = 0.0f;
        }
    }

    for (int kk = 0; kk < K; kk += BLOCK_K) {
        for (int y = ty; y < BLOCK_M; y += blockDim.y) {
            for (int x = tx * 4; x < BLOCK_K; x += blockDim.x * 4) {
                int global_row = blockRow + y;
                int global_col = kk + x;

                if (global_row < M && (global_col + 3) < K) {
                    *((float4*)&A_block[y][x]) =
                        *((const float4*)&A[global_row * K + global_col]);
                } else {
                    for (int lane = 0; lane < 4; lane += 1) {
                        int col = global_col + lane;
                        A_block[y][x + lane] = (global_row < M && col < K) ?
                            A[global_row * K + col] : 0.0f;
                    }
                }
            }
        }
        for (int y = ty; y < BLOCK_K; y += blockDim.y) {
            for (int x = tx * 4; x < BLOCK_N; x += blockDim.x * 4) {
                int global_row = kk + y;
                int global_col = blockCol + x;

                if (global_row < K && (global_col + 3) < N) {
                    *((float4*)&B_block[y][x]) =
                        *((const float4*)&B[global_row * N + global_col]);
                } else {
                    for (int lane = 0; lane < 4; lane += 1) {
                        int col = global_col + lane;
                        B_block[y][x + lane] = (global_row < K && col < N) ?
                            B[global_row * N + col] : 0.0f;
                    }
                }
            }
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_K; k += 1) {
            for (int y = 0; y < THREAD_TILE_Y; y += 1) {
                for (int x = 0; x < THREAD_TILE_X; x += 1) {
                    accum[y][x] += A_block[ty * THREAD_TILE_Y + y][k]
                        * B_block[k][tx * THREAD_TILE_X + x];
                }
            }
        }

        __syncthreads();
    }
    for (int y = 0; y < THREAD_TILE_Y; y += 1) {
        for (int x = 0; x < THREAD_TILE_X; x += 1) {
            int global_row = i + y;
            int global_col = j + x;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = accum[y][x] * alpha;
            }
        }
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
