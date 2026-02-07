#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

# define blocksize 16


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


struct Options {
    int M = 1024;
    int K = 256;
    int N = 128;
    int batch = 1;
    int repeats = 10;
    std::string kernel = "naive";
};

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "Options:\n"
        << "  --kernel naive|tiled|batch-naive|batch-tiled\n"
        << "  --m <int>    rows of A / C\n"
        << "  --k <int>    cols of A / rows of B\n"
        << "  --n <int>    cols of B / C\n"
        << "  --batch <int> batch size for batched kernels\n"
        << "  --repeats <int> number of timed launches\n"
        << "  -h, --help   show this help\n";
}

static bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (!std::strcmp(arg, "--kernel") && i + 1 < argc) {
            opt.kernel = argv[++i];
        } else if (!std::strcmp(arg, "--m") && i + 1 < argc) {
            opt.M = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--k") && i + 1 < argc) {
            opt.K = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--n") && i + 1 < argc) {
            opt.N = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--batch") && i + 1 < argc) {
            opt.batch = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--repeats") && i + 1 < argc) {
            opt.repeats = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "-h") || !std::strcmp(arg, "--help")) {
            return false;
        } else {
            std::cerr << "Unknown or incomplete arg: " << arg << "\n";
            return false;
        }
    }
    return true;
}

template <typename LaunchFn>
static float time_kernel_ms(int repeats, LaunchFn launch) {
    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up (1 launch)
    launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeats;
}

static int run_naive(const Options& opt) {
    std::cout << "Running naive: M=" << opt.M << " K=" << opt.K
              << " N=" << opt.N << " repeats=" << opt.repeats << "\n";

    // TODO: Allocate host/device buffers and memcpy (you said you want to do this).
    float* h_A = malloc(sizeof(float) * opt.M * opt.K);
    float* h_B = malloc(sizeof(float) * opt.K * opt.N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * opt.M * opt.K);
    cudaMalloc(&d_B, sizeof(float) * opt.K * opt.N);
    cudaMalloc(&d_C, sizeof(float) * opt.M * opt.N);


    cudaMemcpy(d_A, A, sizeof(float) * opt.M * opt.K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * opt.N * opt.K, cudaMemcpyHostToDevice);
    // TODO: Compute dim3 block/grid.
    dim3 block = (blocksize, blocksize);
    dim3 grid = ((M + blocksize - 1) / blocksize, (N + blocksize - 1) / blocksize);
    // TODO: Wrap your launch in time_kernel_ms like below.
    
    float avg_ms = time_kernel_ms(opt.repeats, [&]() {
        naiveMul<<<grid, block>>>(d_A, d_B, d_C, opt.M, opt.K, opt.N);
    });
    std::cout << "avg_ms=" << avg_ms << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

static int run_tiled(const Options& opt) {
    std::cout << "Running tiled: N=" << opt.N
              << " repeats=" << opt.repeats << "\n";

    // TODO: Allocate buffers and memcpy for square N x N case.
    float* h_A = malloc(sizeof(float) * opt.N * opt.N);
    float* h_B = malloc(sizeof(float) * opt.N * opt.N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * opt.N * opt.N);
    cudaMalloc(&d_B, sizeof(float) * opt.N * opt.N);
    cudaMalloc(&d_C, sizeof(float) * opt.N * opt.N);
    // TODO: Compute dim3 block/grid.
    dim3 block = (blocksize, blocksize);
    dim3 grid = ((N + blocksize - 1) / blocksize, (N + blocksize - 1) / blocksize);
    // TODO: Use time_kernel_ms around tilingMul<<<...>>>(...)
    float avg_ms = time_kernel_ms(opt.repeats, [&]() {
        naiveMul<<<grid, block>>>(d_A, d_B, d_C, opt.M, opt.K, opt.N);
    });
    std::cout << "avg_ms=" << avg_ms << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

static int run_batch_naive(const Options& opt) {
    std::cout << "Running batch-naive: batch=" << opt.batch
              << " M=" << opt.M << " K=" << opt.K << " N=" << opt.N << "\n";

    // TODO: Allocate buffers and memcpy for batched case.
    const size_t bytesA = opt.batch * opt.M * opt.K * sizeof(float);
    const size_t bytesB = opt.batch * opt.K * opt.N * sizeof(float);
    const size_t bytesC = opt.batch * opt.M * opt.N * sizeof(float);

    // Host buffers
    std::vector<float> hA(opt.batch * opt.M * opt.K);
    std::vector<float> hB(opt.batch * opt.K * opt.N);
    std::vector<float> hC(opt.batch * opt.M * opt.N);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);

    cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice);


    // TODO: Compute dim3 block/grid (grid.x = batch, grid.y = tiles).
    dim3 block = (blocksize, blocksize);
    dim3 grid = (opt.batch, (M + block.x - 1) / block.x * (N + block.y - 1) / block.y);

    // TODO: Use time_kernel_ms around batchNaiveMul<<<...>>>(...)
    float avg_ms = time_kernel_ms(opt.repeats, [&]() {
        naiveMul<<<grid, block>>>(d_A, d_B, d_C, opt.M, opt.K, opt.N);
    });
    std::cout << "avg_ms=" << avg_ms << "\n";
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

static int run_batch_tiled(const Options& opt) {
    std::cout << "Running batch-tiled: batch=" << opt.batch
              << " M=" << opt.M << " K=" << opt.K << " N=" << opt.N << "\n";

    const size_t bytesA = opt.batch * opt.M * opt.K * sizeof(float);
    const size_t bytesB = opt.batch * opt.K * opt.N * sizeof(float);
    const size_t bytesC = opt.batch * opt.M * opt.N * sizeof(float);

    // Host buffers
    std::vector<float> hA(opt.batch * opt.M * opt.K);
    std::vector<float> hB(opt.batch * opt.K * opt.N);
    std::vector<float> hC(opt.batch * opt.M * opt.N);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);

    cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice);


    // Compute dim3 block/grid (grid.x = batch, grid.y = tiles).
    dim3 block = (blocksize, blocksize);
    dim3 grid = (opt.batch, (M + block.x - 1) / block.x * (N + block.y - 1) / block.y);

    //Use time_kernel_ms around batchNaiveMul<<<...>>>(...)
    float avg_ms = time_kernel_ms(opt.repeats, [&]() {
        naiveMul<<<grid, block>>>(d_A, d_B, d_C, opt.M, opt.K, opt.N);
    });
    std::cout << "avg_ms=" << avg_ms << "\n";
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    if (opt.kernel == "naive") return run_naive(opt);
    if (opt.kernel == "tiled") return run_tiled(opt);
    if (opt.kernel == "batch-naive") return run_batch_naive(opt);
    if (opt.kernel == "batch-tiled") return run_batch_tiled(opt);

    std::cerr << "Unknown kernel: " << opt.kernel << "\n";
    print_usage(argv[0]);
    return 1;
}

