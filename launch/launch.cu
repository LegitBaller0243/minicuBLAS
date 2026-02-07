#include "launch/tiling_launch.h"

#include "kernels/tiling_kernels.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

void print_usage(const char* prog) {
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

bool parse_args(int argc, char** argv, Options& opt) {
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

int run_naive(const Options& opt) {
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

int run_tiled(const Options& opt) {
    std::cout << "Running tiled: N=" << opt.N
              << " repeats=" << opt.repeats << "\n";

    // TODO: Allocate buffers and memcpy for square N x N case.
    // TODO: Compute dim3 block/grid.
    // TODO: Use time_kernel_ms around tilingMul<<<...>>>(...)
    return 0;
}

int run_batch_naive(const Options& opt) {
    std::cout << "Running batch-naive: batch=" << opt.batch
              << " M=" << opt.M << " K=" << opt.K << " N=" << opt.N << "\n";

    const size_t a_elems = static_cast<size_t>(opt.batch) * opt.M * opt.K;
    const size_t b_elems = static_cast<size_t>(opt.batch) * opt.K * opt.N;
    const size_t c_elems = static_cast<size_t>(opt.batch) * opt.M * opt.N;
    const size_t a_bytes = a_elems * sizeof(float);
    const size_t b_bytes = b_elems * sizeof(float);
    const size_t c_bytes = c_elems * sizeof(float);

    float* h_A = nullptr;
    float* h_B = nullptr;
    cudaMallocHost(&h_A, a_bytes);
    cudaMallocHost(&h_B, b_bytes);
    for (size_t i = 0; i < a_elems; ++i) h_A[i] = 1.0f;
    for (size_t i = 0; i < b_elems; ++i) h_B[i] = 1.0f;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, a_bytes);
    cudaMalloc(&d_B, b_bytes);
    cudaMalloc(&d_C, c_bytes);
    cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice);
    // TODO: Compute dim3 block/grid (grid.x = batch, grid.y = tiles).
    // TODO: Use time_kernel_ms around batchNaiveMul<<<...>>>(...)
    return 0;
}

int run_batch_tiled(const Options& opt) {
    std::cout << "Running batch-tiled: batch=" << opt.batch
              << " M=" << opt.M << " K=" << opt.K << " N=" << opt.N << "\n";

    const size_t a_elems = static_cast<size_t>(opt.batch) * opt.M * opt.K;
    const size_t b_elems = static_cast<size_t>(opt.batch) * opt.K * opt.N;
    const size_t c_elems = static_cast<size_t>(opt.batch) * opt.M * opt.N;
    const size_t a_bytes = a_elems * sizeof(float);
    const size_t b_bytes = b_elems * sizeof(float);
    const size_t c_bytes = c_elems * sizeof(float);

    float* h_A = nullptr;
    float* h_B = nullptr;
    cudaMallocHost(&h_A, a_bytes);
    cudaMallocHost(&h_B, b_bytes);
    for (size_t i = 0; i < a_elems; ++i) h_A[i] = 1.0f;
    for (size_t i = 0; i < b_elems; ++i) h_B[i] = 1.0f;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, a_bytes);
    cudaMalloc(&d_B, b_bytes);
    cudaMalloc(&d_C, c_bytes);
    cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice);
    // TODO: Compute dim3 block/grid (grid.x = batch, grid.y = tiles).
    // TODO: Use time_kernel_ms around batchTilingMul<<<...>>>(...)
    return 0;
}
