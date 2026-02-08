#include "launch/launch.h"
#include "kernels/kernels.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct MatmulBuffers {
    float* h_A = nullptr;
    float* h_B = nullptr;
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
};

static void init_host(float* ptr, size_t elems, float value) {
    for (size_t i = 0; i < elems; ++i) ptr[i] = value;
}

static int alloc_and_copy_single(const Options& opt, MatmulBuffers& buf) {
    const size_t a_elems = static_cast<size_t>(opt.M) * opt.K;
    const size_t b_elems = static_cast<size_t>(opt.K) * opt.N;
    const size_t c_elems = static_cast<size_t>(opt.M) * opt.N;
    const size_t a_bytes = a_elems * sizeof(float);
    const size_t b_bytes = b_elems * sizeof(float);
    const size_t c_bytes = c_elems * sizeof(float);

    if (cudaMallocHost(&buf.h_A, a_bytes) != cudaSuccess) return 1;
    if (cudaMallocHost(&buf.h_B, b_bytes) != cudaSuccess) return 1;
    init_host(buf.h_A, a_elems, 1.0f);
    init_host(buf.h_B, b_elems, 1.0f);

    if (cudaMalloc(&buf.d_A, a_bytes) != cudaSuccess) return 1;
    if (cudaMalloc(&buf.d_B, b_bytes) != cudaSuccess) return 1;
    if (cudaMalloc(&buf.d_C, c_bytes) != cudaSuccess) return 1;
    if (cudaMemcpy(buf.d_A, buf.h_A, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemcpy(buf.d_B, buf.h_B, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemset(buf.d_C, 0, c_bytes) != cudaSuccess) return 1;
    return 0;
}

static int alloc_and_copy_batched(const Options& opt, MatmulBuffers& buf) {
    const size_t a_elems = static_cast<size_t>(opt.batch) * opt.M * opt.K;
    const size_t b_elems = static_cast<size_t>(opt.batch) * opt.K * opt.N;
    const size_t c_elems = static_cast<size_t>(opt.batch) * opt.M * opt.N;
    const size_t a_bytes = a_elems * sizeof(float);
    const size_t b_bytes = b_elems * sizeof(float);
    const size_t c_bytes = c_elems * sizeof(float);

    if (cudaMallocHost(&buf.h_A, a_bytes) != cudaSuccess) return 1;
    if (cudaMallocHost(&buf.h_B, b_bytes) != cudaSuccess) return 1;
    init_host(buf.h_A, a_elems, 1.0f);
    init_host(buf.h_B, b_elems, 1.0f);

    if (cudaMalloc(&buf.d_A, a_bytes) != cudaSuccess) return 1;
    if (cudaMalloc(&buf.d_B, b_bytes) != cudaSuccess) return 1;
    if (cudaMalloc(&buf.d_C, c_bytes) != cudaSuccess) return 1;
    if (cudaMemcpy(buf.d_A, buf.h_A, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemcpy(buf.d_B, buf.h_B, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemset(buf.d_C, 0, c_bytes) != cudaSuccess) return 1;
    return 0;
}

static void free_buffers(MatmulBuffers& buf) {
    if (buf.d_A) cudaFree(buf.d_A);
    if (buf.d_B) cudaFree(buf.d_B);
    if (buf.d_C) cudaFree(buf.d_C);
    if (buf.h_A) cudaFreeHost(buf.h_A);
    if (buf.h_B) cudaFreeHost(buf.h_B);
}

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "Options:\n"
        << "  --kernel naive|tiled|transpose-tiled|batch-naive|batch-tiled\n"
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

    MatmulBuffers buf{};
    if (alloc_and_copy_single(opt, buf) != 0) {
        std::cerr << "Allocation/memcpy failed in run_naive\n";
        free_buffers(buf);
        return 1;
    }

    dim3 block(blocksize, blocksize);
    const int tiles_m = (opt.M + blocksize - 1) / blocksize;
    const int tiles_n = (opt.N + blocksize - 1) / blocksize;
    dim3 grid(tiles_m * tiles_n);

    float avg_ms = time_kernel_ms(opt.repeats, [&]() {
        naiveMul<<<grid, block>>>(buf.d_A, buf.d_B, buf.d_C, opt.M, opt.K, opt.N, 1.0f, 0.0f);
    });
    std::cout << "avg_ms=" << avg_ms << "\n";

    free_buffers(buf);
    return 0;
}

int run_tiled(const Options& opt) {
    std::cout << "Running tiled: M=" << opt.M << " K=" << opt.K << " N=" << opt.N
              << " repeats=" << opt.repeats << "\n";

    MatmulBuffers buf{};
    if (alloc_and_copy_single(opt, buf) != 0) {
        std::cerr << "Allocation/memcpy failed in run_tiled\n";
        free_buffers(buf);
        return 1;
    }

    dim3 block(blocksize, blocksize);
    const int tiles_m = (opt.M + blocksize - 1) / blocksize;
    const int tiles_n = (opt.N + blocksize - 1) / blocksize;
    dim3 grid(tiles_m * tiles_n);

    float avg_ms = time_kernel_ms(opt.repeats, [&]() {
        tilingMul<<<grid, block>>>(buf.d_A, buf.d_B, buf.d_C, opt.M, opt.N, opt.K, 1.0f, 0.0f);
    });
    std::cout << "avg_ms=" << avg_ms << "\n";

    free_buffers(buf);
    return 0;
}

int run_transpose_tiled(const Options& opt) {
    std::cout << "Running transpose-tiled: M=" << opt.M << " K=" << opt.K << " N=" << opt.N
              << " repeats=" << opt.repeats << " transA=T transB=N\n";

    MatmulBuffers buf{};
    if (alloc_and_copy_single(opt, buf) != 0) {
        std::cerr << "Allocation/memcpy failed in run_transpose_tiled\n";
        free_buffers(buf);
        return 1;
    }

    dim3 block(blocksize, blocksize);
    const int tiles_m = (opt.M + blocksize - 1) / blocksize;
    const int tiles_n = (opt.N + blocksize - 1) / blocksize;
    dim3 grid(tiles_m * tiles_n);
    const Transpose trans{OP_T, OP_N};

    float avg_ms = time_kernel_ms(opt.repeats, [&]() {
        transposeTilingMul<<<grid, block>>>(buf.d_A, buf.d_B, buf.d_C, opt.M, opt.N, opt.K, 1.0f, 0.0f, trans);
    });
    std::cout << "avg_ms=" << avg_ms << "\n";

    free_buffers(buf);
    return 0;
}

int run_batch_naive(const Options& opt) {
    std::cout << "Running batch-naive: batch=" << opt.batch
              << " M=" << opt.M << " K=" << opt.K << " N=" << opt.N << "\n";

    MatmulBuffers buf{};
    if (alloc_and_copy_batched(opt, buf) != 0) {
        std::cerr << "Allocation/memcpy failed in run_batch_naive\n";
        free_buffers(buf);
        return 1;
    }

    dim3 block(blocksize, blocksize);
    const int tiles_m = (opt.M + blocksize - 1) / blocksize;
    const int tiles_n = (opt.N + blocksize - 1) / blocksize;
    dim3 grid(opt.batch, tiles_m * tiles_n);

    float avg_ms = time_kernel_ms(opt.repeats, [&]() {
        batchNaiveMul<<<grid, block>>>(buf.d_A, buf.d_B, buf.d_C, opt.M, opt.K, opt.N);
    });
    std::cout << "avg_ms=" << avg_ms << "\n";

    free_buffers(buf);
    return 0;
}

int run_batch_tiled(const Options& opt) {
    std::cout << "Running batch-tiled: batch=" << opt.batch
              << " M=" << opt.M << " K=" << opt.K << " N=" << opt.N << "\n";

    MatmulBuffers buf{};
    if (alloc_and_copy_batched(opt, buf) != 0) {
        std::cerr << "Allocation/memcpy failed in run_batch_tiled\n";
        free_buffers(buf);
        return 1;
    }

    dim3 block(blocksize, blocksize);
    const int tiles_m = (opt.M + blocksize - 1) / blocksize;
    const int tiles_n = (opt.N + blocksize - 1) / blocksize;
    dim3 grid(opt.batch, tiles_m * tiles_n);

    float avg_ms = time_kernel_ms(opt.repeats, [&]() {
        batchStridedMul<<<grid, block>>>(buf.d_A, buf.d_B, buf.d_C, opt.M, opt.K, opt.N, 1.0f, 0.0f);
    });
    std::cout << "avg_ms=" << avg_ms << "\n";

    free_buffers(buf);
    return 0;
}
