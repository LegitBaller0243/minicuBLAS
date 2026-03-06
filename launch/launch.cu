#include "launch/launch.h"
#include "kernels/flash_attention.cuh"
#include "kernels/gemm-kernels.cuh"
#include "kernels/rk4_heat3d.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
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
        << "  --kernel naive|tiled|transpose-tiled|batch-naive|batch-tiled|flash-attn|rk4-heat3d\n"
        << "  --m <int>    rows of A / C\n"
        << "  --k <int>    cols of A / rows of B\n"
        << "  --n <int>    cols of B / C\n"
        << "  --batch <int> batch size for batched kernels\n"
        << "  --repeats <int> number of timed launches\n"
        << "Flash-attn mapping: B=batch H=n S=m D=k\n"
        << "RK4 mapping: nx=m ny=k nz=n\n"
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
static bool time_kernel_ms(int repeats, LaunchFn launch, float& avg_ms) {
    avg_ms = 0.0f;
    if (repeats <= 0) {
        std::cerr << "Invalid repeats: " << repeats << "\n";
        return false;
    }

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up (1 launch)
    launch();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed during warm-up: " << cudaGetErrorString(err) << "\n";
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        launch();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed during timed run: " << cudaGetErrorString(err) << "\n";
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return false;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    avg_ms = ms / repeats;
    return true;
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
    dim3 grid(tiles_n, tiles_m);

    float avg_ms = 0.0f;
    if (!time_kernel_ms(opt.repeats, [&]() {
        naiveMul<<<grid, block>>>(buf.d_A, buf.d_B, buf.d_C, opt.M, opt.K, opt.N, 1.0f);
    }, avg_ms)) {
        std::cerr << "Timing failed in run_naive\n";
        free_buffers(buf);
        return 1;
    }
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
    dim3 grid(tiles_n, tiles_m);

    float avg_ms = 0.0f;
    if (!time_kernel_ms(opt.repeats, [&]() {
        tilingMul<<<grid, block>>>(buf.d_A, buf.d_B, buf.d_C, opt.M, opt.N, opt.K, 1.0f);
    }, avg_ms)) {
        std::cerr << "Timing failed in run_tiled\n";
        free_buffers(buf);
        return 1;
    }
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
    dim3 grid(tiles_n, tiles_m);
    const Transpose trans{OP_T, OP_N};

    float avg_ms = 0.0f;
    if (!time_kernel_ms(opt.repeats, [&]() {
        transposeTilingMul<<<grid, block>>>(buf.d_A, buf.d_B, buf.d_C, opt.M, opt.N, opt.K, 1.0f, trans);
    }, avg_ms)) {
        std::cerr << "Timing failed in run_transpose_tiled\n";
        free_buffers(buf);
        return 1;
    }
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
    dim3 grid(opt.batch, tiles_n, tiles_m);

    float avg_ms = 0.0f;
    if (!time_kernel_ms(opt.repeats, [&]() {
        batchNaiveMul<<<grid, block>>>(buf.d_A, buf.d_B, buf.d_C, opt.M, opt.K, opt.N);
    }, avg_ms)) {
        std::cerr << "Timing failed in run_batch_naive\n";
        free_buffers(buf);
        return 1;
    }
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
    dim3 grid(opt.batch, tiles_n, tiles_m);

    float avg_ms = 0.0f;
    if (!time_kernel_ms(opt.repeats, [&]() {
        batchStridedMul<<<grid, block>>>(buf.d_A, buf.d_B, buf.d_C, opt.M, opt.K, opt.N, 1.0f);
    }, avg_ms)) {
        std::cerr << "Timing failed in run_batch_tiled\n";
        free_buffers(buf);
        return 1;
    }
    std::cout << "avg_ms=" << avg_ms << "\n";

    free_buffers(buf);
    return 0;
}

int run_flash_attn(const Options& opt) {
    const int B = opt.batch;
    const int H = opt.N;
    const int S = opt.M;
    const int D = opt.K;
    if (B <= 0 || H <= 0 || S <= 0 || D <= 0) {
        std::cerr << "Invalid flash-attn shape: B=" << B << " H=" << H << " S=" << S << " D=" << D << "\n";
        return 1;
    }

    std::cout << "Running flash-attn: B=" << B << " H=" << H << " S=" << S << " D=" << D
              << " repeats=" << opt.repeats << " causal=1\n";

    const size_t qkv_elems = static_cast<size_t>(B) * H * S * D;
    const size_t qkv_bytes = qkv_elems * sizeof(float);
    const size_t o_bytes = qkv_bytes;

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
    const auto free_flash = [&]() {
        if (d_Q) cudaFree(d_Q);
        if (d_K) cudaFree(d_K);
        if (d_V) cudaFree(d_V);
        if (d_O) cudaFree(d_O);
    };
    if (cudaMalloc(&d_Q, qkv_bytes) != cudaSuccess) return 1;
    if (cudaMalloc(&d_K, qkv_bytes) != cudaSuccess) {
        free_flash();
        return 1;
    }
    if (cudaMalloc(&d_V, qkv_bytes) != cudaSuccess) {
        free_flash();
        return 1;
    }
    if (cudaMalloc(&d_O, o_bytes) != cudaSuccess) {
        free_flash();
        return 1;
    }
    if (cudaMemset(d_Q, 0, qkv_bytes) != cudaSuccess) {
        free_flash();
        return 1;
    }
    if (cudaMemset(d_K, 0, qkv_bytes) != cudaSuccess) {
        free_flash();
        return 1;
    }
    if (cudaMemset(d_V, 0, qkv_bytes) != cudaSuccess) {
        free_flash();
        return 1;
    }
    if (cudaMemset(d_O, 0, o_bytes) != cudaSuccess) {
        free_flash();
        return 1;
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    float avg_ms = 0.0f;
    if (!time_kernel_ms(opt.repeats, [&]() {
        launchFlashAttnCausalForwardKernel(d_Q, d_K, d_V, d_O, B, H, S, D, scale, true);
    }, avg_ms)) {
        std::cerr << "Timing failed in run_flash_attn\n";
        free_flash();
        return 1;
    }
    std::cout << "avg_ms=" << avg_ms << "\n";

    free_flash();
    return 0;
}

int run_rk4_heat3d(const Options& opt) {
    const int nx = opt.M;
    const int ny = opt.K;
    const int nz = opt.N;
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        std::cerr << "Invalid rk4 grid: nx=" << nx << " ny=" << ny << " nz=" << nz << "\n";
        return 1;
    }

    std::cout << "Running rk4-heat3d: nx=" << nx << " ny=" << ny << " nz=" << nz
              << " repeats=" << opt.repeats << "\n";

    const size_t elems = static_cast<size_t>(nx) * ny * nz;
    const size_t bytes = elems * sizeof(float);

    float *u = nullptr, *u_next = nullptr, *k1 = nullptr, *k2 = nullptr, *k3 = nullptr, *k4 = nullptr, *u_stage = nullptr;
    const auto free_rk4 = [&]() {
        if (u) cudaFree(u);
        if (u_next) cudaFree(u_next);
        if (k1) cudaFree(k1);
        if (k2) cudaFree(k2);
        if (k3) cudaFree(k3);
        if (k4) cudaFree(k4);
        if (u_stage) cudaFree(u_stage);
    };
    if (cudaMalloc(&u, bytes) != cudaSuccess) return 1;
    if (cudaMalloc(&u_next, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMalloc(&k1, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMalloc(&k2, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMalloc(&k3, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMalloc(&k4, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMalloc(&u_stage, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMemset(u, 0, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMemset(u_next, 0, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMemset(k1, 0, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMemset(k2, 0, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMemset(k3, 0, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMemset(k4, 0, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }
    if (cudaMemset(u_stage, 0, bytes) != cudaSuccess) {
        free_rk4();
        return 1;
    }

    constexpr float alpha = 1.0f;
    constexpr float dt = 0.1f;
    constexpr float h = 1.0f;
    constexpr float boundary_value = 0.0f;
    const float inv_h2 = 1.0f / (h * h);

    float avg_ms = 0.0f;
    if (!time_kernel_ms(opt.repeats, [&]() {
        launchRK4Heat3DStepStaged(u, u_next, k1, k2, k3, k4, u_stage,
                                  nx, ny, nz, alpha, dt, inv_h2, boundary_value);
    }, avg_ms)) {
        std::cerr << "Timing failed in run_rk4_heat3d\n";
        free_rk4();
        return 1;
    }
    std::cout << "avg_ms=" << avg_ms << "\n";

    free_rk4();
    return 0;
}
