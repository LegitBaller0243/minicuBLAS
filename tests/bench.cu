#include "../kernels/gemm-kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        cudaError_t _err = (expr);                                               \
        if (_err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            return false;                                                        \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(expr)                                                       \
    do {                                                                         \
        cublasStatus_t _st = (expr);                                             \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                      \
            std::cerr << "cuBLAS error: " << static_cast<int>(_st)               \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            return false;                                                        \
        }                                                                        \
    } while (0)

struct Size3D {
    int M;
    int K;
    int N;
};

struct Config {
    std::string kernel = "all";
    int repeats = 50;
    int warmup = 10;
    int cpu_repeats = 3;
    int batch = 8;
};

static void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

template <typename F>
static bool time_cuda_ms(int warmup, int repeats, F launch, float& avg_ms) {
    for (int i = 0; i < warmup; ++i) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    avg_ms = total_ms / static_cast<float>(repeats);
    return true;
}

template <typename F>
static float time_cpu_ms(int repeats, F fn) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    const auto total_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    return static_cast<float>(total_ms / static_cast<double>(repeats));
}

static void cpu_row_major_gemm(const float* A, const float* B, float* C, int M, int K, int N) {
    std::fill(C, C + static_cast<size_t>(M) * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        float* c_row = C + static_cast<size_t>(i) * N;
        for (int k = 0; k < K; ++k) {
            const float a = A[static_cast<size_t>(i) * K + k];
            const float* b_row = B + static_cast<size_t>(k) * N;
            for (int j = 0; j < N; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

static void cpu_row_major_gemm_batched(const float* A, const float* B, float* C,
                                       int batch, int M, int K, int N) {
    const size_t strideA = static_cast<size_t>(M) * K;
    const size_t strideB = static_cast<size_t>(K) * N;
    const size_t strideC = static_cast<size_t>(M) * N;
    for (int b = 0; b < batch; ++b) {
        cpu_row_major_gemm(A + static_cast<size_t>(b) * strideA,
                           B + static_cast<size_t>(b) * strideB,
                           C + static_cast<size_t>(b) * strideC,
                           M, K, N);
    }
}

static bool cublas_row_major_gemm(cublasHandle_t handle, const float* d_A, const float* d_B,
                                  float* d_C, int M, int K, int N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int m = N;
    const int n = M;
    const int k = K;
    const int ldaB = N;
    const int ldaA = K;
    const int ldc = N;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_B, ldaB, d_A,
                             ldaA, &beta, d_C, ldc));
    return true;
}

static bool run_cublas_reference_once(const Config& cfg, const Size3D& s) {
    const size_t a_elems = static_cast<size_t>(s.M) * s.K;
    const size_t b_elems = static_cast<size_t>(s.K) * s.N;
    const size_t c_elems = static_cast<size_t>(s.M) * s.N;
    const size_t a_bytes = a_elems * sizeof(float);
    const size_t b_bytes = b_elems * sizeof(float);
    const size_t c_bytes = c_elems * sizeof(float);

    std::vector<float> h_A(a_elems), h_B(b_elems);
    std::mt19937 rng(777);
    fill_random(h_A, rng);
    fill_random(h_B, rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, c_bytes));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    float cublas_ms = 0.0f;
    bool cublas_ok = true;
    if (!time_cuda_ms(cfg.warmup, cfg.repeats, [&]() {
            cublas_ok &= cublas_row_major_gemm(handle, d_A, d_B, d_C, s.M, s.K, s.N);
        }, cublas_ms)) {
        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        return false;
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    if (!cublas_ok) return false;

    const double flops = 2.0 * static_cast<double>(s.M) * s.K * s.N;
    const float cublas_gflops = static_cast<float>(flops / (cublas_ms * 1.0e6));
    std::cout << "\n== cuBLAS Reference (single run) ==\n";
    std::cout << "size=" << s.M << "x" << s.K << "x" << s.N
              << " cublas_ms=" << std::fixed << std::setprecision(4) << cublas_ms
              << " cublas_GF/s=" << std::setprecision(2) << cublas_gflops << "\n";
    return true;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

static void print_header(const std::string& name, bool batched = false) {
    std::cout << "\n== " << name << " (blocksize=" << blocksize << ") ==\n";
    std::cout << std::left << std::setw(12) << "size" << std::setw(8) << "batch"
              << std::setw(14) << "kernel_ms" << std::setw(14) << "cpu_ms"
              << std::setw(14) << "speedup" << std::setw(14) << "kernel_GF/s"
              << std::setw(16) << "kernel_GB/s" << std::setw(12) << "max_abs" << "\n";
    if (!batched) {
        std::cout << std::left << std::setw(12) << "(MxKxN)" << std::setw(8) << "-"
                  << "\n";
    }
}

static void print_row(const Size3D& s, int batch, float kernel_ms, float cpu_ms,
                      float kernel_gflops, float kernel_gbps, float max_abs) {
    const float speedup = kernel_ms > 0.0f ? (cpu_ms / kernel_ms) : 0.0f;
    std::string label = std::to_string(s.M) + "x" + std::to_string(s.K) + "x" + std::to_string(s.N);
    std::cout << std::left << std::setw(12) << label << std::setw(8) << batch << std::setw(14)
              << std::fixed << std::setprecision(4) << kernel_ms << std::setw(14) << cpu_ms
              << std::setw(14) << std::setprecision(2) << speedup << std::setw(14)
              << kernel_gflops << std::setw(16) << kernel_gbps
              << std::setw(12) << std::setprecision(3) << max_abs << "\n";
}

static bool run_single_suite(const std::string& name, const std::vector<Size3D>& sizes,
                             const Config& cfg,
                             const std::function<void(dim3, dim3, float*, float*, float*,
                                                      int, int, int)>& launch_kernel) {
    print_header(name);

    std::mt19937 rng(1234);
    for (const Size3D& s : sizes) {
        const size_t a_elems = static_cast<size_t>(s.M) * s.K;
        const size_t b_elems = static_cast<size_t>(s.K) * s.N;
        const size_t c_elems = static_cast<size_t>(s.M) * s.N;
        const size_t a_bytes = a_elems * sizeof(float);
        const size_t b_bytes = b_elems * sizeof(float);
        const size_t c_bytes = c_elems * sizeof(float);

        std::vector<float> h_A(a_elems), h_B(b_elems), h_kernel(c_elems), h_cpu(c_elems), h_cpu_tmp(c_elems);
        fill_random(h_A, rng);
        fill_random(h_B, rng);

        float *d_A = nullptr, *d_B = nullptr, *d_kernel = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
        CUDA_CHECK(cudaMalloc(&d_kernel, c_bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_kernel, 0, c_bytes));

        dim3 block(blocksize, blocksize);
        const int tiles_m = (s.M + blocksize - 1) / blocksize;
        const int tiles_n = (s.N + blocksize - 1) / blocksize;
        dim3 grid(tiles_n, tiles_m);

        launch_kernel(grid, block, d_A, d_B, d_kernel, s.M, s.K, s.N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_kernel.data(), d_kernel, c_bytes, cudaMemcpyDeviceToHost));
        cpu_row_major_gemm(h_A.data(), h_B.data(), h_cpu.data(), s.M, s.K, s.N);
        const float diff = max_abs_diff(h_kernel, h_cpu);

        float kernel_ms = 0.0f;
        if (!time_cuda_ms(cfg.warmup, cfg.repeats, [&]() {
                launch_kernel(grid, block, d_A, d_B, d_kernel, s.M, s.K, s.N);
            }, kernel_ms))
            return false;
        const float cpu_ms = time_cpu_ms(cfg.cpu_repeats, [&]() {
            cpu_row_major_gemm(h_A.data(), h_B.data(), h_cpu_tmp.data(), s.M, s.K, s.N);
        });

        const double flops = 2.0 * static_cast<double>(s.M) * s.K * s.N;
        const float kernel_gflops = static_cast<float>(flops / (kernel_ms * 1.0e6));
        const double min_bytes = static_cast<double>(a_bytes + b_bytes + c_bytes);
        const float kernel_gbps = static_cast<float>(min_bytes / (kernel_ms * 1.0e6));
        print_row(s, 1, kernel_ms, cpu_ms, kernel_gflops, kernel_gbps, diff);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_kernel));
    }
    return true;
}

static bool run_batched_suite(const std::string& name, const std::vector<Size3D>& sizes,
                              const Config& cfg,
                              const std::function<void(dim3, dim3, float*, float*,
                                                       float*, int, int, int, int)>& launch_kernel) {
    print_header(name, true);

    std::mt19937 rng(1234);
    for (const Size3D& s : sizes) {
        const size_t a_elems = static_cast<size_t>(cfg.batch) * s.M * s.K;
        const size_t b_elems = static_cast<size_t>(cfg.batch) * s.K * s.N;
        const size_t c_elems = static_cast<size_t>(cfg.batch) * s.M * s.N;
        const size_t a_bytes = a_elems * sizeof(float);
        const size_t b_bytes = b_elems * sizeof(float);
        const size_t c_bytes = c_elems * sizeof(float);

        std::vector<float> h_A(a_elems), h_B(b_elems), h_kernel(c_elems), h_cpu(c_elems), h_cpu_tmp(c_elems);
        fill_random(h_A, rng);
        fill_random(h_B, rng);

        float *d_A = nullptr, *d_B = nullptr, *d_kernel = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
        CUDA_CHECK(cudaMalloc(&d_kernel, c_bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_kernel, 0, c_bytes));

        dim3 block(blocksize, blocksize);
        const int tiles_m = (s.M + blocksize - 1) / blocksize;
        const int tiles_n = (s.N + blocksize - 1) / blocksize;
        dim3 grid(cfg.batch, tiles_n, tiles_m);

        launch_kernel(grid, block, d_A, d_B, d_kernel, cfg.batch, s.M, s.K, s.N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_kernel.data(), d_kernel, c_bytes, cudaMemcpyDeviceToHost));
        cpu_row_major_gemm_batched(h_A.data(), h_B.data(), h_cpu.data(), cfg.batch, s.M, s.K, s.N);
        const float diff = max_abs_diff(h_kernel, h_cpu);

        float kernel_ms = 0.0f;
        if (!time_cuda_ms(cfg.warmup, cfg.repeats, [&]() {
                launch_kernel(grid, block, d_A, d_B, d_kernel, cfg.batch, s.M, s.K, s.N);
            }, kernel_ms))
            return false;
        const float cpu_ms = time_cpu_ms(cfg.cpu_repeats, [&]() {
            cpu_row_major_gemm_batched(h_A.data(), h_B.data(), h_cpu_tmp.data(), cfg.batch, s.M, s.K, s.N);
        });

        const double flops = 2.0 * static_cast<double>(cfg.batch) * s.M * s.K * s.N;
        const float kernel_gflops = static_cast<float>(flops / (kernel_ms * 1.0e6));
        const double min_bytes = static_cast<double>(a_bytes + b_bytes + c_bytes);
        const float kernel_gbps = static_cast<float>(min_bytes / (kernel_ms * 1.0e6));
        print_row(s, cfg.batch, kernel_ms, cpu_ms, kernel_gflops, kernel_gbps, diff);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_kernel));
    }
    return true;
}

static bool parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--kernel" && i + 1 < argc) {
            cfg.kernel = argv[++i];
        } else if (arg == "--repeats" && i + 1 < argc) {
            cfg.repeats = std::atoi(argv[++i]);
        } else if (arg == "--cpu-repeats" && i + 1 < argc) {
            cfg.cpu_repeats = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup = std::atoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            cfg.batch = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            return false;
        } else {
            std::cerr << "Unknown/incomplete argument: " << arg << "\n";
            return false;
        }
    }

    if (cfg.repeats <= 0 || cfg.cpu_repeats <= 0 || cfg.warmup < 0 || cfg.batch <= 0) {
        std::cerr << "Expected repeats > 0, cpu-repeats > 0, warmup >= 0, batch > 0\n";
        return false;
    }
    return true;
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [--kernel all|naive|tiled|reg-shared|register-tiling|batch-naive|batch-tiled]"
              << " [--repeats N] [--cpu-repeats N] [--warmup N] [--batch N]\n";
}

int main(int argc, char** argv) {
    Config cfg{};
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Benchmark config: repeats=" << cfg.repeats << " cpu_repeats=" << cfg.cpu_repeats
              << " warmup=" << cfg.warmup << " batch=" << cfg.batch << "\n";

    const std::vector<Size3D> sizes = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 256, 512},
        {1024, 1024, 1024},
    };

    const bool run_all = (cfg.kernel == "all");
    bool ok = true;

    // Print one cuBLAS anchor point once so results stay CPU-centric but still include vendor baseline.
    if (!run_cublas_reference_once(cfg, sizes.back())) return 1;

    if (run_all || cfg.kernel == "naive") {
        ok &= run_single_suite(
            "naiveMul", sizes, cfg,
            [](dim3 grid, dim3 block, float* d_A, float* d_B, float* d_C, int M, int K,
               int N) {
                naiveMul<<<grid, block>>>(d_A, d_B, d_C, M, K, N, 1.0f);
            });
    }

    if (run_all || cfg.kernel == "tiled") {
        ok &= run_single_suite(
            "tilingMul", sizes, cfg,
            [](dim3 grid, dim3 block, float* d_A, float* d_B, float* d_C, int M, int K,
               int N) { tilingMul<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f); });
    }

    if (run_all || cfg.kernel == "reg-shared" || cfg.kernel == "register-tiling") {
        ok &= run_single_suite(
            "registerTilingMul", sizes, cfg,
            [](dim3 /*grid*/, dim3 block, float* d_A, float* d_B, float* d_C, int M, int K,
               int N) {
                dim3 reg_grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
                regSharedTilingMul<<<reg_grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f);
            });
    }

    if (run_all || cfg.kernel == "batch-naive") {
        ok &= run_batched_suite(
            "batchNaiveMul", sizes, cfg,
            [](dim3 grid, dim3 block, float* d_A, float* d_B, float* d_C, int /*batch*/,
               int M, int K, int N) { batchNaiveMul<<<grid, block>>>(d_A, d_B, d_C, M, K, N); });
    }

    if (run_all || cfg.kernel == "batch-tiled") {
        ok &= run_batched_suite(
            "batchStridedMul", sizes, cfg,
            [](dim3 grid, dim3 block, float* d_A, float* d_B, float* d_C, int /*batch*/,
               int M, int K, int N) {
                batchStridedMul<<<grid, block>>>(d_A, d_B, d_C, M, K, N, 1.0f);
            });
    }

    if (!ok) return 1;
    if (!run_all && cfg.kernel != "naive" && cfg.kernel != "tiled" && cfg.kernel != "batch-naive" &&
        cfg.kernel != "batch-tiled" && cfg.kernel != "reg-shared" && cfg.kernel != "register-tiling") {
        std::cerr << "Unsupported --kernel value: " << cfg.kernel << "\n";
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
