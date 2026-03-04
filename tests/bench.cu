#include "../kernels/kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
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

static bool cublas_row_major_gemm_strided_batched(cublasHandle_t handle, const float* d_A,
                                                  const float* d_B, float* d_C, int batch, int M,
                                                  int K, int N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int m = N;
    const int n = M;
    const int k = K;
    const int ldaB = N;
    const int ldaA = K;
    const int ldc = N;
    const long long strideA = static_cast<long long>(M) * K;
    const long long strideB = static_cast<long long>(K) * N;
    const long long strideC = static_cast<long long>(M) * N;
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_B,
                                           ldaB, strideB, d_A, ldaA, strideA, &beta, d_C, ldc,
                                           strideC, batch));
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
              << std::setw(14) << "kernel_ms" << std::setw(14) << "cublas_ms"
              << std::setw(14) << "kernel_GF/s" << std::setw(14) << "cublas_GF/s"
              << std::setw(12) << "speedup" << std::setw(12) << "max_abs" << "\n";
    if (!batched) {
        std::cout << std::left << std::setw(12) << "(MxKxN)" << std::setw(8) << "-"
                  << "\n";
    }
}

static void print_row(const Size3D& s, int batch, float kernel_ms, float cublas_ms,
                      float kernel_gflops, float cublas_gflops, float max_abs) {
    const float speedup = kernel_ms > 0.0f ? (cublas_ms / kernel_ms) : 0.0f;
    std::string label = std::to_string(s.M) + "x" + std::to_string(s.K) + "x" + std::to_string(s.N);
    std::cout << std::left << std::setw(12) << label << std::setw(8) << batch << std::setw(14)
              << std::fixed << std::setprecision(4) << kernel_ms << std::setw(14) << cublas_ms
              << std::setw(14) << std::setprecision(2) << kernel_gflops << std::setw(14)
              << cublas_gflops << std::setw(12) << std::setprecision(2) << speedup
              << std::setw(12) << std::setprecision(3) << max_abs << "\n";
}

static bool run_single_suite(const std::string& name, const std::vector<Size3D>& sizes,
                             const Config& cfg,
                             const std::function<void(dim3, dim3, float*, float*, float*,
                                                      int, int, int)>& launch_kernel) {
    print_header(name);

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::mt19937 rng(1234);
    for (const Size3D& s : sizes) {
        const size_t a_elems = static_cast<size_t>(s.M) * s.K;
        const size_t b_elems = static_cast<size_t>(s.K) * s.N;
        const size_t c_elems = static_cast<size_t>(s.M) * s.N;
        const size_t a_bytes = a_elems * sizeof(float);
        const size_t b_bytes = b_elems * sizeof(float);
        const size_t c_bytes = c_elems * sizeof(float);

        std::vector<float> h_A(a_elems), h_B(b_elems), h_kernel(c_elems), h_cublas(c_elems);
        fill_random(h_A, rng);
        fill_random(h_B, rng);

        float *d_A = nullptr, *d_B = nullptr, *d_kernel = nullptr, *d_cublas = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
        CUDA_CHECK(cudaMalloc(&d_kernel, c_bytes));
        CUDA_CHECK(cudaMalloc(&d_cublas, c_bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_kernel, 0, c_bytes));
        CUDA_CHECK(cudaMemset(d_cublas, 0, c_bytes));

        dim3 block(blocksize, blocksize);
        const int tiles_m = (s.M + blocksize - 1) / blocksize;
        const int tiles_n = (s.N + blocksize - 1) / blocksize;
        dim3 grid(tiles_n, tiles_m);

        launch_kernel(grid, block, d_A, d_B, d_kernel, s.M, s.K, s.N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!cublas_row_major_gemm(handle, d_A, d_B, d_cublas, s.M, s.K, s.N)) return false;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_kernel.data(), d_kernel, c_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_cublas, c_bytes, cudaMemcpyDeviceToHost));
        const float diff = max_abs_diff(h_kernel, h_cublas);

        float kernel_ms = 0.0f;
        float cublas_ms = 0.0f;
        if (!time_cuda_ms(cfg.warmup, cfg.repeats, [&]() {
                launch_kernel(grid, block, d_A, d_B, d_kernel, s.M, s.K, s.N);
            }, kernel_ms))
            return false;
        bool cublas_ok = true;
        if (!time_cuda_ms(cfg.warmup, cfg.repeats, [&]() {
                cublas_ok &= cublas_row_major_gemm(handle, d_A, d_B, d_cublas, s.M, s.K, s.N);
            }, cublas_ms))
            return false;
        if (!cublas_ok) return false;

        const double flops = 2.0 * static_cast<double>(s.M) * s.K * s.N;
        const float kernel_gflops = static_cast<float>(flops / (kernel_ms * 1.0e6));
        const float cublas_gflops = static_cast<float>(flops / (cublas_ms * 1.0e6));
        print_row(s, 1, kernel_ms, cublas_ms, kernel_gflops, cublas_gflops, diff);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_kernel));
        CUDA_CHECK(cudaFree(d_cublas));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return true;
}

static bool run_batched_suite(const std::string& name, const std::vector<Size3D>& sizes,
                              const Config& cfg,
                              const std::function<void(dim3, dim3, float*, float*,
                                                       float*, int, int, int, int)>& launch_kernel) {
    print_header(name, true);

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::mt19937 rng(1234);
    for (const Size3D& s : sizes) {
        const size_t a_elems = static_cast<size_t>(cfg.batch) * s.M * s.K;
        const size_t b_elems = static_cast<size_t>(cfg.batch) * s.K * s.N;
        const size_t c_elems = static_cast<size_t>(cfg.batch) * s.M * s.N;
        const size_t a_bytes = a_elems * sizeof(float);
        const size_t b_bytes = b_elems * sizeof(float);
        const size_t c_bytes = c_elems * sizeof(float);

        std::vector<float> h_A(a_elems), h_B(b_elems), h_kernel(c_elems), h_cublas(c_elems);
        fill_random(h_A, rng);
        fill_random(h_B, rng);

        float *d_A = nullptr, *d_B = nullptr, *d_kernel = nullptr, *d_cublas = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
        CUDA_CHECK(cudaMalloc(&d_kernel, c_bytes));
        CUDA_CHECK(cudaMalloc(&d_cublas, c_bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_kernel, 0, c_bytes));
        CUDA_CHECK(cudaMemset(d_cublas, 0, c_bytes));

        dim3 block(blocksize, blocksize);
        const int tiles_m = (s.M + blocksize - 1) / blocksize;
        const int tiles_n = (s.N + blocksize - 1) / blocksize;
        dim3 grid(cfg.batch, tiles_n, tiles_m);

        launch_kernel(grid, block, d_A, d_B, d_kernel, cfg.batch, s.M, s.K, s.N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!cublas_row_major_gemm_strided_batched(handle, d_A, d_B, d_cublas, cfg.batch, s.M, s.K,
                                                   s.N))
            return false;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_kernel.data(), d_kernel, c_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_cublas, c_bytes, cudaMemcpyDeviceToHost));
        const float diff = max_abs_diff(h_kernel, h_cublas);

        float kernel_ms = 0.0f;
        float cublas_ms = 0.0f;
        if (!time_cuda_ms(cfg.warmup, cfg.repeats, [&]() {
                launch_kernel(grid, block, d_A, d_B, d_kernel, cfg.batch, s.M, s.K, s.N);
            }, kernel_ms))
            return false;
        bool cublas_ok = true;
        if (!time_cuda_ms(cfg.warmup, cfg.repeats, [&]() {
                cublas_ok &= cublas_row_major_gemm_strided_batched(handle, d_A, d_B, d_cublas,
                                                                    cfg.batch, s.M, s.K, s.N);
            }, cublas_ms))
            return false;
        if (!cublas_ok) return false;

        const double flops = 2.0 * static_cast<double>(cfg.batch) * s.M * s.K * s.N;
        const float kernel_gflops = static_cast<float>(flops / (kernel_ms * 1.0e6));
        const float cublas_gflops = static_cast<float>(flops / (cublas_ms * 1.0e6));
        print_row(s, cfg.batch, kernel_ms, cublas_ms, kernel_gflops, cublas_gflops, diff);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_kernel));
        CUDA_CHECK(cudaFree(d_cublas));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return true;
}

static bool parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--kernel" && i + 1 < argc) {
            cfg.kernel = argv[++i];
        } else if (arg == "--repeats" && i + 1 < argc) {
            cfg.repeats = std::atoi(argv[++i]);
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

    if (cfg.repeats <= 0 || cfg.warmup < 0 || cfg.batch <= 0) {
        std::cerr << "Expected repeats > 0, warmup >= 0, batch > 0\n";
        return false;
    }
    return true;
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [--kernel all|naive|tiled|batch-naive|batch-tiled]"
              << " [--repeats N] [--warmup N] [--batch N]\n";
}

int main(int argc, char** argv) {
    Config cfg{};
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Benchmark config: repeats=" << cfg.repeats << " warmup=" << cfg.warmup
              << " batch=" << cfg.batch << "\n";

    const std::vector<Size3D> sizes = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 256, 512},
        {1024, 1024, 1024},
    };

    const bool run_all = (cfg.kernel == "all");
    bool ok = true;

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
        cfg.kernel != "batch-tiled") {
        std::cerr << "Unsupported --kernel value: " << cfg.kernel << "\n";
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
