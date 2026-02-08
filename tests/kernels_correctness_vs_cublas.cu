#include "../kernels/kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

static void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

static bool allclose(const std::vector<float>& got, const std::vector<float>& ref,
                     float atol, float rtol, const std::string& name) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t bad_idx = got.size();
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        const float tol = atol + rtol * std::fabs(ref[i]);
        if (diff > tol && bad_idx == got.size()) bad_idx = i;
        max_abs = std::max(max_abs, diff);
        if (std::fabs(ref[i]) > 0.0f) max_rel = std::max(max_rel, diff / std::fabs(ref[i]));
    }

    if (bad_idx != got.size()) {
        std::cerr << "[FAIL] " << name << " first mismatch at index " << bad_idx
                  << " got=" << got[bad_idx] << " ref=" << ref[bad_idx]
                  << " max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
        return false;
    }

    std::cout << "[PASS] " << name << " max_abs=" << max_abs << " max_rel=" << max_rel
              << "\n";
    return true;
}

static bool cublas_row_major_gemm(cublasHandle_t handle, const float* d_A, const float* d_B,
                                  float* d_C, int M, int K, int N, float alpha, float beta,
                                  cublasOperation_t opA_row, cublasOperation_t opB_row) {
    // Row-major C = opA(A) * opB(B), computed via column-major:
    // C_col = opB(B)_col * opA(A)_col, with dimensions (N x M).
    const int m = N;
    const int n = M;
    const int k = K;
    const int ldaB = N;
    const int ldaA = K;
    const int ldc = N;
    CUBLAS_CHECK(cublasSgemm(handle, opB_row, opA_row, m, n, k, &alpha, d_B, ldaB, d_A, ldaA,
                             &beta, d_C, ldc));
    return true;
}

static bool test_naive(cublasHandle_t handle, std::mt19937& rng) {
    const int M = 37, K = 23, N = 29;
    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(M) * N;

    std::vector<float> h_A(a_elems), h_B(b_elems), h_C(c_elems), h_ref(c_elems);
    fill_random(h_A, rng);
    fill_random(h_B, rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ref, 0, c_elems * sizeof(float)));

    dim3 block(blocksize, blocksize);
    const int tiles_m = (M + blocksize - 1) / blocksize;
    const int tiles_n = (N + blocksize - 1) / blocksize;
    dim3 grid(tiles_m * tiles_n);
    naiveMul<<<grid, block>>>(d_A, d_B, d_C, M, K, N, 1.0f, 0.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!cublas_row_major_gemm(handle, d_A, d_B, d_ref, M, K, N, 1.0f, 0.0f, CUBLAS_OP_N,
                               CUBLAS_OP_N))
        return false;
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return allclose(h_C, h_ref, 1e-3f, 1e-3f, "naiveMul");
}

static bool test_tiled(cublasHandle_t handle, std::mt19937& rng) {
    const int M = 63, K = 35, N = 41;
    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(M) * N;

    std::vector<float> h_A(a_elems), h_B(b_elems), h_C(c_elems), h_ref(c_elems);
    fill_random(h_A, rng);
    fill_random(h_B, rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ref, 0, c_elems * sizeof(float)));

    dim3 block(blocksize, blocksize);
    const int tiles_m = (M + blocksize - 1) / blocksize;
    const int tiles_n = (N + blocksize - 1) / blocksize;
    dim3 grid(tiles_m * tiles_n);
    tilingMul<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!cublas_row_major_gemm(handle, d_A, d_B, d_ref, M, K, N, 1.0f, 0.0f, CUBLAS_OP_N,
                               CUBLAS_OP_N))
        return false;
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return allclose(h_C, h_ref, 1e-3f, 1e-3f, "tilingMul");
}

static bool test_transpose_tiled(cublasHandle_t handle, std::mt19937& rng) {
    const int M = 31, K = 31, N = 19;
    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(M) * N;

    std::vector<float> h_A(a_elems), h_B(b_elems), h_C(c_elems), h_ref(c_elems);
    fill_random(h_A, rng);
    fill_random(h_B, rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ref, 0, c_elems * sizeof(float)));

    dim3 block(blocksize, blocksize);
    const int tiles_m = (M + blocksize - 1) / blocksize;
    const int tiles_n = (N + blocksize - 1) / blocksize;
    dim3 grid(tiles_m * tiles_n);
    const Transpose t{OP_T, OP_N};
    transposeTilingMul<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, t);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!cublas_row_major_gemm(handle, d_A, d_B, d_ref, M, K, N, 1.0f, 0.0f, CUBLAS_OP_T,
                               CUBLAS_OP_N))
        return false;
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return allclose(h_C, h_ref, 1e-3f, 1e-3f, "transposeTilingMul(transA=T,transB=N)");
}

static bool test_batch_naive(cublasHandle_t handle, std::mt19937& rng) {
    const int B = 4, M = 25, K = 19, N = 33;
    const size_t a_elems = static_cast<size_t>(B) * M * K;
    const size_t b_elems = static_cast<size_t>(B) * K * N;
    const size_t c_elems = static_cast<size_t>(B) * M * N;

    std::vector<float> h_A(a_elems), h_B(b_elems), h_C(c_elems), h_ref(c_elems);
    fill_random(h_A, rng);
    fill_random(h_B, rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ref, 0, c_elems * sizeof(float)));

    dim3 block(blocksize, blocksize);
    const int tiles_m = (M + blocksize - 1) / blocksize;
    const int tiles_n = (N + blocksize - 1) / blocksize;
    dim3 grid(B, tiles_m * tiles_n);
    batchNaiveMul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t a_stride = static_cast<size_t>(M) * K;
    const size_t b_stride = static_cast<size_t>(K) * N;
    const size_t c_stride = static_cast<size_t>(M) * N;
    for (int b = 0; b < B; ++b) {
        if (!cublas_row_major_gemm(handle, d_A + b * a_stride, d_B + b * b_stride,
                                   d_ref + b * c_stride, M, K, N, 1.0f, 0.0f, CUBLAS_OP_N,
                                   CUBLAS_OP_N))
            return false;
    }

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return allclose(h_C, h_ref, 1e-3f, 1e-3f, "batchNaiveMul");
}

static bool test_batch_tiled(cublasHandle_t handle, std::mt19937& rng) {
    const int B = 3, M = 47, K = 21, N = 35;
    const size_t a_elems = static_cast<size_t>(B) * M * K;
    const size_t b_elems = static_cast<size_t>(B) * K * N;
    const size_t c_elems = static_cast<size_t>(B) * M * N;

    std::vector<float> h_A(a_elems), h_B(b_elems), h_C(c_elems), h_ref(c_elems);
    fill_random(h_A, rng);
    fill_random(h_B, rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ref, 0, c_elems * sizeof(float)));

    dim3 block(blocksize, blocksize);
    const int tiles_m = (M + blocksize - 1) / blocksize;
    const int tiles_n = (N + blocksize - 1) / blocksize;
    dim3 grid(B, tiles_m * tiles_n);
    batchStridedMul<<<grid, block>>>(d_A, d_B, d_C, M, K, N, 1.0f, 0.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t a_stride = static_cast<size_t>(M) * K;
    const size_t b_stride = static_cast<size_t>(K) * N;
    const size_t c_stride = static_cast<size_t>(M) * N;
    for (int b = 0; b < B; ++b) {
        if (!cublas_row_major_gemm(handle, d_A + b * a_stride, d_B + b * b_stride,
                                   d_ref + b * c_stride, M, K, N, 1.0f, 0.0f, CUBLAS_OP_N,
                                   CUBLAS_OP_N))
            return false;
    }

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return allclose(h_C, h_ref, 1e-3f, 1e-3f, "batchStridedMul");
}

int main() {
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle\n";
        return 1;
    }

    std::mt19937 rng(12345);
    bool ok = true;
    ok = test_naive(handle, rng) && ok;
    ok = test_tiled(handle, rng) && ok;
    ok = test_transpose_tiled(handle, rng) && ok;
    ok = test_batch_naive(handle, rng) && ok;
    ok = test_batch_tiled(handle, rng) && ok;

    cublasDestroy(handle);
    if (!ok) {
        std::cerr << "Kernel correctness tests failed.\n";
        return 1;
    }
    std::cout << "All kernel correctness tests passed.\n";
    return 0;
}
