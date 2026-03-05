#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

struct FlashAttentionConfig {
    int B = 1;
    int H = 1;
    int S = 1;
    int D = 1;
    // Placeholder per-CTA on-chip SRAM budget for tile selection/tuning.
    int sram_budget_bytes = 128 * 1024;
    float scale = 1.0f;
    bool causal = true;
};

constexpr int kFlashAttnDefaultBlockR = 16;
constexpr int kFlashAttnDefaultBlockC = 16;
constexpr int kFlashAttnDefaultDTile = 64;

template <int BLOCK_R, int BLOCK_C, int D_TILE>
__global__ void flashAttnCausalForwardKernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V,
                                             float* __restrict__ O,
                                             int B, int H, int S, int D,
                                             float scale, bool causal);

void launchFlashAttnCausalForwardKernel(const float* Q, const float* K, const float* V,
                                        float* O, int B, int H, int S, int D,
                                        float scale, bool causal,
                                        cudaStream_t stream = 0);
