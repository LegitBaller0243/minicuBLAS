#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

struct FlashAttentionConfig {
    int B = 1;
    int H = 1;
    int S = 1;
    int D = 1;
    float scale = 1.0f;
    bool causal = true;
};

__global__ void flashAttnCausalForwardKernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V,
                                             float* __restrict__ O,
                                             int B, int H, int S, int D,
                                             float scale, bool causal);

