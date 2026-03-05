#include "kernels/flash_attention.cuh"

__global__ void flashAttnCausalForwardKernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V,
                                             float* __restrict__ O,
                                             int B, int H, int S, int D,
                                             float scale, bool causal) {
    (void)Q;
    (void)K;
    (void)V;
    (void)O;
    (void)B;
    (void)H;
    (void)S;
    (void)D;
    (void)scale;
    (void)causal;
}

