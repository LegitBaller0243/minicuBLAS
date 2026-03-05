#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

struct RK4Heat3DConfig {
    int nx = 1;
    int ny = 1;
    int nz = 1;
    float alpha = 1.0f;
    float dt = 1.0f;
    float h = 1.0f;
    float boundary_value = 0.0f;
    int steps = 1;
};

__global__ void rk4Heat3DStepKernel(
    const float* __restrict__ u,
    float* __restrict__ u_next,
    float* __restrict__ k1,
    float* __restrict__ k2,
    float* __restrict__ k3,
    float* __restrict__ k4,
    float* __restrict__ u_stage1,
    float* __restrict__ u_stage2,
    int nx, int ny, int nz,
    float alpha, float dt, float inv_h2, float boundary_value);
