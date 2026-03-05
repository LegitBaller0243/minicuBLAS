#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

struct RK4Heat3DConfig {
    int nx = 1;
    int ny = 1;
    int nz = 1;
    float alpha = 1.0f;
    float dt = 1.0f;
    int steps = 1;
};

__global__ void heat3dLaplacianKernel(const float* __restrict__ u,
                                      float* __restrict__ lap,
                                      int nx, int ny, int nz);

__global__ void rk4Heat3DStepKernel(const float* __restrict__ u,
                                    float* __restrict__ u_next,
                                    int nx, int ny, int nz,
                                    float alpha, float dt);

