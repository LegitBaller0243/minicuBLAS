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

__global__ void applyBoundaryKernel(float* __restrict__ field,
                                    int nx, int ny, int nz,
                                    float boundary_value);

__global__ void buildStageInteriorKernel(const float* __restrict__ u,
                                         const float* __restrict__ k,
                                         float* __restrict__ u_stage,
                                         float cdt,
                                         int nx, int ny, int nz);

__global__ void laplacianInteriorKernel(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int nx, int ny, int nz,
                                        float alpha, float inv_h2);

__global__ void finalCombineInteriorKernel(const float* __restrict__ u,
                                           const float* __restrict__ k1,
                                           const float* __restrict__ k2,
                                           const float* __restrict__ k3,
                                           const float* __restrict__ k4,
                                           float* __restrict__ u_next,
                                           int nx, int ny, int nz,
                                           float dt);

void launchRK4Heat3DStepStaged(const float* u,
                               float* u_next,
                               float* k1,
                               float* k2,
                               float* k3,
                               float* k4,
                               float* u_stage,
                               int nx, int ny, int nz,
                               float alpha, float dt, float inv_h2,
                               float boundary_value,
                               cudaStream_t stream = 0);
