#include "kernels/rk4_heat3d.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __constant__ float C0 = -201.0f / 72.0f;
__device__ __constant__ float C1 = 8.0f / 5.0f;
__device__ __constant__ float C2 = -1.0f / 5.0f;
__device__ __constant__ float C3 = 8.0f / 315.0f;
__device__ __constant__ float C4 = -1.0f / 560.0f;

__device__ __forceinline__ float laplacian25_uniform_h(
    const float* __restrict__ u,
    int i, int j, int k,
    int nx, int ny, int nz,
    float inv_h2) {

    (void)nz; // only needed for bounds in caller
    const int sx = 1;
    const int sy = nx;
    const int sz = nx * ny;
    const int idx = i + j * nx + k * nx * ny;

    float sum = C0 * u[idx];

#pragma unroll
    for (int d = 1; d <= 4; ++d) {
        const float c = (d == 1 ? C1 : d == 2 ? C2 : d == 3 ? C3 : C4);
        sum += c * (u[idx + d * sx] + u[idx - d * sx]);
        sum += c * (u[idx + d * sy] + u[idx - d * sy]);
        sum += c * (u[idx + d * sz] + u[idx - d * sz]);
    }

    return sum * inv_h2;
}

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
    float alpha, float dt, float inv_h2, float boundary_value) {

    cg::grid_group grid = cg::this_grid();

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    const bool in_bounds = (i < nx && j < ny && k < nz);
    const int idx = in_bounds ? (i + j * nx + k * nx * ny) : 0;

    const bool interior = in_bounds &&
                          (i >= 4 && i < nx - 4 &&
                           j >= 4 && j < ny - 4 &&
                           k >= 4 && k < nz - 4);

    if (in_bounds) {
        k1[idx] = interior ? alpha * laplacian25_uniform_h(u, i, j, k, nx, ny, nz, inv_h2) : 0.0f;
    }
    grid.sync();

    if (in_bounds) {
        u_stage1[idx] = interior ? (u[idx] + 0.5f * dt * k1[idx]) : boundary_value;
    }
    grid.sync();

    if (in_bounds) {
        k2[idx] = interior ? alpha * laplacian25_uniform_h(u_stage1, i, j, k, nx, ny, nz, inv_h2) : 0.0f;
    }
    grid.sync();

    if (in_bounds) {
        u_stage2[idx] = interior ? (u[idx] + 0.5f * dt * k2[idx]) : boundary_value;
    }
    grid.sync();

    if (in_bounds) {
        k3[idx] = interior ? alpha * laplacian25_uniform_h(u_stage2, i, j, k, nx, ny, nz, inv_h2) : 0.0f;
    }
    grid.sync();

    if (in_bounds) {
        u_stage1[idx] = interior ? (u[idx] + dt * k3[idx]) : boundary_value;
    }
    grid.sync();

    if (in_bounds) {
        k4[idx] = interior ? alpha * laplacian25_uniform_h(u_stage1, i, j, k, nx, ny, nz, inv_h2) : 0.0f;
    }
    grid.sync();

    if (in_bounds) {
        if (interior) {
            u_next[idx] = u[idx] + (dt / 6.0f) * (k1[idx] + 2.0f * k2[idx] + 2.0f * k3[idx] + k4[idx]);
        } else {
            u_next[idx] = boundary_value;
        }
    }
}
