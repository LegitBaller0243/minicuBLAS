#include "kernels/rk4_heat3d.cuh"

__device__ __constant__ float C0 = -201.0f / 72.0f;
__device__ __constant__ float C1 = 8.0f / 5.0f;
__device__ __constant__ float C2 = -1.0f / 5.0f;
__device__ __constant__ float C3 = 8.0f / 315.0f;
__device__ __constant__ float C4 = -1.0f / 560.0f;

__device__ __forceinline__ bool isInterior(int i, int j, int k,
                                            int nx, int ny, int nz) {
    return (i >= 4 && i < nx - 4 &&
            j >= 4 && j < ny - 4 &&
            k >= 4 && k < nz - 4);
}

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

__global__ void applyBoundaryKernel(float* __restrict__ field,
                                    int nx, int ny, int nz,
                                    float boundary_value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    if (!isInterior(i, j, k, nx, ny, nz)) {
        const int idx = i + j * nx + k * nx * ny;
        field[idx] = boundary_value;
    }
}

__global__ void buildStageInteriorKernel(const float* __restrict__ u,
                                         const float* __restrict__ k,
                                         float* __restrict__ u_stage,
                                         float cdt,
                                         int nx, int ny, int nz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_idx = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k_idx >= nz) {
        return;
    }
    if (!isInterior(i, j, k_idx, nx, ny, nz)) {
        return;
    }

    const int idx = i + j * nx + k_idx * nx * ny;
    u_stage[idx] = u[idx] + cdt * k[idx];
}

__global__ void laplacianInteriorKernel(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int nx, int ny, int nz,
                                        float alpha, float inv_h2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }
    if (!isInterior(i, j, k, nx, ny, nz)) {
        return;
    }

    const int idx = i + j * nx + k * nx * ny;
    out[idx] = alpha * laplacian25_uniform_h(in, i, j, k, nx, ny, nz, inv_h2);
}

__global__ void finalCombineInteriorKernel(const float* __restrict__ u,
                                           const float* __restrict__ k1,
                                           const float* __restrict__ k2,
                                           const float* __restrict__ k3,
                                           const float* __restrict__ k4,
                                           float* __restrict__ u_next,
                                           int nx, int ny, int nz,
                                           float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }
    if (!isInterior(i, j, k, nx, ny, nz)) {
        return;
    }

    const int idx = i + j * nx + k * nx * ny;
    u_next[idx] = u[idx] + (dt / 6.0f) * (k1[idx] + 2.0f * k2[idx] + 2.0f * k3[idx] + k4[idx]);
}

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
                               cudaStream_t stream) {
    const dim3 block(8, 8, 8);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (nz + block.z - 1) / block.z);

    laplacianInteriorKernel<<<grid, block, 0, stream>>>(u, k1, nx, ny, nz, alpha, inv_h2);

    buildStageInteriorKernel<<<grid, block, 0, stream>>>(u, k1, u_stage, 0.5f * dt, nx, ny, nz);
    applyBoundaryKernel<<<grid, block, 0, stream>>>(u_stage, nx, ny, nz, boundary_value);

    laplacianInteriorKernel<<<grid, block, 0, stream>>>(u_stage, k2, nx, ny, nz, alpha, inv_h2);

    buildStageInteriorKernel<<<grid, block, 0, stream>>>(u, k2, u_stage, 0.5f * dt, nx, ny, nz);
    applyBoundaryKernel<<<grid, block, 0, stream>>>(u_stage, nx, ny, nz, boundary_value);

    laplacianInteriorKernel<<<grid, block, 0, stream>>>(u_stage, k3, nx, ny, nz, alpha, inv_h2);

    buildStageInteriorKernel<<<grid, block, 0, stream>>>(u, k3, u_stage, dt, nx, ny, nz);
    applyBoundaryKernel<<<grid, block, 0, stream>>>(u_stage, nx, ny, nz, boundary_value);

    laplacianInteriorKernel<<<grid, block, 0, stream>>>(u_stage, k4, nx, ny, nz, alpha, inv_h2);

    finalCombineInteriorKernel<<<grid, block, 0, stream>>>(u, k1, k2, k3, k4, u_next, nx, ny, nz, dt);
    applyBoundaryKernel<<<grid, block, 0, stream>>>(u_next, nx, ny, nz, boundary_value);
}
