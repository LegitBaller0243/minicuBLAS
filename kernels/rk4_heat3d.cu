#include "kernels/rk4_heat3d.cuh"

__global__ void heat3dLaplacianKernel(const float* __restrict__ u,
                                      float* __restrict__ lap,
                                      int nx, int ny, int nz) {
    (void)u;
    (void)lap;
    (void)nx;
    (void)ny;
    (void)nz;
}

__global__ void rk4Heat3DStepKernel(const float* __restrict__ u,
                                    float* __restrict__ u_next,
                                    int nx, int ny, int nz,
                                    float alpha, float dt) {
    (void)u;
    (void)u_next;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)alpha;
    (void)dt;
}

