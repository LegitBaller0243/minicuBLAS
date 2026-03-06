#include "kernels/flash_attention.cuh"

#include <math.h>
#include <math_constants.h>

template <int BLOCK_R, int BLOCK_C, int D_TILE>
__global__ void flashAttnCausalForwardKernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V,
                                             float* __restrict__ O,
                                             int B, int H, int S, int D,
                                             float scale, bool causal) {
    // Launch contract:
    //   block = dim3(BLOCK_C, BLOCK_R)
    //   grid  = dim3(ceil_div(S, BLOCK_R), H, B)
    // Mapping: blockIdx.x -> query tile, blockIdx.y -> head, blockIdx.z -> batch.
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int q_tile = blockIdx.x;
    const int h_idx = blockIdx.y;
    const int b_idx = blockIdx.z;

    if (b_idx >= B || h_idx >= H || y >= BLOCK_R || x >= BLOCK_C) return;

    __shared__ float max_tile[BLOCK_R];
    __shared__ float row_sum_tile[BLOCK_R];
    __shared__ float max_old_tile[BLOCK_R];
    __shared__ float row_sum_old_tile[BLOCK_R];

    __shared__ float max_tilde_tile[BLOCK_R];
    __shared__ float row_sum_tilde_tile[BLOCK_R];
    __shared__ float score_tile[BLOCK_R][BLOCK_C];

    __shared__ float Q_block[BLOCK_R][D_TILE];
    __shared__ float K_block[BLOCK_C][D_TILE];
    __shared__ float V_block[BLOCK_C][D_TILE];

    const int q_row = q_tile * BLOCK_R + y;

    if (x == 0) {
        max_tile[y] = -CUDART_INF_F;
        row_sum_tile[y] = 0.0f;
    }
    __syncthreads();

    for (int tc = 0; tc < S; tc += BLOCK_C) {
        const int kv_row = tc + x;

        float score = 0.0f;
        for (int k0 = 0; k0 < D; k0 += D_TILE) {
            const int d_lim = min(D - k0, D_TILE);

            for (int d = x; d < d_lim; d += blockDim.x) {
                const int q_col = k0 + d;
                const int q_idx = ((b_idx * H + h_idx) * S + q_row) * D + q_col;
                Q_block[y][d] = (q_row < S) ? Q[q_idx] : 0.0f;
            }
            for (int d = y; d < d_lim; d += blockDim.y) {
                const int kv_col = k0 + d;
                const int kv_idx = ((b_idx * H + h_idx) * S + kv_row) * D + kv_col;
                K_block[x][d] = (kv_row < S) ? K[kv_idx] : 0.0f;
            }
            __syncthreads();

            for (int d = 0; d < d_lim; ++d) {
                score += Q_block[y][d] * K_block[x][d];
            }
            __syncthreads();
        }

        const bool in_bounds = (q_row < S && kv_row < S);
        const bool causal_ok = (!causal) || (q_row >= kv_row);
        score_tile[y][x] = (in_bounds && causal_ok) ? (score * scale) : -CUDART_INF_F;
        __syncthreads();

        if (x == 0) {
            const int valid_cols = min(BLOCK_C, S - tc);
            float m_tilde = -CUDART_INF_F;
            float l_tilde = 0.0f;

            for (int c = 0; c < valid_cols; ++c) {
                m_tilde = fmaxf(m_tilde, score_tile[y][c]);
            }
            for (int c = 0; c < valid_cols; ++c) {
                l_tilde += expf(score_tile[y][c] - m_tilde);
            }

            max_tilde_tile[y] = m_tilde;
            row_sum_tilde_tile[y] = l_tilde;

            const float m_old = max_tile[y];
            const float l_old = row_sum_tile[y];
            max_old_tile[y] = m_old;
            row_sum_old_tile[y] = l_old;

            const float m_new = fmaxf(m_old, m_tilde);
            const float l_new = expf(m_old - m_new) * l_old + expf(m_tilde - m_new) * l_tilde;
            max_tile[y] = m_new;
            row_sum_tile[y] = l_new;
        }
        __syncthreads();

        const int valid_cols = min(BLOCK_C, S - tc);
        for (int d0 = 0; d0 < D; d0 += D_TILE) {
            const int d_lim = min(D - d0, D_TILE);

            for (int d = y; d < d_lim; d += blockDim.y) {
                const int kv_col = d0 + d;
                const int kv_idx = ((b_idx * H + h_idx) * S + kv_row) * D + kv_col;
                V_block[x][d] = (kv_row < S) ? V[kv_idx] : 0.0f;
            }
            __syncthreads();

            for (int d_local = x; d_local < d_lim; d_local += blockDim.x) {
                if (q_row >= S) continue;

                float pv_tilde = 0.0f;
                for (int c = 0; c < valid_cols; ++c) {
                    const float p = expf(score_tile[y][c] - max_tilde_tile[y]);
                    pv_tilde += p * V_block[c][d_local];
                }

                const int d = d0 + d_local;
                const int o_idx = ((b_idx * H + h_idx) * S + q_row) * D + d;
                const float o_old = O[o_idx];

                const float m_old = max_old_tile[y];
                const float l_old = row_sum_old_tile[y];
                const float m_new = max_tile[y];
                const float l_new = row_sum_tile[y];
                const float m_tilde = max_tilde_tile[y];

                const float numer_old = expf(m_old - m_new) * l_old * o_old;
                const float numer_new = expf(m_tilde - m_new) * pv_tilde;
                O[o_idx] = (numer_old + numer_new) / l_new;
            }
            __syncthreads();
        }
    }
}

void launchFlashAttnCausalForwardKernel(const float* Q, const float* K, const float* V,
                                        float* O, int B, int H, int S, int D,
                                        float scale, bool causal,
                                        cudaStream_t stream) {
    dim3 block(kFlashAttnDefaultBlockC, kFlashAttnDefaultBlockR);
    dim3 grid((S + kFlashAttnDefaultBlockR - 1) / kFlashAttnDefaultBlockR, H, B);

    flashAttnCausalForwardKernel<kFlashAttnDefaultBlockR,
                                 kFlashAttnDefaultBlockC,
                                 kFlashAttnDefaultDTile>
        <<<grid, block, 0, stream>>>(Q, K, V, O, B, H, S, D, scale, causal);
}
