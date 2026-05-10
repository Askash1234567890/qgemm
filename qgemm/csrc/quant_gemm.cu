/*
 * INT8 fake-quant forward kernels.
 *
 * Two tile sizes via a single template:
 *   quantized_gemm_forward_kernel<16>  — lower register pressure
 *   quantized_gemm_forward_kernel<32>  — higher arithmetic intensity
 *
 * Each kernel:
 *   1. Loads INT8 tiles from A [M×K] and B [K×N] into __shared__ float.
 *   2. Each thread dequantizes on load: val = (float)int8 * scale.
 *   3. Accumulates dot-product in a register, writes FP32 to Y [M×N].
 *
 * One thread per output element; tiles stepped along K.
 * #pragma unroll on the inner accumulation loop.
 * __launch_bounds__ controls occupancy.
 *
 * Note: float4 / vectorised INT8 loads require careful re-mapping of the
 * thread→shared-memory layout (tid/sh_row arithmetic differs from the
 * scalar case). That optimisation is deferred to Phase 4 to avoid silent
 * out-of-bounds writes in shared memory.
 *
 * Assumptions: row-major (C-contiguous), no transpose.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ── tiled INT8→FP32 matmul ───────────────────────────────────────────────────

template <int TILE>
__global__
__launch_bounds__(TILE * TILE, 4)
void quantized_gemm_forward_kernel(
    const int8_t* __restrict__ A,   // [M, K]
    const int8_t* __restrict__ B,   // [K, N]
    float*        __restrict__ Y,   // [M, N]
    float scale_a,
    float scale_b,
    int M, int K, int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < CEIL_DIV(K, TILE); ++t) {

        // Load tile of A: thread (ty, tx) loads A[row, t*TILE+tx]
        const int a_col = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? static_cast<float>(A[row * K + a_col]) * scale_a
            : 0.0f;

        // Load tile of B: thread (ty, tx) loads B[t*TILE+ty, col]
        const int b_row = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? static_cast<float>(B[b_row * N + col]) * scale_b
            : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        Y[row * N + col] = acc;
}

// ── dispatcher ───────────────────────────────────────────────────────────────

torch::Tensor quantized_gemm_forward(
    torch::Tensor A,      // [M, K] int8, CUDA
    torch::Tensor B,      // [K, N] int8, CUDA
    float scale_a,
    float scale_b,
    int   tile_size)      // 16 or 32
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kInt8 && B.dtype() == torch::kInt8,
                "inputs must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(0), "inner dimensions must match");
    TORCH_CHECK(tile_size == 16 || tile_size == 32, "tile_size must be 16 or 32");

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    auto Y = torch::zeros({M, N}, A.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (tile_size == 16) {
        constexpr int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
        quantized_gemm_forward_kernel<TILE><<<grid, block, 0, stream>>>(
            A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
            Y.data_ptr<float>(),
            scale_a, scale_b, M, K, N);
    } else {
        constexpr int TILE = 32;
        dim3 block(TILE, TILE);
        dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
        quantized_gemm_forward_kernel<TILE><<<grid, block, 0, stream>>>(
            A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
            Y.data_ptr<float>(),
            scale_a, scale_b, M, K, N);
    }

    return Y;
}
