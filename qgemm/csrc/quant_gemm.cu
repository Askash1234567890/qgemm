/*
 * INT8 fake-quant forward kernels.
 *
 * Two tile sizes are provided:
 *   quantized_gemm_forward_16  — TILE=16, lower register pressure
 *   quantized_gemm_forward_32  — TILE=32, higher arithmetic intensity
 *
 * Both kernels:
 *   1. Load INT8 tiles from A (M×K) and B (K×N) into shared memory.
 *   2. Cast to float and accumulate in registers.
 *   3. Write FP32 result Y (M×N).
 *
 * float4 vectorised loads are used when the K dimension is 4-aligned.
 * Scalar fallback handles the remainder / non-aligned case.
 *
 * Assumptions: row-major (C contiguous), no transpose.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

// ── helpers ──────────────────────────────────────────────────────────────────

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ── 16×16 kernel ─────────────────────────────────────────────────────────────

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
        // ── load tile of A ──
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K)
            sA[threadIdx.y][threadIdx.x] = static_cast<float>(A[row * K + a_col]) * scale_a;
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // ── load tile of B ──
        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N)
            sB[threadIdx.y][threadIdx.x] = static_cast<float>(B[b_row * N + col]) * scale_b;
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        Y[row * N + col] = acc;
}

// ── float4 vectorised variant (32×32 only, K must be 4-aligned) ──────────────
//
// Each thread loads 4 int8 values at once from A (along K) and 4 from B,
// then scatters them into shared memory as floats.  The inner accumulation
// loop is identical to the scalar path.

__global__
__launch_bounds__(32 * 32, 2)
void quantized_gemm_forward_32_vec4(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    float*        __restrict__ Y,
    float scale_a,
    float scale_b,
    int M, int K, int N)
{
    constexpr int TILE = 32;

    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    // Each tile step advances by TILE along K.
    // float4 covers 4 consecutive int8 bytes (reinterpreted as one 32-bit load).
    static_assert(TILE % 4 == 0, "TILE must be divisible by 4 for vec4 path");

    for (int t = 0; t < CEIL_DIV(K, TILE); ++t) {

        // ── load A tile via float4 ──────────────────────────────────────────
        // threadIdx.x iterates over TILE columns of A in groups of 4.
        // We process (TILE/4) groups per row.
        {
            int base_col = t * TILE;                    // first K-col of this tile
            int tid = threadIdx.y * TILE + threadIdx.x;

            // Map flat tid → (shared_row, group_of_4)
            int sh_row = tid / (TILE / 4);             // which row in sA
            int group = tid % (TILE / 4);             // which group-of-4 in that row
            int a_col = base_col + group * 4;        // global K offset

            if (sh_row < M - blockIdx.y * TILE && a_col + 3 < K) {
                int global_row = blockIdx.y * TILE + sh_row;
                // 32-bit aligned load of 4 int8 bytes
                int packed = *reinterpret_cast<const int*>(A + global_row * K + a_col);
                int8_t b0 = (packed >>  0) & 0xFF;
                int8_t b1 = (packed >>  8) & 0xFF;
                int8_t b2 = (packed >> 16) & 0xFF;
                int8_t b3 = (packed >> 24) & 0xFF;
                sA[sh_row][group * 4 + 0] = static_cast<float>(b0) * scale_a;
                sA[sh_row][group * 4 + 1] = static_cast<float>(b1) * scale_a;
                sA[sh_row][group * 4 + 2] = static_cast<float>(b2) * scale_a;
                sA[sh_row][group * 4 + 3] = static_cast<float>(b3) * scale_a;
            } else {
                for (int i = 0; i < 4; ++i) {
                    int c = a_col + i;
                    int r = blockIdx.y * TILE + sh_row;
                    sA[sh_row][group * 4 + i] =
                        (sh_row < TILE && r < M && c < K)
                        ? static_cast<float>(A[r * K + c]) * scale_a
                        : 0.0f;
                }
            }
        }

        // ── load B tile via float4 ──────────────────────────────────────────
        {
            int base_row = t * TILE;
            int tid      = threadIdx.y * TILE + threadIdx.x;

            int sh_col  = tid / (TILE / 4);
            int group   = tid % (TILE / 4);
            int b_row   = base_row + group * 4;

            if (sh_col < N - blockIdx.x * TILE && b_row + 3 < K) {
                int global_col = blockIdx.x * TILE + sh_col;
                // B is [K, N] row-major; stride = N
                // 4 consecutive rows, same column
                sB[group * 4 + 0][sh_col] = static_cast<float>(B[(b_row + 0) * N + global_col]) * scale_b;
                sB[group * 4 + 1][sh_col] = static_cast<float>(B[(b_row + 1) * N + global_col]) * scale_b;
                sB[group * 4 + 2][sh_col] = static_cast<float>(B[(b_row + 2) * N + global_col]) * scale_b;
                sB[group * 4 + 3][sh_col] = static_cast<float>(B[(b_row + 3) * N + global_col]) * scale_b;
            } else {
                for (int i = 0; i < 4; ++i) {
                    int r = b_row + i;
                    int c = blockIdx.x * TILE + sh_col;
                    sB[group * 4 + i][sh_col] =
                        (r < K && c < N)
                        ? static_cast<float>(B[r * N + c]) * scale_b
                        : 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll 8
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

    const int M = A.size(0), K = A.size(1), N = B.size(1);

    auto Y = torch::zeros({M, N}, A.options().dtype(torch::kFloat32));

    const auto stream = at::cuda::getCurrentCUDAStream();

    if (tile_size == 16) {
        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(N, 16), CEIL_DIV(M, 16));
        quantized_gemm_forward_kernel<16><<<grid, block, 0, stream>>>(
            A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
            Y.data_ptr<float>(),
            scale_a, scale_b, M, K, N);
    } else {
        dim3 block(32, 32);
        dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        if (K % 4 == 0) {
            quantized_gemm_forward_32_vec4<<<grid, block, 0, stream>>>(
                A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
                Y.data_ptr<float>(),
                scale_a, scale_b, M, K, N);
        } else {
            quantized_gemm_forward_kernel<32><<<grid, block, 0, stream>>>(
                A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
                Y.data_ptr<float>(),
                scale_a, scale_b, M, K, N);
        }
    }
    return Y;
}
