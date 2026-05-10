/*
 * STE backward kernels for fake-quant GEMM.
 *
 * Given:
 *   dY      — upstream gradient [M, N]  float32
 *   A_dq    — dequantized A     [M, K]  float32
 *   B_dq    — dequantized B     [K, N]  float32
 *   mask_A  — STE mask for A    [M, K]  bool  (x_q ∈ (-128, 127))
 *   mask_B  — STE mask for B    [K, N]  bool
 *
 * Computes:
 *   dA = (dY @ B_dq.T) * mask_A       [M, K]
 *   dB = (A_dq.T @ dY) * mask_B       [K, N]
 *
 * Strategy: reuse the same tiled matmul kernel for both products,
 * then apply the mask in a separate elementwise pass (one thread per element,
 * coalesced row-major access).
 *
 * Using TILE=32 for the gradient matmuls; mask application is lightweight.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ── tiled float32 matmul (for gradients) ─────────────────────────────────────
//
// C = A_fp @ B_fp   where both operands are already float32.

template <int TILE>
__global__
__launch_bounds__(TILE * TILE, 4)
void fp32_gemm_kernel(
    const float* __restrict__ A,   // [M, K]
    const float* __restrict__ B,   // [K, N]
    float*       __restrict__ C,   // [M, N]
    int M, int K, int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < CEIL_DIV(K, TILE); ++t) {
        int a_col = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;

        int b_row = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ── STE mask application ──────────────────────────────────────────────────────
//
// out[i] = in[i] * mask[i]   (mask is bool, cast to float on the fly)

__global__ void apply_ste_mask(
    float*       __restrict__ grad,   // [numel] in-place
    const bool*  __restrict__ mask,   // [numel]
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
        grad[idx] *= static_cast<float>(mask[idx]);
}

// ── dispatcher ───────────────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor> quantized_gemm_backward(
    torch::Tensor grad_output,   // [M, N]
    torch::Tensor A_dq,          // [M, K]
    torch::Tensor B_dq,          // [K, N]
    torch::Tensor mask_A,        // [M, K] bool
    torch::Tensor mask_B)        // [K, N] bool
{
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "grad_output must be float32");

    const int M = A_dq.size(0), K = A_dq.size(1), N = B_dq.size(1);
    const auto stream = at::cuda::getCurrentCUDAStream();

    // dA = dY @ B_dq.T   [M, N] x [N, K] → [M, K]
    // B_dq.T has shape [N, K], but we pass B_dq [K, N] and note the matmul
    // is  dY (M×N) × B_dq^T (N×K).
    // We'll compute this as: for each (m, k): sum_n dY[m,n] * B_dq[k,n]
    // Equivalently: grad_A = dY @ B_dq_T, where B_dq_T = B_dq.T (contiguous).
    auto B_dq_t = B_dq.t().contiguous();   // [N, K]
    auto A_dq_t = A_dq.t().contiguous();   // [K, M]

    auto grad_A = torch::empty({M, K}, grad_output.options());
    auto grad_B = torch::empty({K, N}, grad_output.options());

    constexpr int TILE = 32;

    // dA = dY [M,N] @ B_dq_t [N,K] → [M,K]
    {
        dim3 block(TILE, TILE);
        dim3 grid(CEIL_DIV(K, TILE), CEIL_DIV(M, TILE));
        fp32_gemm_kernel<TILE><<<grid, block, 0, stream>>>(
            grad_output.data_ptr<float>(),
            B_dq_t.data_ptr<float>(),
            grad_A.data_ptr<float>(),
            M, N, K);
    }

    // dB = A_dq_t [K,M] @ dY [M,N] → [K,N]
    {
        dim3 block(TILE, TILE);
        dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(K, TILE));
        fp32_gemm_kernel<TILE><<<grid, block, 0, stream>>>(
            A_dq_t.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_B.data_ptr<float>(),
            K, M, N);
    }

    // Apply STE masks in-place
    {
        constexpr int THREADS = 256;
        
        int numel_A = M * K;
        apply_ste_mask<<<CEIL_DIV(numel_A, THREADS), THREADS, 0, stream>>>(
            grad_A.data_ptr<float>(),
            mask_A.data_ptr<bool>(),
            numel_A);

        int numel_B = K * N;
        apply_ste_mask<<<CEIL_DIV(numel_B, THREADS), THREADS, 0, stream>>>(
            grad_B.data_ptr<float>(),
            mask_B.data_ptr<bool>(),
            numel_B);
    }
    return {grad_A, grad_B};
}
