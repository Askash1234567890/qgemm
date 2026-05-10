# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] - 2026-04-10

### Added
- **Phase 1** — CPU/PyTorch fake-quant reference
  - `fake_quantize(x, scale)`: per-tensor INT8 quantization + STE mask
  - `ref_quant_matmul(A, B)`: CPU/GPU reference matmul via fake-quant
  - `QuantMatmul` as `torch.autograd.Function` with PyTorch fallback
  - `setup.py` for building CUDA `.so` via `torch.utils.cpp_extension`
  - `VERSION`, `CHANGELOG.md` for downstream project tracking

- **Phase 2** — CUDA forward kernels (`quant_gemm.cu`)
  - 16×16 tiled kernel: lower register pressure, broader occupancy
  - 32×32 tiled kernel + `float4` vectorised loads (16-byte aligned)
  - Scalar fallback for K % 4 ≠ 0 boundary case
  - `__launch_bounds__(256, 4)` and `#pragma unroll` on inner loop
  - `QuantMatmul.forward` auto-dispatches to CUDA kernel when extension is built

- **Phase 3** — CUDA backward / STE (`quant_gemm_backward.cu`)
  - `fp32_gemm_kernel<32>` for gradient matmuls (dA, dB)
  - `apply_ste_mask` elementwise kernel: zeroes gradients at clipped values
  - `QuantMatmul.backward` dispatches to CUDA when extension available
  - `gradcheck` passes with `eps=1e-4`, `atol=1e-3`

- **Phase 4** — Benchmarks
  - `benchmarks/bench_quant_matmul.py`: wall-time table (16×16 vs 32×32 vs cuBLAS baseline)
  - `ncu` profile command printed automatically
