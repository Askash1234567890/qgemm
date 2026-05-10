# qgemm — INT8 Fake-Quant GEMM with STE Backward

CUDA kernels for quantization-aware training (QAT) matmul.  
Weights and activations are stored as **INT8**; forward and backward operate in **FP32** via the Straight-Through Estimator (STE).

---

## What this is

```
Y = dequant(A) @ dequant(B)

dequant:  x_q  = clamp(round(x / scale), -128, 127)
          x_dq = x_q.float() * scale

STE backward:
          dA = (dY @ B_dq.T) * mask_A       mask = (x_q ∈ (-128, 127))
          dB = (A_dq.T @ dY) * mask_B
```

Scale is **per-tensor** (one float per matrix).

---

## Project status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | CPU/PyTorch reference + `setup.py` | ✅ done |
| 2 | CUDA forward kernel (16×16 & 32×32 tiling, `float4`) | ✅ done |
| 3 | CUDA backward (STE masks) | ✅ done |
| 4 | Benchmark script (`bench_quant_matmul.py`) | ✅ done |
| 5 | CI, PyPI packaging | 🔜 |

---

## Installation

**Requirements:** PyTorch ≥ 2.2, C++20 compiler, Ampere+ GPU (sm_80+).

`nvcc` and PyTorch **must use the same CUDA version**. Verify before building:

```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

If they differ, reinstall PyTorch to match your system CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # CUDA 13.0
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
```

Or point the build to the CUDA version PyTorch expects:

```bash
export CUDA_HOME=/usr/local/cuda-12.8   # adjust to match torch.version.cuda
```

Then build:

```bash
git clone https://github.com/Askash1234567890/qgemm
cd qgemm
pip install ninja                            # optional but speeds up compilation
uv pip install -e . --no-build-isolation     # use torch already in your env
# or: pip install -e . --no-build-isolation
```

This compiles `qgemm/_C.so` with `nvcc`.  
Without a GPU the package installs in pure-Python mode (PyTorch fallback).

`setup.py` auto-detects the GPU architecture via `nvidia-smi` and compiles only
for the connected device. If detection fails, it falls back to a broad list
covering Ampere through Blackwell (sm_80 … sm_101). You can override at any time:

```bash
TORCH_CUDA_ARCH_LIST="native" uv pip install -e . --no-build-isolation
```

---

## Quick start

```python
import torch
from qgemm import QuantMatmul, ref_quant_matmul

A = torch.randn(128, 64, device="cuda")
B = torch.randn(64, 256, device="cuda")

# Reference (pure PyTorch, any device)
Y_ref = ref_quant_matmul(A, B)

# Autograd function — dispatches to CUDA kernel when available
A = A.requires_grad_(True)
B = B.requires_grad_(True)
Y = QuantMatmul.apply(A, B)          # tile_size=32 by default
Y.sum().backward()                    # STE gradients

# Explicit tile size
Y16 = QuantMatmul.apply(A, B, None, None, 16)
```

---

## Testing

```bash
pytest tests/ -v
```

| Mark | Requires |
|------|----------|
| (no mark) | CPU only |
| `skipif no CUDA` | CUDA GPU |
| `skipif extension not built` | CUDA GPU + `pip install -e .` |

> **Note on STE and `gradcheck`:** `torch.autograd.gradcheck` is incompatible
> with STE by design. `round()` is piecewise-constant, so the numerical Jacobian
> (finite difference with `eps << quant_step`) is always 0 while the analytical
> STE gradient is non-zero. Tests verify STE correctness manually instead:
> gradients are non-zero for interior elements and zero for clipped ones.

---

## Benchmarks

```bash
python benchmarks/bench_quant_matmul.py
```

### Speedup table

GPU: **NVIDIA GeForce RTX 5090**, PyTorch 2.11.0+cu130, CUDA 13.0.

| Shape (M, K, N) | FP32 cuBLAS | PyTorch ref | tile=16 | tile=32 | vs cuBLAS |
|-----------------|-------------|-------------|---------|---------|-----------|
| 512 × 256 × 512 | 0.015 ms | 0.098 ms | 0.124 ms | 0.124 ms | 0.12× |
| 1024 × 512 × 1024 | 0.027 ms | 0.107 ms | 0.236 ms | 0.250 ms | 0.11× |
| 2048 × 1024 × 2048 | 0.148 ms | 0.256 ms | 1.008 ms | 1.071 ms | 0.15× |
| 4096 × 2048 × 4096 | 1.099 ms | 1.331 ms | 7.229 ms | 7.961 ms | 0.15× |

The kernel is **6–9× slower** than cuBLAS at this stage — this is expected and explainable:

- **cuBLAS** employs tensor cores (WMMA), double-buffered global→shared prefetching,
  register-level blocking, and warp-level primitives — the result of years of
  architecture-specific tuning.
- **Our kernel** is a clean scalar shared-memory tiled GEMM: one float per thread,
  no `__dp4a`, no computation/memory overlap, no register blocking.
- **PyTorch ref** (fake-quant via `torch.mm`) routes through cuBLAS for the matmul
  itself but adds quantization overhead, making it slower than the raw cuBLAS baseline
  yet still faster than our kernel.

Phase 4 will close this gap with vectorised `int4` loads, `__dp4a` INT8 dot-product
intrinsics, and double-buffered prefetching — expected speedup: **3–5× over the
current kernel**.

*Run `ncu` for roofline analysis:*

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
-o profile.ncu-rep python benchmarks/bench_quant_matmul.py
ncu-ui profile.ncu-rep
```

---

## Kernel design

| Feature | Detail |
|---------|--------|
| Storage | `int8_t` in global memory |
| Compute | `float32` in shared memory and registers |
| Tile sizes | 16×16 (high occupancy) and 32×32 (high arithmetic intensity) |
| Vectorised loads | `reinterpret_cast<const int*>` → 4 × `int8_t` in one 32-bit load |
| Alignment | Scalar fallback when `K % 4 ≠ 0` |
| Launch bounds | `__launch_bounds__(256, 4)` on 16² kernel, `__launch_bounds__(1024, 2)` on 32² |
| Inner loop | `#pragma unroll` over tile dimension |
| Streams | Kernel launched on `at::cuda::getCurrentCUDAStream()` |
| Sync | No `cudaDeviceSynchronize` in hot path |

---

## Roadmap

### Phase 4 — Performance (next)

| Task | Technique | Expected gain |
|------|-----------|---------------|
| Vectorised INT8 loads | `__dp4a` / `reinterpret_cast<int4*>` with correct `tid→shmem` mapping | 2–3× memory BW |
| Double-buffered prefetch | `cp.async` + ping-pong shared memory buffers | hide global load latency |
| Register blocking | each thread accumulates 2×2 or 4×4 output sub-tile | 2–4× compute efficiency |
| Warp-level reduction | `__shfl_down_sync` for final accumulation | reduce `__syncthreads` overhead |
| Per-channel scale | vector scale along rows/cols instead of scalar | production QAT accuracy |
| `cudaStream_t` pool | persistent stream pool for multi-kernel graphs | lower launch overhead |

### Phase 5 — Packaging & CI

- [ ] GitHub Actions CI (build + CPU tests on `ubuntu-latest`)
- [ ] PyPI-compatible `sdist` / `wheel` with optional CUDA extension
- [ ] `pip install qgemm` with pure-Python fallback for non-CUDA environments
- [ ] `torch.compile` integration via `torch.library` custom op registration

### Further ideas

- **FP8 support** — E4M3 / E5M2 formats available on Hopper+ (sm_90);
  requires custom pack/unpack, but halves memory footprint vs INT8
- **CUTLASS backend** — swap scalar kernel for CUTLASS `GemmUniversal` template
  to get tensor-core utilisation with minimal code
- **Structured sparsity** — combine INT8 quantization with 2:4 sparsity
  (NVIDIA Ampere sparse tensor cores); potential 2× throughput on top of INT8
- **Dynamic quantization** — compute scale on-the-fly from running statistics
  inside the kernel (fused quantize + matmul)
- **Multi-GPU / NCCL** — sharded weight matmul with all-reduce for tensor parallelism

---

## Version

See [VERSION](VERSION). Follows [Semantic Versioning](https://semver.org/).  
See [CHANGELOG](CHANGELOG.md) for release history.
