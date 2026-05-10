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

### Speedup table (TBD — fill after running on target GPU)

| Shape (M=N=K) | FP32 cuBLAS | PyTorch ref | tile=16 | tile=32 | vs cuBLAS |
|---------------|-------------|-------------|---------|---------|-----------|
| 512 | — ms | — ms | — ms | — ms | — |
| 1024 | — ms | — ms | — ms | — ms | — |
| 2048 | — ms | — ms | — ms | — ms | — |
| 4096 | — ms | — ms | — ms | — ms | — |

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

## Version

See [VERSION](VERSION). Follows [Semantic Versioning](https://semver.org/).  
See [CHANGELOG](CHANGELOG.md) for release history.
