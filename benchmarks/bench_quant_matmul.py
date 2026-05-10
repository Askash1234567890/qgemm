"""
Benchmark: CUDA forward kernels vs PyTorch FP32 baseline.

Usage:
    python benchmarks/bench_quant_matmul.py

Requires: CUDA GPU, compiled extension (`pip install -e .`),
          torch.utils.benchmark (bundled with PyTorch >= 1.8).

Output: wall-time table + ncu profile command.
"""

import torch
import torch.utils.benchmark as benchmark

from qgemm import ref_quant_matmul, QuantMatmul
from qgemm._ext import cuda_forward_available


SHAPES = [
    (512,  256,  512),
    (1024, 512, 1024),
    (2048, 1024, 2048),
    (4096, 2048, 4096),
]

TILE_SIZES = [16, 32]

WARMUP   = 5
REPEATS  = 50


def make_inputs(M, K, N, device="cuda"):
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    return A, B


def bench(label: str, fn, min_run_time: float = 0.5):
    t = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
        label=label,
        num_threads=1,
    )
    return t.blocked_autorange(min_run_time=min_run_time)


def main():
    if not torch.cuda.is_available():
        print("No CUDA device found — skipping benchmark.")
        return

    if not cuda_forward_available():
        print("Extension not built. Run `pip install -e .` first.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    results = []

    for M, K, N in SHAPES:
        A, B = make_inputs(M, K, N)
        label = f"M={M} K={K} N={N}"

        # baseline: torch FP32 matmul (cuBLAS)
        r_base = bench(
            f"[baseline FP32] {label}",
            lambda: torch.mm(A, B),
        )

        # reference fake-quant (torch ops, no CUDA kernel)
        r_ref = bench(
            f"[ref PyTorch]   {label}",
            lambda: ref_quant_matmul(A, B),
        )

        for tile in TILE_SIZES:
            r_cuda = bench(
                f"[CUDA tile={tile}]  {label}",
                lambda t=tile: QuantMatmul.apply(A, B, None, None, t),
            )
            results.append((label, tile, r_base, r_ref, r_cuda))

    # ── print table ──────────────────────────────────────────────────────────
    print(f"{'Shape':<22} {'Tile':>6} {'FP32 base':>12} {'PyTorch ref':>13} {'CUDA kernel':>13} {'vs base':>9}")
    print("-" * 82)

    seen = set()
    for label, tile, r_base, r_ref, r_cuda in results:
        base_ms  = r_base.mean  * 1e3
        ref_ms   = r_ref.mean   * 1e3
        cuda_ms  = r_cuda.mean  * 1e3
        speedup  = base_ms / cuda_ms

        if label not in seen:
            seen.add(label)

        print(
            f"{label:<22} {tile:>6}   "
            f"{base_ms:>10.3f}ms   {ref_ms:>10.3f}ms   "
            f"{cuda_ms:>10.3f}ms   {speedup:>7.2f}x"
        )

    print()
    print("ncu profile command:")
    print(
        "  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,"
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum "
        "-o profile.ncu-rep python benchmarks/bench_quant_matmul.py"
    )


if __name__ == "__main__":
    main()
