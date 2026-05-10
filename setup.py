"""
CUDA extension build logic.

Metadata (name, version, deps, etc.) lives in pyproject.toml.
This file exists solely because torch.utils.cpp_extension.BuildExtension
is a setuptools command — it cannot be expressed in TOML.

Build:
    pip install -e . --no-build-isolation

Architecture selection (in order of priority):
    1. TORCH_CUDA_ARCH_LIST env var — set manually, torch handles gencode
    2. Auto-detect from connected GPU at build time
    3. Fallback: compile for all known Ampere–Blackwell archs
"""

import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_SOURCES = [
    "qgemm/csrc/bindings.cpp",
    "qgemm/csrc/quant_gemm.cu",
    "qgemm/csrc/quant_gemm_backward.cu",
]


def _detect_arch_flags() -> list[str]:
    """Return -gencode flags for the current GPU, or a broad fallback list."""
    # If user set TORCH_CUDA_ARCH_LIST, torch handles it — don't add gencode.
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []

    # Try to detect the connected GPU's compute capability via nvidia-smi.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
        flags = []
        for cap in set(out):
            major, minor = cap.strip().split(".")
            cc = f"{major}{minor}"
            flags.append(f"-gencode=arch=compute_{cc},code=sm_{cc}")
        if flags:
            return flags
    except Exception:
        pass

    # Fallback: cover Ampere → Blackwell
    return [
        "-gencode=arch=compute_80,code=sm_80",   # A100
        "-gencode=arch=compute_86,code=sm_86",   # RTX 30xx
        "-gencode=arch=compute_89,code=sm_89",   # RTX 40xx / Ada
        "-gencode=arch=compute_90,code=sm_90",   # H100 Hopper
        "-gencode=arch=compute_100,code=sm_100", # B100 Blackwell
        "-gencode=arch=compute_101,code=sm_101", # B200 Blackwell
    ]


setup(
    ext_modules=[
        CUDAExtension(
            name="qgemm._C",
            sources=CUDA_SOURCES,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "--use_fast_math",
                    *_detect_arch_flags(),
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
