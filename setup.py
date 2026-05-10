"""
CUDA extension build logic.

Metadata (name, version, deps, etc.) lives in pyproject.toml.
This file exists solely because torch.utils.cpp_extension.BuildExtension
is a setuptools command — it cannot be expressed in TOML.

Build:
    pip install -e .
    pip install -e . --no-build-isolation   # if torch is already installed
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_SOURCES = [
    "qgemm/csrc/bindings.cpp",
    "qgemm/csrc/quant_gemm.cu",
    "qgemm/csrc/quant_gemm_backward.cu",
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
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
