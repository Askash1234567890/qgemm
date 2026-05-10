"""Lazy loader for the compiled CUDA extension."""

from typing import Optional

_C: Optional[object] = None
_loaded: bool = False


def _load() -> None:
    global _C, _loaded
    if _loaded:
        return
    try:
        from qgemm import _C as ext  # type: ignore[attr-defined]
        _C = ext
    except ImportError:
        _C = None
    _loaded = True


def cuda_forward_available() -> bool:
    _load()
    return _C is not None


def cuda_forward(A, B, scale_a: float, scale_b: float, tile_size: int = 32):
    _load()
    if _C is None:
        raise RuntimeError("CUDA extension not built. Run `pip install -e .`.")
    return _C.forward(A, B, scale_a, scale_b, tile_size)


def cuda_backward(grad_output, A_dq, B_dq, mask_A, mask_B):
    _load()
    if _C is None:
        raise RuntimeError("CUDA extension not built. Run `pip install -e .`.")
    return _C.backward(grad_output, A_dq, B_dq, mask_A, mask_B)
