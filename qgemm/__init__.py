try:
    from importlib.metadata import version
    __version__ = version("qgemm")
except Exception:
    from pathlib import Path
    __version__ = (Path(__file__).parent.parent / "VERSION").read_text().strip()

from .ref_quant_matmul import fake_quantize, ref_quant_matmul, QuantMatmul

__all__ = ["fake_quantize", "ref_quant_matmul", "QuantMatmul", "__version__"]
