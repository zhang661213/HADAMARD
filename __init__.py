"""
HADAMARD — KV Cache Quantization Library

Based on Google 2026 TurboQuant paper (ICLR 2026 style).
Target: Cosine Similarity > 0.995, Compression Ratio > 3x.
"""

from .lloyd_max import LloydMaxCodebook
from .turboquant import BitPacker, MSECompressor, TurboQuantKV
from .rotation import generate_rotation_matrix

__all__ = [
    "LloydMaxCodebook",
    "BitPacker",
    "MSECompressor",
    "TurboQuantKV",
    "generate_rotation_matrix",
]
