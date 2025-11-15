"""
Utility metrics for codebook analysis and nominal bitrate calculations.
"""

from __future__ import annotations
import numpy as np


def entropy_bits(p: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon entropy H(p) in bits for a discrete distribution."""
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    mask = p > 0
    return float(-(p[mask] * np.log2(p[mask])).sum())


def codebook_usage(hist: np.ndarray) -> tuple[int, float]:
    """Return (num_used, ratio) from integer histogram per codebook."""
    hist = np.asarray(hist)
    used = int((hist > 0).sum())
    ratio = used / float(hist.size) if hist.size else 0.0
    return used, ratio


def bitrate_estimate(frames_per_second: float, groups: int, codebook_size: int) -> float:
    """Nominal bitrate in bits/sec, ignoring entropy coding gains."""
    if groups <= 0 or codebook_size <= 1 or frames_per_second <= 0:
        return 0.0
    return frames_per_second * groups * np.log2(codebook_size)
