"""
third_octave_codec/bands_utils.py
Utility functions and classes for third-octave band computation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional
import torch
import numpy as np
from tqdm.auto import tqdm


@dataclass
class ThirdOctaveBand:
    fc: float
    f_lo: float
    f_hi: float


def _generate_third_octave_bands(fmin: float, fmax: float, f_ref: float = 1000.0) -> List[ThirdOctaveBand]:
    """Generate IEC nominal 1/3-octave bands covering [fmin, fmax].
    Clamps upper and lower frequency limits to avoid index overflow in FFT bins.
    """
    bands: List[ThirdOctaveBand] = []
    half = 2 ** (1.0 / 6.0)
    k_min = math.ceil(3 * math.log2(max(fmin, 1e-6) / f_ref)) if fmin > 0 else -120
    k_max = math.floor(3 * math.log2(fmax / f_ref))

    for k in range(k_min, k_max + 1):
        fc = f_ref * (2.0 ** (k / 3.0))
        f_lo = fc / half
        f_hi = fc * half

        # Clamp inside valid range
        f_lo = max(f_lo, fmin)
        f_hi = min(f_hi, fmax * 0.9999)  # ensure no bin exceeds Nyquist

        if f_hi <= f_lo:
            continue

        bands.append(ThirdOctaveBand(fc=fc, f_lo=f_lo, f_hi=f_hi))

    return bands


@dataclass
class ThirdOctaveMapper:
    sr: int
    n_fft: int
    fmin: float
    fmax: float
    bands: List[ThirdOctaveBand]
    mapping: List[torch.Tensor]
    bin_freqs: torch.Tensor

    @staticmethod
    def build(sr: int, n_fft: int, fmin: float = 20.0, fmax: Optional[float] = None) -> "ThirdOctaveMapper":
        if fmax is None:
            fmax = sr / 2.0
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).cpu()  # (F,)
        bands = _generate_third_octave_bands(fmin, fmax)
        mapping: List[torch.Tensor] = []
        F = freqs.numel()

        desc = f"🔧 Creating fast third-octave bands ({len(bands)} total)"
        for b in tqdm(bands, desc=desc, unit="band"):
            raw = torch.nonzero((freqs >= b.f_lo) & (freqs < b.f_hi), as_tuple=False).squeeze(-1)
            if raw.numel() == 0:
                mapping.append(torch.empty(0, dtype=torch.long))
                continue
            raw = torch.unique(raw.to(torch.long), sorted=True)
            raw = raw[(raw >= 0) & (raw < F)]
            mapping.append(raw.cpu())

        print(f"✅ Completed creation of {len(mapping)} third-octave bands.")
        return ThirdOctaveMapper(sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax,
                                 bands=bands, mapping=mapping, bin_freqs=freqs)

    @property
    def num_bands(self) -> int:
        return len(self.bands)

    @property
    def centers_hz(self) -> np.ndarray:
        return np.array([b.fc for b in self.bands], dtype=np.float64)

    def aggregate_power(self, mag: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Aggregate rFFT magnitudes into third-octave band powers.
        Works for input shapes (T, F) or (F, T); removes singleton batch if present.
        Returns (T, Bands).
        """
        device = mag.device

        # Remove singleton batch if present
        if mag.ndim == 3 and mag.shape[0] == 1:
            mag = mag.squeeze(0)

        # Expect 2D; orient as (T, F)
        if mag.ndim != 2:
            raise ValueError(f"Expected 2D tensor after squeeze, got {mag.shape}")
        # If first dim looks like F, transpose
        if mag.shape[0] < mag.shape[1]:
            mag = mag.transpose(0, 1)

        T, F = mag.shape
        mag2 = mag ** 2
        out = mag.new_zeros((T, self.num_bands), device=device)

        for b, idx in enumerate(self.mapping):
            if idx.numel() == 0:
                continue
            idx = idx.to(device)
            idx = idx[(idx >= 0) & (idx < F)]
            if idx.numel() == 0:
                continue
            out[:, b] = mag2[:, idx].sum(dim=1)

        return out.clamp_min_(eps)
