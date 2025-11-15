"""
third_octave_codec/features.py
Feature extraction for third-octave codec.
Includes STFT → third-octave band mapping → normalization.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .bands_utils import ThirdOctaveMapper


class FeatureExtractor(nn.Module):
    """
    Converts raw waveform → third-octave magnitude bands.
    Supports both single and batched inputs on CPU or GPU.
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop: int,
        fmin: float,
        fmax: float | None,
        power: float = 1.0,
        use_log: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.power = power
        self.use_log = use_log
        self.device = torch.device(device)

        # Build third-octave mapper
        self.mapper = ThirdOctaveMapper.build(sr, n_fft, fmin, fmax)

        # Move everything to device
        self.to(self.device)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Compute third-octave magnitude features.
        Args:
            wav: Tensor of shape (B, N) or (N,)
        Returns:
            bands: (B, T, Bands)
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # → (1, N)

        wav = wav.to(self.device)
        window = torch.hann_window(self.n_fft, device=self.device)

        # STFT → (B, F, T)
        S = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )

        # Magnitude → (B, T, F)
        mag = S.abs().permute(0, 2, 1)

        # Power scaling
        if self.power != 1.0:
            mag = mag.clamp_min(1e-12) ** (self.power / 2.0)

        # Map each example in batch to third-octave bands
        bands_list = []
        for m in mag:  # each m: (T, F)
            bands_list.append(self.mapper.aggregate_power(m))

        bands = torch.stack(bands_list, dim=0)  # (B, T, Bands)

        # Optional log compression
        if self.use_log:
            bands = torch.log1p(bands)

        return bands


class FeatureNormalizer(nn.Module):
    """
    Normalizes third-octave features (not waveforms).
    Input:  (B, T, Bands) or (T, Bands)
    Output: normalized tensor of same shape
    - Per-sample normalization: NON mantiene stato (niente fit/salvataggio).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            mean = x.mean(dim=0, keepdim=True)
            std  = x.std(dim=0, keepdim=True)
        elif x.ndim == 3:
            mean = x.mean(dim=1, keepdim=True)
            std  = x.std(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unexpected input shape for normalization: {x.shape}")
        return (x - mean) / (std + self.eps)

    # Alias per retro-compatibilità con vecchie chiamate
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
