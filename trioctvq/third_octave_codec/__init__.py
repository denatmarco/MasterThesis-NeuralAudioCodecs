"""
third_octave_codec
==================

This package exposes:
- Training utilities (`train_encoder`, `encode_dataset`)
- Encoder system configuration and checkpoint handling (`EncoderSystem`, `CodecConfig`)
- Audio feature extraction (`FeatureExtractor`, `FeatureNormalizer`)
- Band analysis and utilities (`ThirdOctaveMapper`, `ThirdOctaveBand`)
- Metrics for evaluation (`entropy_bits`, `codebook_usage`, `bitrate_estimate`)
"""

# === Core system ===
from .encoder_system import CodecConfig, EncoderSystem

# === Training ===
from .training import train_encoder, encode_dataset

# === Features and third-octave mapping ===
from .features import FeatureExtractor, FeatureNormalizer
from .bands_utils import ThirdOctaveMapper, ThirdOctaveBand

# === Evaluation utilities ===
from .metrics_utils import entropy_bits, codebook_usage, bitrate_estimate

__all__ = [
    # Core
    "CodecConfig",
    "EncoderSystem",
    # Training
    "train_encoder",
    "encode_dataset",
    # Features
    "FeatureExtractor",
    "FeatureNormalizer",
    # Bands
    "ThirdOctaveMapper",
    "ThirdOctaveBand",
    # Metrics
    "entropy_bits",
    "codebook_usage",
    "bitrate_estimate",
]
