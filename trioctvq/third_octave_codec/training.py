"""
High-level training and dataset encoding helpers.
"""

from __future__ import annotations
from typing import Optional

from .encoder_system import CodecConfig, EncoderSystem


def train_encoder(
    train_folder: str,
    out_dir: str = "./checkpoints",
    sr: int = 16000,
    n_fft: int = 512,
    hop: int = 128,
    fmin: float = 50.0,
    fmax: Optional[float] = None,
    latent_dim: int = 24,
    hidden: int = 96,
    num_groups: int = 2,
    codebook_size: int = 256,
    beta: float = 0.25,
    batch_size: int = 512,
    epochs: int = 25,
    lr: float = 2e-3,
    device: str = "cuda",
    seed: int = 0,
) -> str:
    """Train the encoder on WAV datasets and save a checkpoint. Return path."""
    cfg = CodecConfig(
        sr=sr,
        n_fft=n_fft,
        hop=hop,
        fmin=fmin,
        fmax=fmax,
        latent_dim=latent_dim,
        hidden=hidden,
        num_groups=num_groups,
        codebook_size=codebook_size,
        beta=beta,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device=device,
        train_folder=train_folder,
        val_folder=None,
        out_dir=out_dir,
        seed=seed,
    )
    system = EncoderSystem(cfg)
    return system.train()


def encode_dataset(ckpt_path: str, input_folder: str, output_folder: str) -> None:
    """Load a trained encoder and encode all WAVs in input_folder."""
    system = EncoderSystem.load_from_checkpoint(ckpt_path)
    system.encode_folder(input_folder, output_folder)
