#!/usr/bin/env python3
"""
Simple CLI for training the Third-Octave VQ Encoder.
This script provides 3 main actions:
- prepare: convert all audio under --input into WAV under --output/temp
- train: train the encoder using the WAVs under --input
- full: prepare + train in a single command

Usage example:
    python train_trioct.py full --input C:/Datasets/Lion --output ./checkpoints_mammals
"""

import os
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from third_octave_codec import (
    EncoderSystem,
    train_encoder,
    FeatureExtractor
)

# ============================================
# GLOBAL AUDIO/MODEL DEFAULTS
# ============================================

SR = 16000
N_FFT = 512
HOP = 256
FMIN = 50.0
FMAX = None

LATENT_DIM = 96
HIDDEN = 256
NUM_GROUPS = 2
CODEBOOK_SIZE = 256
BETA = 0.05

DEFAULT_EPOCHS = 40
DEFAULT_BATCH = 1
DEFAULT_LR = 2e-3
DEFAULT_DEVICE = "cuda"
DEFAULT_SEED = 42


# =============================================================
# =============== AUDIO PREPARATION UTILITIES =================
# =============================================================

def convert_to_wav(path_in: str, path_out: str, target_sr: int = SR) -> str | None:
    """Convert any WAV/MP3 file to standardized WAV."""
    try:
        wav, sr = torchaudio.load(path_in)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        torchaudio.save(path_out, wav, target_sr)
        return path_out
    except Exception:
        return None


def collect_and_convert(folder: str, temp_dir: str) -> list[str]:
    """Recursively collect WAV/MP3 and convert everything into WAV."""
    os.makedirs(temp_dir, exist_ok=True)
    paths = []

    all_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".wav", ".mp3")):
                all_files.append(os.path.join(root, f))

    for src in tqdm(all_files, desc=f"Converting audio in {folder}"):
        base = os.path.splitext(os.path.basename(src))[0] + ".wav"
        out = os.path.join(temp_dir, base)
        res = convert_to_wav(src, out)
        if res:
            paths.append(res)

    return paths


# =============================================================
# ====================== TRAINING UTILITIES ===================
# =============================================================

def train_wav_folder(
    wav_folder: str,
    out_dir: str,
    epochs: int,
    batch: int,
    lr: float,
    device: str,
    prev_ckpt: str | None = None
):
    """Train the encoder on a WAV-only folder."""
    os.makedirs(out_dir, exist_ok=True)

    if prev_ckpt is None:
        ckpt = train_encoder(
            train_folder=wav_folder,
            out_dir=out_dir,
            sr=SR,
            n_fft=N_FFT,
            hop=HOP,
            latent_dim=LATENT_DIM,
            hidden=HIDDEN,
            num_groups=NUM_GROUPS,
            codebook_size=CODEBOOK_SIZE,
            beta=BETA,
            epochs=epochs,
            batch_size=batch,
            lr=lr,
            device=device,
            seed=DEFAULT_SEED
        )
    else:
        system = EncoderSystem.load_from_checkpoint(prev_ckpt)
        system.cfg.train_folder = wav_folder
        system.cfg.epochs = epochs
        system.cfg.batch_size = batch
        system.cfg.lr = lr
        system.cfg.device = device
        ckpt = system.train()

    # Evaluate metrics on this folder
    system = EncoderSystem.load_from_checkpoint(ckpt)
    metrics = system.evaluate_folder(wav_folder)

    return ckpt, metrics


# =============================================================
# ============================ CLI ============================
# =============================================================

def cli_prepare(args):
    """Convert input audio into standardized WAVs under output/temp."""
    temp_dir = os.path.join(args.output, "temp_wavs")
    shutil.rmtree(temp_dir, ignore_errors=True)

    wavs = collect_and_convert(args.input, temp_dir)

    print(f"\n✓ Prepared {len(wavs)} WAV files in: {temp_dir}")
    if len(wavs) == 0:
        print("⚠ No audio found.")
    return temp_dir


def cli_train(args):
    """Train model using WAVs under --input."""
    if not os.path.isdir(args.input):
        raise ValueError("Training requires a folder with WAV files.")

    ckpt, metrics = train_wav_folder(
        wav_folder=args.input,
        out_dir=args.output,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        prev_ckpt=args.checkpoint
    )

    out_ckpt = os.path.join(args.output, "encoder_final.pt")
    shutil.copy(ckpt, out_ckpt)

    summary_path = os.path.join(args.output, "training_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "final_checkpoint": out_ckpt,
        "metrics": metrics,
        "config": vars(args),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Training complete.")
    print(f"  Final model: {out_ckpt}")
    print(f"  Summary: {summary_path}")


def cli_full(args):
    """Perform full pipeline: prepare audio → train."""
    temp_dir = cli_prepare(args)

    print("\n--- START TRAINING ---\n")

    args_for_train = argparse.Namespace(
        input=temp_dir,
        output=args.output,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        checkpoint=None
    )
    cli_train(args_for_train)


# =============================================================
# ======================== MAIN ENTRY ==========================
# =============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Simple CLI for Third-Octave VQ Encoder training."
    )

    sub = parser.add_subparsers(dest="action", required=True)

    # prepare
    p = sub.add_parser("prepare", help="Convert audio to WAV format.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cli_prepare)

    # train
    p = sub.add_parser("train", help="Train encoder on WAVs.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr",    type=float, default=DEFAULT_LR)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.set_defaults(func=cli_train)

    # full
    p = sub.add_parser("full", help="Prepare + train.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr",    type=float, default=DEFAULT_LR)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.set_defaults(func=cli_full)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
