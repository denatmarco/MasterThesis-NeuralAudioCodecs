#!/usr/bin/env python3
"""
CLI for extracting Third-Octave VQ embeddings from a trained encoder.
Provides 3 actions:
- audit: verify segmentation and write audit CSV
- extract: extract embeddings and write fragments/index CSVs
- full: run audit + extract together
"""

import os
import csv
import json
import shutil
import argparse
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# ======================================================================
# =============== Reuse utilities from the training CLI =================
# ======================================================================

def convert_to_wav(path_in: str, path_out: str, sr: int) -> str | None:
    """Convert any WAV/MP3 file to standardized WAV."""
    try:
        wav, sr0 = torchaudio.load(path_in)
        if sr0 != sr:
            wav = torchaudio.functional.resample(wav, sr0, sr)
        torchaudio.save(path_out, wav, sr)
        return path_out
    except Exception:
        return None


def collect_and_convert(folder: str, temp_dir: str, sr: int) -> list[str]:
    """Recursively collect WAV/MP3 and convert into target sr WAV."""
    os.makedirs(temp_dir, exist_ok=True)
    out = []

    all_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".wav", ".mp3")):
                all_files.append(os.path.join(root, f))

    for src in tqdm(all_files, desc=f"Convert→WAV", unit="file"):
        base = os.path.splitext(os.path.basename(src))[0] + ".wav"
        dst = os.path.join(temp_dir, base)
        res = convert_to_wav(src, dst, sr)
        if res:
            out.append(res)

    return out


# ======================================================================
# ========================= Audio utilities =============================
# ======================================================================

def to_mono_safe(wav: torch.Tensor) -> torch.Tensor:
    """Ensure mono; handle different channel lengths safely."""
    if wav.dim() == 1:
        return wav
    if wav.size(0) == 1:
        return wav.squeeze(0)

    max_len = wav.size(1)
    chans = []
    for ch in wav:
        if len(ch) < max_len:
            pad = torch.flip(ch, dims=[0])[:max_len - len(ch)]
            ch = torch.cat([ch, pad])
        else:
            ch = ch[:max_len]
        chans.append(ch)

    return torch.stack(chans).mean(dim=0)


# ======================================================================
# ==================== Continuous Embedding Extractor ===================
# ======================================================================

class ContinuousEmbeddingExtractor:
    """
    Extract continuous latent embeddings (z_e) using the FULL PIPELINE:
    waveform → FeatureExtractor → FeatureNormalizer → VQ Autoencoder
    """

    def __init__(self, ckpt: str, output_root: str, window_seconds: float, summary_name: str = "summary.json"):
        from third_octave_codec import EncoderSystem
        self.system = EncoderSystem.load_from_checkpoint(ckpt)
        self.system.model.eval()

        self.output_root = output_root
        self.window_seconds = window_seconds
        self.summary_name = summary_name

        self.sr = self.system.cfg.sr
        self.samples_per_window = int(self.window_seconds * self.sr)

    def segment_audio(self, wav: torch.Tensor):
        """Split audio into fixed-size windows (mirror-pad tail)."""
        N = wav.numel()
        W = self.samples_per_window

        full = N // W
        rem = N % W

        out = [wav[i*W:(i+1)*W] for i in range(full)]

        if rem > 0:
            tail = wav[-rem:]
            pad = torch.flip(tail, dims=[0])[:W-rem]
            out.append(torch.cat([tail, pad]))

        return out

    def encode_segment(self, seg: torch.Tensor):
        """Correct pipeline: extract → normalize → VQ-AE → z_e."""
        with torch.no_grad():
            bands = self.system.extractor(seg.unsqueeze(0))
            x = self.system.normalizer(bands)
            _, z_e, _, _, _ = self.system.model(x)
        return z_e.squeeze(0)   # (T, latent_dim)


# ======================================================================
# ================================ AUDIT ================================
# ======================================================================

def run_audit(dataset_root: str, extractor: ContinuousEmbeddingExtractor, temp_dir: str, audit_csv: str):

    valid = collect_and_convert(dataset_root, temp_dir, extractor.sr)
    os.makedirs(os.path.dirname(audit_csv), exist_ok=True)

    rows = []
    for wavpath in tqdm(valid, desc="Audit segments", unit="file"):
        wav, sr0 = torchaudio.load(wavpath)
        wav = to_mono_safe(wav)

        if sr0 != extractor.sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr0, extractor.sr).squeeze(0)

        orig = wav.numel()
        win = extractor.samples_per_window

        r = orig % win
        if r != 0:
            pad = torch.flip(wav, dims=[0])[:win - r]
            wav = torch.cat([wav, pad])

        padded = wav.numel()
        segs = extractor.segment_audio(wav)

        for idx, seg in enumerate(segs):
            rows.append([
                os.path.relpath(wavpath, dataset_root),
                idx, sr0, orig, padded, orig/sr0, extractor.window_seconds, r
            ])

    with open(audit_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file_path_rel","segment_idx","original_sr","original_samples",
            "padded_samples","duration_sec","window_sec","residual_before_pad"
        ])
        w.writerows(rows)

    print(f"✓ Audit completed → {audit_csv}")


# ======================================================================
# ============================= EXTRACTION ==============================
# ======================================================================

def extract_full_dataset(dataset_root: str, extractor: ContinuousEmbeddingExtractor,
                         temp_dir: str, out_root: str, batch_size=128):

    valid = collect_and_convert(dataset_root, temp_dir, extractor.sr)
    os.makedirs(out_root, exist_ok=True)

    index_csv = os.path.join(out_root, "fragments_index.csv")
    rows = []
    count = 0

    for wavpath in tqdm(valid, desc="Extract embeddings", unit="file"):
        wav, sr0 = torchaudio.load(wavpath)
        wav = to_mono_safe(wav)

        if sr0 != extractor.sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr0, extractor.sr).squeeze(0)

        segs = extractor.segment_audio(wav)
        rel = os.path.relpath(wavpath, dataset_root)

        tgt = os.path.join(out_root, Path(rel).parent)
        os.makedirs(tgt, exist_ok=True)

        for i, seg in enumerate(segs):
            z = extractor.encode_segment(seg)   # (T, latent_dim)

            outp = os.path.join(tgt, f"{Path(rel).stem}_seg{i:04d}.txt")
            np.savetxt(outp, z.cpu().numpy(), fmt="%.6f")

            rows.append([outp, rel, i])
            count += 1

    with open(index_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z_path","file_path_rel","segment_idx"])
        w.writerows(rows)

    print(f"✓ Extracted {count} embeddings")
    print(f"📄 Index saved → {index_csv}")


# ======================================================================
# ================================= CLI ================================
# ======================================================================

def cli_audit(args):
    extractor = ContinuousEmbeddingExtractor(args.checkpoint, args.output, args.window)
    temp = os.path.join(args.output, "temp_wavs")
    shutil.rmtree(temp, ignore_errors=True)
    run_audit(args.input, extractor, temp, os.path.join(args.output, "audit_segments.csv"))
    shutil.rmtree(temp, ignore_errors=True)


def cli_extract(args):
    extractor = ContinuousEmbeddingExtractor(args.checkpoint, args.output, args.window)
    temp = os.path.join(args.output, "temp_wavs")
    shutil.rmtree(temp, ignore_errors=True)
    extract_full_dataset(args.input, extractor, temp, args.output, batch_size=args.batch)
    shutil.rmtree(temp, ignore_errors=True)


def cli_full(args):
    cli_audit(args)
    cli_extract(args)


def build_parser():
    p = argparse.ArgumentParser(description="Third-Octave VQ Encoder – Embedding Extractor CLI")
    sub = p.add_subparsers(dest="action", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--window", type=float, default=1.0)
    a.set_defaults(func=cli_audit)

    e = sub.add_parser("extract")
    e.add_argument("--input", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--window", type=float, default=1.0)
    e.add_argument("--batch", type=int, default=256)
    e.set_defaults(func=cli_extract)

    f = sub.add_parser("full")
    f.add_argument("--input", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--checkpoint", required=True)
    f.add_argument("--window", type=float, default=1.0)
    f.add_argument("--batch", type=int, default=256)
    f.set_defaults(func=cli_full)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
