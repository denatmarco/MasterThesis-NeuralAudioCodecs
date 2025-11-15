#!/usr/bin/env python3
"""
Encodec Embedding Extraction (24 kHz / 48 kHz)
PER-FRAGMENT EMBEDDING PIPELINE
"""

import warnings
warnings.filterwarnings("ignore")

import os
import math
import argparse
import subprocess

import numpy as np
import torch
import torchaudio
from torchaudio.io import StreamReader
from transformers import EncodecModel, AutoProcessor


# =====================================================================
# FFmpeg check
# =====================================================================
def _has_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return True
    except Exception:
        return False


# =====================================================================
# Resample + mono
# =====================================================================
def _resample_mono(audio: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor | None:
    try:
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)

        if audio.dim() == 2 and audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        return audio
    except Exception as e:
        print(f"⚠️ Resample failed: {e}")
        return None


# =====================================================================
# Robust loader
# =====================================================================
def prepare_audio(path: str, target_sr: int) -> torch.Tensor | None:
    if not isinstance(path, str) or not os.path.exists(path):
        print(f"⚠️ Invalid path: {path}")
        return None

    # torchaudio.load
    try:
        audio, sr = torchaudio.load(path)
        return _resample_mono(audio, sr, target_sr)
    except Exception:
        pass

    # StreamReader
    if _has_ffmpeg():
        try:
            reader = StreamReader(src=path)
            reader.add_audio_stream(frames_per_chunk=0, sample_rate=target_sr,
                                     channels=1, format="f32")
            chunks = []
            for (chunk,) in reader.stream():
                if chunk is None:
                    break
                chunks.append(chunk)
            if chunks:
                return torch.cat(chunks, dim=-1).unsqueeze(0)
        except Exception:
            pass

    # ffmpeg raw decode
    try:
        cmd = [
            "ffmpeg", "-v", "error", "-i", path,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(target_sr), "pipe:1"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=15, check=False)
        if proc.returncode == 0 and len(proc.stdout) > 0:
            wav = torch.frombuffer(np.frombuffer(proc.stdout, dtype=np.float32),
                                   dtype=torch.float32)
            return wav.unsqueeze(0)
    except Exception:
        pass

    # librosa fallback
    try:
        import librosa
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        if len(y) > 0:
            return torch.from_numpy(y).unsqueeze(0)
    except Exception:
        pass

    print(f"❌ All loaders failed for: {path}")
    return None


# =====================================================================
# Slice audio BEFORE Encodec
# =====================================================================
def slice_audio_before_encoder(audio: torch.Tensor, sr: int, window_sec: float = 5.0):
    samples_per_frag = int(round(window_sec * sr))
    total = audio.shape[1]

    n_frags = max(1, math.ceil(total / samples_per_frag))
    slices = []

    for i in range(n_frags):
        s = i * samples_per_frag
        e = s + samples_per_frag
        frag = audio[:, s:e]

        if frag.shape[1] < samples_per_frag:
            deficit = samples_per_frag - frag.shape[1]
            if frag.shape[1] > 0:
                pad = torch.flip(frag[:, -min(deficit, frag.shape[1]):], dims=[1])
                while pad.shape[1] < deficit:
                    pad = torch.cat([pad, torch.flip(pad, dims=[1])], dim=1)
                pad = pad[:, :deficit]
            else:
                pad = torch.zeros(1, deficit, dtype=audio.dtype, device=audio.device)
            frag = torch.cat([frag, pad], dim=1)

        slices.append(frag)

    return slices, samples_per_frag


# =====================================================================
# Normalize latent shape
# =====================================================================
def normalize_latent_shape(enc_out: torch.Tensor, window_frames: int, verbose=False):
    if enc_out.dim() != 2:
        raise ValueError(f"Expected 2D encoder output, got {enc_out.shape}")

    C, T = enc_out.shape
    if verbose:
        print(f"    [latent raw] shape = (C={C}, T={T})")

    if T < window_frames:
        deficit = window_frames - T
        pad = torch.flip(enc_out[:, -min(deficit, T):], dims=[1])
        while pad.shape[1] < deficit:
            pad = torch.cat([pad, torch.flip(pad, dims=[1])], dim=1)
        pad = pad[:, :deficit]
        enc_out = torch.cat([enc_out, pad], dim=1)
        if verbose:
            print(f"    [latent padded] new T = {enc_out.shape[1]}")

    elif T > window_frames:
        enc_out = enc_out[:, :window_frames]
        if verbose:
            print(f"    [latent cropped] new T = {enc_out.shape[1]}")

    return enc_out


# =====================================================================
# Verify saved fragment
# =====================================================================
def verify_saved_fragment(txt_path: str, original_latent: torch.Tensor, verbose=False):
    if not verbose:
        return True

    if not os.path.exists(txt_path):
        print(f"    [VERIFY] ❌ Missing: {txt_path}")
        return False

    loaded = np.loadtxt(txt_path, delimiter=",")
    if loaded.ndim == 1:
        loaded = loaded.reshape(1, -1)

    orig = original_latent.cpu().numpy()

    print(f"    [VERIFY] saved shape   = {loaded.shape}")
    print(f"    [VERIFY] latent shape  = {orig.shape}")

    if loaded.shape != orig.shape:
        print("    [VERIFY] ❌ Shape mismatch!")
        return False

    diff = np.abs(loaded - orig)
    max_diff = diff.max() if diff.size > 0 else 0.0
    print(f"    [VERIFY] max |Δ| = {max_diff:.3e}")

    return True


# =====================================================================
# Synthetic tone
# =====================================================================
def generate_test_wave(path: str, duration_sec: float = 3.0, sr: int = 24000) -> str:
    t = torch.linspace(0, duration_sec, int(duration_sec * sr))
    audio = 0.2 * torch.sin(2 * torch.pi * 440 * t)
    torchaudio.save(path, audio.unsqueeze(0), sr)
    return path


# =====================================================================
# Main extraction
# =====================================================================
def extract_encodec(audio_path: str, freq: int, output_dir: str,
                    window_sec: float = 5.0, verbose=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = f"facebook/encodec_{freq}khz"
    SAMPLE_RATE = freq * 1000

    os.makedirs(output_dir, exist_ok=True)

    print("🔹 Loading audio:", audio_path)
    audio = prepare_audio(audio_path, SAMPLE_RATE)
    if audio is None or audio.numel() == 0:
        print("❌ Failed to load audio.")
        return []

    print("🔹 Loading model:", MODEL_NAME)
    model = EncodecModel.from_pretrained(MODEL_NAME).to(device)
    _ = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"✂️  Splitting audio into {window_sec}-second chunks…")
    slices, samples_per_fragment = slice_audio_before_encoder(
        audio.to(device), SAMPLE_RATE, window_sec
    )
    print(f"   Total fragments: {len(slices)}")

    base = os.path.splitext(os.path.basename(audio_path))[0]
    results = []
    total_bytes = 0

    # Inspect first fragment to determine WINDOW_FRAMES
    first_frag = slices[0]
    with torch.no_grad():
        enc_first = model.encoder(first_frag.unsqueeze(0))[0]

    C_first, T_first = enc_first.shape
    WINDOW_FRAMES = T_first

    if verbose:
        print(f"\n   First latent shape: (C={C_first}, T={T_first})")
        print(f"   → WINDOW_FRAMES = {WINDOW_FRAMES}\n")

    # Process fragments
    for idx, frag_audio in enumerate(slices):
        if verbose:
            print(f"================ Fragment {idx} ================")
            print("  [audio] shape:", tuple(frag_audio.shape))

        with torch.no_grad():
            enc = model.encoder(frag_audio.unsqueeze(0))[0]

        if verbose:
            print("  [encoder raw] shape:", tuple(enc.shape))

        latent = normalize_latent_shape(enc, WINDOW_FRAMES, verbose)

        if verbose:
            print("  [latent final] shape:", tuple(latent.shape))

        txt_out = os.path.join(output_dir, f"{base}_frag{idx}.txt")
        np.savetxt(txt_out, latent.cpu().numpy(), fmt="%.6f", delimiter=",")

        verify_saved_fragment(txt_out, latent, verbose)

        results.append((txt_out, latent.shape))
        total_bytes += os.path.getsize(txt_out)

    # Summary (always shown)
    print("\n==============================")
    print("📌 EXTRACTION SUMMARY")
    print("==============================")
    print(f"📁 Output: {output_dir}")
    print(f"🔢 Fragments: {len(results)}")
    print(f"🧩 Latent shape (C x T): {C_first} x {WINDOW_FRAMES}")
    print(f"🔊 Samples per fragment: {samples_per_fragment}")

    for txt_p, shape in results:
        txt_kb = os.path.getsize(txt_p) / 1024
        print(f" - {os.path.basename(txt_p)} | latent={shape} | txt={txt_kb:.1f} KB")

    print(f"\n📦 Total size: {total_bytes/1024:.1f} KB")
    print("==============================\n")
    print("✅ Done.\n")

    return results


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=int, default=24, choices=[24, 48])
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default="encodec_output")
    parser.add_argument("--duration_sec", type=float, default=3.0)
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true",
                        help="Enable detailed per-fragment logging.")

    args = parser.parse_args()
    SAMPLE_RATE = args.freq * 1000

    if args.input is None:
        os.makedirs(args.output, exist_ok=True)
        test_audio = os.path.join(
            args.output,
            f"synthetic_{args.freq}khz_{args.duration_sec:.1f}s.wav",
        )
        print(f"⚠️ No input → generating synthetic tone: {test_audio}")
        generate_test_wave(test_audio, duration_sec=args.duration_sec, sr=SAMPLE_RATE)
        audio_path = test_audio
    else:
        audio_path = args.input

    extract_encodec(
        audio_path,
        args.freq,
        args.output,
        window_sec=args.window_sec,
        verbose=args.verbose,
    )
