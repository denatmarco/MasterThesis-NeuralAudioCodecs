#!/usr/bin/env python3
"""
Perch 2.0 Embedding Extraction
PER-FRAGMENT EMBEDDING PIPELINE

- Loads a single audio file
- Resamples and converts to mono
- Splits into fixed-length fragments (default: 5s @ 32 kHz)
  or (10s @ 16 kHz) when --force_10sec is used
- Extracts one Perch 2.0 embedding per fragment
- Saves each embedding as a .txt file (CSV row)
"""

import os
import warnings
warnings.filterwarnings("ignore")
import math
import argparse
import subprocess

import numpy as np
import soundfile as sf
import librosa

import tensorflow as tf
import keras
from huggingface_hub import snapshot_download


# =====================================================================
# Global model handle
# =====================================================================
_PERCH_INFER = None


def load_perch_model(verbose: bool = True):
    """Load Perch 2.0 model as a global TFSMLayer (lazy initialization)."""
    global _PERCH_INFER
    if _PERCH_INFER is not None:
        return _PERCH_INFER

    if verbose:
        print("========================================")
        print("🧠 Perch 2.0 MODEL LOADING")
        print("========================================")
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print("----------------------------------------")
        print("📥 Downloading Perch 2.0 model from Hugging Face...")

    model_path = snapshot_download(repo_id="cgeorgiaw/Perch")

    if verbose:
        print("🔧 Loading model as TFSMLayer...")

    infer = keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")

    # Quick sanity-check embedding
    if verbose:
        waveform = np.zeros(5 * 32000, dtype=np.float32)
        x = np.expand_dims(waveform, axis=0)
        x_tf = tf.constant(x)
        outputs = infer(inputs=x_tf)
        emb = outputs["embedding"].numpy()
        print("✅ Test embedding shape:", emb.shape)
        print("========================================\n")

    _PERCH_INFER = infer
    return _PERCH_INFER


# =====================================================================
# FFmpeg presence check (for info only)
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
# Robust audio loader (SoundFile + Librosa)
# =====================================================================
def prepare_audio(path: str, target_sr: int) -> np.ndarray | None:
    """
    Load, resample and convert to mono using SoundFile with Librosa fallback.

    Returns:
        np.ndarray with shape (1, T) in float32, or None on failure.
    """
    if not isinstance(path, str) or not os.path.exists(path):
        print(f"⚠️ Invalid path: {path}")
        return None

    # Primary: SoundFile
    try:
        audio, sr = sf.read(path, dtype="float32")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)

        return audio.astype("float32")
    except Exception:
        pass

    # Fallback: Librosa
    try:
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        return audio.astype("float32")
    except Exception:
        print(f"❌ All loaders failed for: {path}")
        return None


# =====================================================================
# Slice audio BEFORE Perch encoder
# =====================================================================
def slice_audio_before_encoder(
    audio: np.ndarray,
    sr: int,
    window_sec: float
):
    """
    Split audio into fixed-length fragments using mirror padding.

    Args:
        audio (np.ndarray): shape (1, T)
        sr (int): sample rate
        window_sec (float): window length in seconds (5.0 or 10.0)

    Returns:
        (list[np.ndarray], int): list of fragments (1, samples_per_frag) and samples_per_frag
    """
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
                pad = np.flip(frag[:, -min(deficit, frag.shape[1]):], axis=1)
                while pad.shape[1] < deficit:
                    pad = np.concatenate([pad, np.flip(pad, axis=1)], axis=1)
                pad = pad[:, :deficit]
            else:
                pad = np.zeros((1, deficit), dtype=audio.dtype)
            frag = np.concatenate([frag, pad], axis=1)

        slices.append(frag.astype("float32"))

    return slices, samples_per_frag


# =====================================================================
# Verify saved fragment
# =====================================================================
def verify_saved_fragment(txt_path: str, original_vec: np.ndarray, verbose: bool = False):
    """
    Optionally verify that the saved embedding matches the original vector.

    Args:
        txt_path (str): path to saved .txt file
        original_vec (np.ndarray): 1D or 2D embedding array
        verbose (bool): if True, print detailed info
    """
    if not verbose:
        return True

    if not os.path.exists(txt_path):
        print(f"    [VERIFY] ❌ Missing: {txt_path}")
        return False

    loaded = np.loadtxt(txt_path, delimiter=",")
    if loaded.ndim == 1:
        loaded = loaded.reshape(1, -1)

    if original_vec.ndim == 1:
        orig = original_vec.reshape(1, -1)
    else:
        orig = original_vec.copy()

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
# Synthetic tone generator
# =====================================================================
def generate_test_wave(path: str, duration_sec: float = 3.0, sr: int = 32000) -> str:
    """
    Generate a synthetic sine wave tone for testing purposes.
    """
    t = np.linspace(0.0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype("float32")
    sf.write(path, audio, sr)
    return path


# =====================================================================
# Single-fragment Perch embedding
# =====================================================================
def extract_embedding_from_fragment(fragment: np.ndarray, infer_layer) -> np.ndarray:
    """
    Extract a single Perch embedding from an audio fragment.

    Args:
        fragment (np.ndarray): shape (1, T), float32
        infer_layer: Keras TFSMLayer for Perch

    Returns:
        np.ndarray: 1D embedding vector (1536,)
    """
    arr = fragment.squeeze(0).astype("float32")
    x_tf = tf.constant(np.expand_dims(arr, axis=0))
    out = infer_layer(inputs=x_tf)
    emb_np = out["embedding"].numpy().squeeze()
    return emb_np  # shape (1536,)


# =====================================================================
# Main extraction routine
# =====================================================================
def extract_perch(
    audio_path: str,
    output_dir: str,
    force_10sec: bool = False,
    verbose: bool = False
):
    """
    Main entry point:
    - load audio
    - split into fixed-length fragments
    - extract one Perch embedding per fragment
    - save as .txt files (CSV row)
    """

    # Default mode: 5s @ 32 kHz
    # Forced mode: 10s @ 16 kHz
    if force_10sec:
        sample_rate = 16000
        window_sec = 10.0
    else:
        sample_rate = 32000
        window_sec = 5.0

    os.makedirs(output_dir, exist_ok=True)

    print("========================================")
    print("🔹 Perch 2.0 EXTRACTION CONFIG")
    print("========================================")
    print(f"🎧 Audio file:     {audio_path}")
    print(f"🎚 Sample rate:    {sample_rate} Hz")
    print(f"⏱  Window length:  {window_sec} seconds")
    print(f"📁 Output folder:  {output_dir}")
    print("----------------------------------------")
    print(f"🔍 FFmpeg present: {_has_ffmpeg()}")
    print("========================================\n")

    print("🔹 Loading audio...")
    audio = prepare_audio(audio_path, sample_rate)
    if audio is None or audio.size == 0:
        print("❌ Failed to load audio.")
        return []

    print("🔹 Loading Perch 2.0 model...")
    infer = load_perch_model(verbose=verbose)

    print(f"✂️  Splitting audio into {window_sec}-second chunks…")
    slices, samples_per_fragment = slice_audio_before_encoder(
        audio, sample_rate, window_sec
    )
    print(f"   Total fragments: {len(slices)}")

    base = os.path.splitext(os.path.basename(audio_path))[0]
    results = []
    total_bytes = 0

    # Inspect first fragment to obtain embedding dimension
    first_frag = slices[0]
    emb_first = extract_embedding_from_fragment(first_frag, infer)
    EMB_DIM = emb_first.shape[0]

    if verbose:
        print(f"\n   First embedding shape: (D={EMB_DIM})\n")

    # Save first embedding
    txt_first = os.path.join(output_dir, f"{base}_frag0.txt")
    np.savetxt(txt_first, emb_first.reshape(1, -1), fmt="%.6f", delimiter=",")
    verify_saved_fragment(txt_first, emb_first, verbose)
    results.append((txt_first, emb_first.shape))
    total_bytes += os.path.getsize(txt_first)

    # Process remaining fragments
    for idx, frag_audio in enumerate(slices[1:], start=1):
        if verbose:
            print(f"================ Fragment {idx} ================")
            print("  [audio] shape:", tuple(frag_audio.shape))

        emb = extract_embedding_from_fragment(frag_audio, infer)

        if verbose:
            print("  [embedding] shape:", emb.shape)

        txt_out = os.path.join(output_dir, f"{base}_frag{idx}.txt")
        np.savetxt(txt_out, emb.reshape(1, -1), fmt="%.6f", delimiter=",")
        verify_saved_fragment(txt_out, emb, verbose)

        results.append((txt_out, emb.shape))
        total_bytes += os.path.getsize(txt_out)

    # Summary (always shown)
    print("\n==============================")
    print("📌 PERCH EXTRACTION SUMMARY")
    print("==============================")
    print(f"📁 Output: {output_dir}")
    print(f"🔢 Fragments: {len(results)}")
    print(f"🧩 Embedding dimension: {EMB_DIM}")
    print(f"🔊 Samples per fragment: {samples_per_fragment}")

    for txt_p, shape in results:
        txt_kb = os.path.getsize(txt_p) / 1024
        print(f" - {os.path.basename(txt_p)} | emb={shape} | txt={txt_kb:.1f} KB")

    print(f"\n📦 Total size: {total_bytes/1024:.1f} KB")
    print("==============================\n")
    print("✅ Done.\n")

    return results


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perch 2.0 per-fragment embedding extraction."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input audio file. If omitted, a synthetic tone is generated."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="perch_output",
        help="Output directory for .txt embeddings."
    )
    parser.add_argument(
        "--duration_sec",
        type=float,
        default=3.0,
        help="Duration in seconds for synthetic tone when --input is not provided."
    )
    parser.add_argument(
        "--force_10sec",
        action="store_true",
        help="Force 10-second window at 16 kHz instead of 5-second window at 32 kHz."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed per-fragment logging."
    )

    args = parser.parse_args()

    # Choose sample rate according to the selected mode
    if args.force_10sec:
        SAMPLE_RATE = 16000
    else:
        SAMPLE_RATE = 32000

    if args.input is None:
        os.makedirs(args.output, exist_ok=True)
        test_audio = os.path.join(
            args.output,
            f"synthetic_perch_{SAMPLE_RATE//1000}khz_{args.duration_sec:.1f}s.wav",
        )
        print(f"⚠️ No input → generating synthetic tone: {test_audio}")
        generate_test_wave(test_audio, duration_sec=args.duration_sec, sr=SAMPLE_RATE)
        audio_path = test_audio
    else:
        audio_path = args.input

    extract_perch(
        audio_path=audio_path,
        output_dir=args.output,
        force_10sec=args.force_10sec,
        verbose=args.verbose,
    )
