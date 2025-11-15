# Encodec Embedding Extraction (24 kHz / 48 kHz)

This script provides a **robust and fully self-contained pipeline** for extracting **per-fragment Encodec latent embeddings** from audio files.  
It supports both **24 kHz** and **48 kHz** Encodec models, performs **automatic resampling**, **mono conversion**, **safe loading** through multiple backends, and **consistent latent normalization** to ensure that all fragments share the same temporal dimension.

All extracted embeddings are saved as **`.txt` files** containing the latent matrix in CSV format.

---

## 1. Features

- Supports **Encodec 24 kHz** and **Encodec 48 kHz**.
- Automatic fallback audio loading using:
  - `torchaudio.load`
  - `ffmpeg` StreamReader
  - raw `ffmpeg` PCM decoding
  - `librosa`
- Robust **mono conversion** and **resampling**.
- Splits audio **before encoding** into fixed-duration fragments (default: 5 seconds).
- Ensures **uniform latent size** across all fragments.
- Optional **verbose inspection** for each fragment.
- Command-line interface and reusable Python API.
- Auto-generation of a test sine wave when no input is provided.

---

## 2. Installation

    pip install torch torchaudio transformers librosa numpy

You also need **FFmpeg** installed and available in your system `PATH` for maximum loader compatibility.

---

## 3. Command-Line Usage

### Basic extraction

    python encodec_extract.py --input audio.wav --freq 24 --output out_dir

### Extract using 48 kHz model

    python encodec_extract.py --input audio.wav --freq 48 --output out_48khz

### Change fragment length (e.g., 10-second windows)

    python encodec_extract.py --input audio.wav --window_sec 10

### Enable verbose logging

    python encodec_extract.py --input audio.wav --verbose

### No input provided → generate synthetic test tone

    python encodec_extract.py

This produces a synthetic sine wave and extracts its embeddings.

---

## 4. Output Format

Each fragment produces a file:

    <basename>_fragX.txt

containing a latent matrix of shape:

    C × T

where:

- `C` = number of latent channels (model-dependent)  
- `T` = fixed number of frames (normalized for all fragments)

**Example:**

    audio_frag0.txt | latent=(128, 195) | 98.4 KB
    audio_frag1.txt | latent=(128, 195) | 98.4 KB

---

## 5. Using the Script as a Python Module

### Extract embeddings programmatically

    from encodec_extract import extract_encodec

    results = extract_encodec(
        audio_path="audio.wav",
        freq=24,                     # 24 or 48 kHz
        output_dir="encodec_latents",
        window_sec=5.0,
        verbose=True
    )

    for path, shape in results:
        print("Saved:", path, "| shape =", shape)

### Load and inspect a latent fragment

    import numpy as np

    latent = np.loadtxt("encodec_latents/audio_frag0.txt", delimiter=",")
    print("Latent shape:", latent.shape)

---

## 6. Pipeline Overview

### Audio loading

- Tries several backends until one succeeds.
- Invalid or corrupted files are skipped.

### Preprocessing

- Resampling to `freq * 1000` Hz.
- Mono conversion.

### Fragmentation

- Splits waveform into fixed-length segments.
- Pads the last fragment by symmetrical reflection.

### Encodec encoding

- Loads `facebook/encodec_<freq>khz`.
- Encodes each fragment independently.

### Latent normalization

- Ensures all latent matrices have identical frame count.
- Pads or crops symmetrically.

### Saving

- Each fragment is stored as a `.txt` file (CSV float format).

### Optional verification

- With `--verbose`, saved fragments are compared to original tensors.

---

## 7. Example Directory Structure

    encodec_output/
    │
    ├── audio_frag0.txt
    ├── audio_frag1.txt
    ├── audio_frag2.txt
    └── synthetic_24khz_3.0s.wav   # only if auto-generated

---

## 8. Notes

- GPU (CUDA) is used automatically if available.
- Fragment padding uses reflection, improving continuity for short clips.
- The script guarantees that all `.txt` files corresponding to one audio file are shape-consistent, which is essential for downstream ML models.

---

## 9. License

This script is provided without warranty. You are free to use and adapt it for research or production.
