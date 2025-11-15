# Perch 2.0 Embedding Extraction (32 kHz / 16 kHz)

This script provides a **robust and fully self-contained pipeline** for extracting **per-fragment Perch 2.0 embeddings** from audio files.  
It supports the official **5-second, 32 kHz Perch configuration**, and an optional **10-second, 16 kHz forced mode** (`--force_10sec`).  
The pipeline performs **safe loading**, **resampling**, **mono conversion**, and **fixed-window slicing**, producing **one 1536-d embedding per fragment**, saved as `.txt` files.

**GPU execution is required**, because Perch 2.0 (TensorFlow/XLA) does not operate correctly on CPU.

---

## 1. Features

- Official Perch 2.0 mode: **32 kHz, 5-second windows**  
- Optional mode: **16 kHz, 10-second windows** (`--force_10sec`)  
- Multi-backend robust audio loading:
  - `soundfile`
  - `librosa`
  - raw PCM decoding via `ffmpeg`
- Automatic **mono conversion** and **resampling**
- Reflection-padded slicing into fixed-length fragments
- One embedding per fragment (shape: **1536**, 1D vector)
- Clean output (suppressed XLA logs)
- Automatic synthetic-tone generation when no input is provided

---

## 2. Installation

You need **TensorFlow with GPU support**, plus audio dependencies:

    pip install tensorflow keras numpy soundfile librosa huggingface-hub

You also need **FFmpeg** installed system-wide for maximum audio compatibility.

---

## 3. Command-Line Usage

### Basic extraction (5 s @ 32 kHz)

    python perch2.0_extract.py --input audio.wav --output out_dir

### Forced 10-second mode (16 kHz)

    python perch2.0_extract.py --input audio.wav --output out_dir --force_10sec

### Enable verbose logging

    python perch2.0_extract.py --input audio.wav --output out_dir --verbose

### No input → generate synthetic test tone

    python perch2.0_extract.py --output out_dir --duration_sec 8

---

## 4. Output Format

Each fragment produces:

    <basename>_fragX.txt

Each `.txt` file contains **a single row of 1536 comma-separated floats**, representing the Perch embedding.

**Example:**

    audio_frag0.txt | emb=(1536,) | 14.4 KB
    audio_frag1.txt | emb=(1536,) | 14.4 KB

---

## 5. Using the Script as a Python Module

### Extract embeddings programmatically

    from perch2_0_extract import extract_perch

    results = extract_perch(
        audio_path="audio.wav",
        output_dir="perch_latents",
        force_10sec=False,
        verbose=True
    )

    for path, shape in results:
        print("Saved:", path, "| shape =", shape)

### Load an embedding

    import numpy as np

    vec = np.loadtxt("perch_latents/audio_frag0.txt", delimiter=",")
    print(vec.shape)   # (1536,)

---

## 6. Pipeline Overview

### Audio Loading

Attempt order:

1. `soundfile`  
2. `librosa`  
3. raw `ffmpeg` PCM decode  

Files that cannot be decoded are rejected.

### Preprocessing

- Resamples to:
  - **32 kHz** (default)
  - **16 kHz** (forced 10-second mode)
- Converts to mono
- Optional synthetic tone generation

### Fragmentation

- Splitting into fixed windows:
  - **5 seconds @ 32 kHz**
  - **10 seconds @ 16 kHz**
- Last fragment padded using reflection padding

### Embedding Extraction

- Uses the **Perch 2.0 TFSMLayer**
- Produces a **1536-dimensional embedding vector** per fragment
- Requires **GPU + XLA**

### Saving

Each fragment is saved as a `.txt` file (CSV floats) with naming scheme:

    <basename>_frag0.txt
    <basename>_frag1.txt
    ...

---

## 7. Example Directory Structure

    perch_output/
    │
    ├── audio_frag0.txt
    ├── audio_frag1.txt
    ├── audio_frag2.txt
    └── synthetic_perch_32khz_8.0s.wav   # only if auto-generated

---

## 8. Notes

- Perch 2.0 requires **GPU acceleration**  
- All embeddings have fixed size: **1536**  
- Reflection padding stabilizes short trailing segments

---

## 9. License

This script is provided without warranty. You are free to use and adapt it for research or production.
