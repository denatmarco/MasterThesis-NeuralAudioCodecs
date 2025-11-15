```markdown
# 📘 Third-Octave VQ Encoder – Unified Command Line Interface

This repository provides two command-line tools for working with the **Third-Octave Vector-Quantized Encoder**:

1. **Training CLI (`train_trioct.py`)**
   Used to prepare audio datasets and train the encoder.

2. **Embedding Extraction CLI (`embedding_cli.py`)**
   Used to extract continuous latent embeddings (`z_e`) from a trained encoder.

Both tools ensure reproducibility, consistency, and ease of use.

---

# 📦 Installation

Install required dependencies:

~~~bash
pip install -r requirements.txt
~~~

The CLIs are:

- `train_trioct.py`
- `embedding_cli.py`

---

# 🚀 Training CLI – Usage Overview

General syntax:

~~~bash
python train_trioct.py <action> [options]
~~~

Available actions:

- `prepare`
- `train`
- `full`

Each requires:

- `--input` dataset path  
- `--output` checkpoints directory  

---

# 🔧 Commands (Training CLI)

## 1. prepare

Converts `.wav` / `.mp3` files to standardized **16 kHz WAV** format.

Pipeline:

- recursive scan  
- audio loading + resampling  
- WAV conversion  
- output to `output/temp_wavs/`

**Example:**

~~~bash
python train_trioct.py prepare \
    --input C:/Datasets/Lion \
    --output ./checkpoints_mammals
~~~

Output directory:

```
./checkpoints_mammals/temp_wavs/
```

---

## 2. train

Trains the Third-Octave VQ encoder.

Options:

- `--epochs` (40 default)  
- `--batch`  (32 default)  
- `--lr`     (2e-3 default)  
- `--device` (cuda default)  
- `--checkpoint` resume from file  

**Example:**

~~~bash
python train_trioct.py train \
    --input ./checkpoints_mammals/temp_wavs \
    --output ./checkpoints_mammals \
    --epochs 40 \
    --batch 32 \
    --lr 0.002
~~~

Generated:

```
encoder_final.pt
training_summary.json
```

---

## 3. full

Runs `prepare → train`.

**Example:**

~~~bash
python train_trioct.py full \
    --input C:/Datasets/Lion \
    --output ./checkpoints_mammals \
    --epochs 40 \
    --batch 32
~~~

---

# 📄 Output Structure (Training)

```
output/
│
├── temp_wavs/
├── encoder_final.pt
└── training_summary.json
```

---

# 🧪 Training Notes

- Ensure dataset cleanliness (no denoising performed).  
- GPU recommended (`--device cuda`).  
- Keep outputs separate per dataset.  
- Use `--checkpoint` to resume training.

---

# 📘 Embedding Extraction CLI – Usage Overview

General syntax:

~~~bash
python embedding_cli.py <action> [options]
~~~

Available actions:

- `audit`
- `extract`
- `full`

Each requires:

- `--input` dataset  
- `--output` embedding folder  
- `--checkpoint` encoder `.pt`  
- `--window`, `--batch`, etc.

Embeddings are saved as `.txt` matrices.

---

# 🔧 Commands (Embedding CLI)

## 1. audit

Generates segmentation metadata.

Steps:

- collect audio  
- convert to WAV  
- segment waveforms  
- export `audit_segments.csv`  

**Example:**

~~~bash
python embedding_cli.py audit \
    --input C:/Datasets/GTZAN \
    --output C:/Datasets/GTZAN/trioct/win_1/embeddings \
    --checkpoint ./checkpoints_music/bands_vq_encoder_full.pt \
    --window 1.0
~~~

Output:

```
output/audit_segments.csv
```

---

## 2. extract

Extracts embeddings for each segmented audio window.

Pipeline:

- audio→WAV  
- segmentation  
- encoder forward pass  
- save embedding as `.txt`  
- produce `fragments_index.csv`  

Embedding shape:

```
(latent_dim, time_frames)
```

**Example:**

~~~bash
python embedding_cli.py extract \
    --input C:/Datasets/GTZAN \
    --output C:/Datasets/GTZAN/trioct/win_1/embeddings \
    --checkpoint ./checkpoints_music/bands_vq_encoder_full.pt \
    --window 1.0 \
    --batch 256
~~~

Output structure:

```
output/
│
├── fragments_index.csv
├── file_subfolder/
│      ├── audio_seg0000.txt
│      ├── audio_seg0001.txt
│      └── ...
└── ...
```

---

## 3. full

Runs `audit → extract`.

**Example:**

~~~bash
python embedding_cli.py full \
    --input C:/Datasets/GTZAN \
    --output C:/Datasets/GTZAN/trioct/win_1/embeddings \
    --checkpoint ./checkpoints_music/bands_vq_encoder_full.pt \
    --window 1.0 \
    --batch 256
~~~

Outputs:

- `audit_segments.csv`  
- `fragments_index.csv`  
- all `.txt` embeddings  

---

# 📄 Output Files Overview

| File                   | Description                 |
|------------------------|-----------------------------|
| audit_segments.csv     | segmentation metadata       |
| fragments_index.csv    | list of fragments           |
| *_segXXXX.txt          | embedding matrices          |

Load an embedding:

~~~python
import numpy as np
z = np.loadtxt("path/to/fragment.txt")
~~~

---

# 🧪 Extraction Notes

- Use the same window size across datasets.  
- Increase `--batch` for speed (SSD recommended).  
- Do not mix datasets in the same output folder.  
- Audio is automatically resampled to match encoder requirements.
```
