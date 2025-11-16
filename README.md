# 📁 Repository Structure – NeuralAudioCodecs

This document briefly describes the purpose and contents of each main directory in the repository.

---

## 📂 encodec_extraction
This directory contains the complete pipeline for extracting **Encodec latent embeddings**  
(24 kHz and 48 kHz).  
It includes:
- multi-backend audio loading and resampling  
- waveform fragmentation  
- HuggingFace EnCodec model loading  
- latent extraction and saving as `.txt` files  

Useful for obtaining **dense continuous latent representations** from EnCodec.

---

## 📂 perch2.0_extraction
This directory provides the GPU-based extractor for **Perch 2.0 embeddings**  
(official 32 kHz 5-second mode + optional 16 kHz 10-second forced mode).  
Contents include:
- robust audio loader (soundfile + librosa)
- fragmentation with reflection padding
- Perch 2.0 TFSMLayer model loading
- generation of **1536-dimensional embeddings** per fragment as `.txt` files  
- synthetic tone generator for testing  

Useful for **semantic embeddings** derived from Perch 2.0.

### ⚠️ Note on Xeno-Canto–based Datasets

Perch 2.0 was partially trained on large-scale bird audio collections that include curated subsets derived from Xeno-Canto.
For this reason, results obtained using datasets sourced from Xeno-Canto should be interpreted with caution, as the pretrained model may implicitly benefit from prior exposure to similar material. This does not invalidate the extraction pipeline, but it means that performance on Xeno-Canto–like datasets may not fully reflect out-of-distribution generalization.
---

## 📂 trioctvq
This directory contains all components related to the **Third-Octave Vector-Quantized Encoder**, including both training and embedding extraction workflows.

Contents include:
- the complete **Third-Octave VQ Encoder** implementation  
- **Training CLI** (`train_trioct.py`)
- **Embedding Extraction CLI** (`embedding_cli.py`)
- dataset preparation pipeline  
- utility scripts for segmentation, metadata, and batching  
- **pretrained models used in the Master Thesis**, specifically:
  - **Mammals** 
  - **BirdZ**   
  - **Music**   

These models are the exact versions used to generate results and analysis in the thesis.

Useful for:
- training new third-octave VQ encoders  
- using the pretrained encoders for inference  
- reproducing the experiments and results of the thesis

---
