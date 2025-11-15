import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

class RandomWaveformDataset(Dataset):
    """
    Loads random waveform segments from a folder.
    Does NOT perform feature extraction here (done later on GPU in training loop).
    """

    def __init__(self, folder: str, target_sr: int = 16000, max_files: int | None = None, crop_duration: float = 2.0):
        assert os.path.isdir(folder), f"Dataset folder not found: {folder}"
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".wav")
        ]
        if max_files is not None:
            self.files = self.files[:max_files]

        self.target_sr = target_sr
        self.crop_samples = int(crop_duration * target_sr)
        print(f"📁 Loaded {len(self.files)} audio files from {folder}")

    def __len__(self):
        return len(self.files) * 16  # 16 random segments per file on average

    def __getitem__(self, idx):
        """Returns a random cropped waveform (mono, float32, CPU tensor)."""
        path = self.files[np.random.randint(0, len(self.files))]
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)  # mono
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        # Random crop of 2 seconds (or less if shorter)
        if wav.numel() > self.crop_samples:
            start = np.random.randint(0, wav.numel() - self.crop_samples)
            wav = wav[start:start + self.crop_samples]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.crop_samples - wav.numel()))

        return wav  # CPU tensor
