"""
third_octave_codec/encoder_system.py
Manages configuration, model initialization, training, checkpointing,
metric tracking, and visualization for the third-octave VQ autoencoder system.
"""

from __future__ import annotations
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchaudio

from .features import FeatureExtractor, FeatureNormalizer
from .vq_autoencoder import BandsVQAutoencoder
from .datasets import RandomWaveformDataset
from .metrics_utils import entropy_bits, codebook_usage, bitrate_estimate


# ===============================
# CONFIGURATION
# ===============================
@dataclass
class CodecConfig:
    sr: int = 16000
    n_fft: int = 512
    hop: int = 512
    fmin: float = 50.0
    fmax: float | None = None
    power: float = 2.0
    use_log: bool = True
    latent_dim: int = 24
    hidden: int = 96
    num_groups: int = 2
    codebook_size: int = 256
    beta: float = 0.25
    batch_size: int = 128
    epochs: int = 10
    lr: float = 2e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_folder: str = "./train_wavs"
    val_folder: str | None = None
    max_files: int | None = None
    out_dir: str = "./checkpoints"
    seed: int = 0


# ===============================
# MAIN SYSTEM
# ===============================
class EncoderSystem:
    def __init__(self, cfg: CodecConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # === Feature extractor and VQ-AE model ===
        self.extractor = FeatureExtractor(
            sr=cfg.sr,
            n_fft=cfg.n_fft,
            hop=cfg.hop,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            power=cfg.power,
            use_log=cfg.use_log,
            device=cfg.device,
        )
        self.normalizer = FeatureNormalizer()
        self.model = BandsVQAutoencoder(
            num_bands=self.extractor.mapper.num_bands,
            latent_dim=cfg.latent_dim,
            hidden=cfg.hidden,
            num_groups=cfg.num_groups,
            codebook_size=cfg.codebook_size,
            beta=cfg.beta,
        ).to(self.device)

    # ===============================
    # TRAINING LOOP
    # ===============================
    def train(self) -> str:
        """
        Train the vector-quantized autoencoder system.
        Returns the path to the best checkpoint.
        """
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.device = torch.device(self.cfg.device)

        print(f"✅ Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        # === Dataset ===
        train_ds = RandomWaveformDataset(
            self.cfg.train_folder, target_sr=self.cfg.sr, max_files=self.cfg.max_files
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

        # === Model / Optimizer / AMP ===
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
        torch.set_float32_matmul_precision("high")

        os.makedirs(self.cfg.out_dir, exist_ok=True)
        ckpt_path = os.path.join(self.cfg.out_dir, "bands_vq_encoder.pt")
        best_loss = float("inf")

        # === Metric logging ===
        metrics_log = {"epoch": [], "loss": [], "entropy_bits": [], "codebook_usage": [], "bitrate_kbps": []}

        for epoch in tqdm(range(1, self.cfg.epochs + 1), desc="Training progress", unit="epoch"):
            self.model.train()
            running, n_rows = 0.0, 0
            all_entropy, all_usage, all_bitrate = [], [], []

            with tqdm(train_loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", unit="batch", leave=False) as pbar:
                for wavs in pbar:
                    wavs = wavs.to(self.device, non_blocking=True)

                    # === Feature extraction ===
                    with torch.no_grad():
                        bands = self.extractor(wavs)  # (B, T, F)

                    # === Per-sample normalization (stateless) ===
                    x = self.normalizer(bands)      # (B, T, F)
                    opt.zero_grad(set_to_none=True)

                    # === Forward pass (epoch-aware beta annealing) ===
                    with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                        bands_hat, z_e, z_q, idx, vq_loss = self.model(x, epoch=epoch)
                        recon = F.mse_loss(bands_hat, x)
                        loss = recon + vq_loss

                    # === Backward ===
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                    # === VQ metrics (per group) ===
                    with torch.no_grad():
                        # idx: (B, T, G)
                        B, T, G = idx.shape
                        K = self.cfg.codebook_size
                        frames_per_second = self.cfg.sr / self.cfg.hop

                        entropies, usages = [], []
                        for g in range(G):
                            inds = idx[:, :, g].detach().reshape(-1).cpu().numpy()
                            hist, _ = np.histogram(inds, bins=np.arange(K + 1), density=False)
                            entropies.append(entropy_bits(hist))
                            _, usage_ratio = codebook_usage(hist)
                            usages.append(usage_ratio * 100.0)

                        H = float(np.mean(entropies))
                        usage = float(np.mean(usages))
                        bitrate = float(bitrate_estimate(frames_per_second, G, K))  # already kbps if defined so

                        all_entropy.append(H)
                        all_usage.append(usage)
                        all_bitrate.append(bitrate)

                    # === Running loss for nice progress bar ===
                    running += loss.item() * x.size(0)
                    n_rows += x.size(0)
                    mean_loss = running / max(1, n_rows)
                    pbar.set_postfix(loss=f"{mean_loss:.6f}", entropy=f"{H:.2f}", usage=f"{usage:.1f}%")

            # === Epoch end ===
            epoch_entropy = float(np.mean(all_entropy))
            epoch_usage = float(np.mean(all_usage))
            epoch_bitrate = float(np.mean(all_bitrate))

            self.print_codebook_stats(idx)
            
            # Log metrics
            metrics_log["epoch"].append(epoch)
            metrics_log["loss"].append(mean_loss)
            metrics_log["entropy_bits"].append(epoch_entropy)
            metrics_log["codebook_usage"].append(epoch_usage)
            metrics_log["bitrate_kbps"].append(epoch_bitrate)

            # Save best checkpoint
            if mean_loss < best_loss:
                best_loss = mean_loss
                self._save_checkpoint(ckpt_path, best_loss)

            # Persist metrics JSON and PNG curves
            metrics_json_path = os.path.join(self.cfg.out_dir, "training_metrics.json")
            with open(metrics_json_path, "w") as f:
                json.dump(metrics_log, f, indent=2)

            plt.figure(figsize=(10, 6))
            plt.subplot(3, 1, 1)
            plt.plot(metrics_log["epoch"], metrics_log["loss"], label="Loss", color="tab:red")
            plt.ylabel("Loss"); plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(metrics_log["epoch"], metrics_log["entropy_bits"], label="Entropy (bits)", color="tab:blue")
            plt.plot(metrics_log["epoch"], metrics_log["codebook_usage"], label="Codebook Usage (%)", color="tab:green")
            plt.ylabel("VQ Metrics"); plt.legend(); plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(metrics_log["epoch"], metrics_log["bitrate_kbps"], label="Bitrate (kbps)", color="tab:orange")
            plt.xlabel("Epoch"); plt.ylabel("Bitrate (kbps)"); plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg.out_dir, "training_curves.png"))
            plt.close()

        print(f"🏁 Training finished. Best loss={best_loss:.6f}. Saved to {ckpt_path}")
        return ckpt_path


    # ===============================
    # CHECKPOINT SAVE / LOAD
    # ===============================
    def _save_checkpoint(self, path: str, best_loss: float):
        """Save minimal state to resume/serve the model later."""
        payload = {
            "cfg": asdict(self.cfg),
            "mapper": {
                "sr": self.extractor.mapper.sr,
                "n_fft": self.extractor.mapper.n_fft,
                "fmin": self.extractor.mapper.fmin,
                "fmax": self.extractor.mapper.fmax,
                "centers_hz": self.extractor.mapper.centers_hz.tolist(),
            },
            "model_state": self.model.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(payload, path)
        print(f"💾 Saved checkpoint: {path}")

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str, map_location: str | None = None) -> "EncoderSystem":
        """
        Load a trained EncoderSystem from a saved checkpoint (.pt).
        Returns a fully initialized EncoderSystem ready for inference or fine-tuning.
        """
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        print(f"📂 Loading checkpoint: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=map_location or "cpu")
        cfg_dict = checkpoint["cfg"]
        cfg = CodecConfig(**cfg_dict)
        system = cls(cfg)

        model_state = checkpoint.get("model_state", None)
        if model_state is not None:
            system.model.load_state_dict(model_state, strict=False)
            print("✅ Model weights loaded successfully.")

        system.model.to(system.device)
        system.model.eval()
        print(f"🚀 EncoderSystem loaded and ready on {system.device}")
        return system


    # ===============================
    # EVALUATION / ENCODING / EXPORT
    # ===============================
    def evaluate_folder(self, folder: str, max_files: int | None = None, seconds: float = 2.0) -> dict:
        """
        Simple evaluation: MSE reconstruction over random 'seconds' crops of WAVs in a folder.
        Returns dict with avg_mse and num_files.
        """
        assert os.path.isdir(folder), f"Folder not found: {folder}"
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".wav")]
        if max_files is not None:
            files = files[:max_files]
        if not files:
            return {"avg_mse": None, "num_files": 0}

        self.model.eval()
        crop = int(seconds * self.cfg.sr)
        mses = []

        with torch.no_grad():
            for path in files:
                wav, sr = torchaudio.load(path)
                wav = wav.mean(dim=0)  # mono
                if sr != self.cfg.sr:
                    wav = torchaudio.functional.resample(wav, sr, self.cfg.sr)
                if wav.numel() >= crop:
                    start = np.random.randint(0, wav.numel() - crop + 1)
                    wav = wav[start:start + crop]
                else:
                    wav = torch.nn.functional.pad(wav, (0, crop - wav.numel()))
                wav = wav.unsqueeze(0).to(self.device)

                bands = self.extractor(wav)
                x = self.normalizer(bands)
                y, *_ = self.model(x)
                mses.append(F.mse_loss(y, x).item())

        return {"avg_mse": float(np.mean(mses)), "num_files": len(files)}

    def encode_folder(self, input_folder: str, output_folder: str) -> None:
        """
        Extract (third-octave) features and VQ indices for each WAV in a folder; save .pt with {'idx','z_q'}.
        """
        assert os.path.isdir(input_folder), f"Input folder not found: {input_folder}"
        os.makedirs(output_folder, exist_ok=True)
        self.model.eval()

        with torch.no_grad():
            for f in os.listdir(input_folder):
                if not f.lower().endswith(".wav"):
                    continue
                path = os.path.join(input_folder, f)
                wav, sr = torchaudio.load(path)
                wav = wav.mean(dim=0)
                if sr != self.cfg.sr:
                    wav = torchaudio.functional.resample(wav, sr, self.cfg.sr)
                wav = wav.unsqueeze(0).to(self.device)

                bands = self.extractor(wav)
                x = self.normalizer(bands)
                _, _, z_q, idx, _ = self.model(x)

                out_name = os.path.splitext(f)[0] + "_codes.pt"
                torch.save({"idx": idx.cpu(), "z_q": z_q.cpu()}, os.path.join(output_folder, out_name))

        print(f"✅ Encoded features saved to: {output_folder}")

    def export_torchscript(self, ts_path: str) -> str:
        """
        Export only the VQ autoencoder (encoder+VQ+decoder) to TorchScript for deployment.
        (FeatureExtractor is not exported here for simplicity.)
        """
        self.model.eval()
        scripted = torch.jit.script(self.model.cpu())
        scripted.save(ts_path)
        self.model.to(self.device).train()
        return ts_path
    # ===============================
    # CODEBOOK DIAGNOSTICS
    # ===============================
    def print_codebook_stats(self, idx: torch.Tensor) -> None:
        """
        Print per-group statistics about codebook usage:
        - number of active codewords
        - top-5 most used indices (by frequency)
        """
        B, T, G = idx.shape
        K = self.cfg.codebook_size
        print("\n📊 Codebook diagnostics:")

        for g in range(G):
            inds = idx[:, :, g].detach().reshape(-1).cpu().numpy()
            hist, _ = np.histogram(inds, bins=np.arange(K + 1), density=False)
            used = np.count_nonzero(hist)
            total = len(hist)
            usage_ratio = 100.0 * used / total

            # top-5 most used codewords
            top5_idx = np.argsort(hist)[-5:][::-1]
            top5_freq = hist[top5_idx]

            print(f"  🧩 Group {g+1}/{G}: {used}/{total} active ({usage_ratio:.1f}%)")
            print(f"     Top-5 indices: {list(top5_idx)} → counts: {list(top5_freq)}")
