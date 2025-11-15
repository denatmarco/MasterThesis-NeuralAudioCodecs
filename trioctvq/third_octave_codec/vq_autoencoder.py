"""
third_octave_codec/vq_autoencoder.py
Vector-quantized autoencoder for third-octave features.
- Encoder: time-distributed MLP over last (band) dimension
- VQ: grouped codebooks, frame-wise (Option A)
- Decoder: time-distributed MLP to reconstruct band energies
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Encoder / Decoder
# -----------------------------
class TimeDistributedMLP(nn.Module):
    """
    Applies an MLP to the last dimension of a (B, T, D_in) tensor.
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_in)
        B, T, D = x.shape
        y = self.net(x.reshape(B * T, D))
        return y.view(B, T, -1)  # (B, T, D_out)


class BandsEncoder(nn.Module):
    def __init__(self, num_bands: int, hidden: int, latent_dim: int):
        super().__init__()
        self.mlp = TimeDistributedMLP(num_bands, hidden, latent_dim)

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        # bands: (B, T, num_bands)
        return self.mlp(bands)  # (B, T, latent_dim)


class BandsDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, num_bands: int):
        super().__init__()
        self.mlp = TimeDistributedMLP(latent_dim, hidden, num_bands)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: (B, T, latent_dim)
        return self.mlp(z_q)  # (B, T, num_bands)


# -----------------------------
# Grouped Vector Quantizer
# -----------------------------
class GroupedVectorQuantizer(nn.Module):
    """
    Grouped VQ with G codebooks.
    - Input z_e: (B, T, D) or (B, D)
    - D must be divisible by num_groups (group_dim = D // G)
    - Each group g uses an nn.Embedding(K, group_dim) as codebook.
    - Frame-wise quantization (Option A): flatten time, quantize, then reshape back.
    """
    def __init__(self, latent_dim: int, num_groups: int, codebook_size: int, beta: float = 0.25):
        super().__init__()
        assert latent_dim % num_groups == 0, "latent_dim must be divisible by num_groups"
        self.latent_dim = latent_dim
        self.num_groups = num_groups
        self.group_dim = latent_dim // num_groups
        self.codebook_size = codebook_size
        self.beta = beta

        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, self.group_dim) for _ in range(num_groups)
        ])
        for emb in self.codebooks:
            nn.init.normal_(emb.weight, mean=0.0, std=0.1)

    # ✅ CORRETTO: indentato dentro la classe
    def forward(self, z_e: torch.Tensor):
        """
        Quantizes the encoder output using grouped codebooks.
        """
        # === Handle shape ===
        if z_e.ndim == 2:
            B, D = z_e.shape
            T = 1
            z_e_flat = z_e
        elif z_e.ndim == 3:
            B, T, D = z_e.shape
            z_e_flat = z_e.reshape(B * T, D)
        else:
            raise ValueError(f"Unexpected z_e shape {z_e.shape}")

        assert D == self.num_groups * self.group_dim, \
            f"Expected latent dim {self.num_groups*self.group_dim}, got {D}"

        parts = z_e_flat.view(-1, self.num_groups, self.group_dim)
        zq_parts, all_idx = [], []
        codebook_loss, commit_loss = 0.0, 0.0

        # === Data-dependent initialization (only first call) ===
        if not getattr(self, "_cb_init_done", False):
            with torch.no_grad():
                for g in range(self.num_groups):
                    z = parts[:, g, :]
                    N = z.size(0)
                    K = self.codebooks[g].num_embeddings
                    take = min(N, K)
                    idx_rand = torch.randperm(N, device=z.device)[:take]
                    self.codebooks[g].weight[:take].copy_(z[idx_rand])
            self._cb_init_done = True

        # === Quantize each group independently ===
        for g in range(self.num_groups):
            z = parts[:, g, :]
            cb = self.codebooks[g].weight

            z_norm = (z ** 2).sum(dim=1, keepdim=True)
            cb_norm = (cb ** 2).sum(dim=1).unsqueeze(0)
            dist = z_norm - 2 * z @ cb.t() + cb_norm

            idx = dist.argmin(dim=1)
            z_q = F.embedding(idx, cb)

            # Straight-Through Estimator
            z_q_st = z + (z_q - z).detach()

            codebook_loss += (z_q.detach() - z).pow(2).mean()
            commit_loss   += (z_q - z.detach()).pow(2).mean()

            zq_parts.append(z_q_st)
            all_idx.append(idx)

        z_q_flat = torch.cat(zq_parts, dim=1)
        idx = torch.stack(all_idx, dim=1)

        if z_e.ndim == 3:
            z_q = z_q_flat.view(B, T, D)
            idx = idx.view(B, T, self.num_groups)
        else:
            z_q = z_q_flat.view(B, D)
            idx = idx.view(B, self.num_groups)

        vq_loss = codebook_loss + commit_loss
        return z_q, vq_loss, idx


# -----------------------------
# Full Model
# -----------------------------
class BandsVQAutoencoder(nn.Module):
    def __init__(self, num_bands: int, latent_dim: int, hidden: int, num_groups: int, codebook_size: int, beta: float):
        super().__init__()
        self.encoder = BandsEncoder(num_bands=num_bands, hidden=hidden, latent_dim=latent_dim)
        self.vq = GroupedVectorQuantizer(latent_dim=latent_dim, num_groups=num_groups, codebook_size=codebook_size, beta=beta)
        self.decoder = BandsDecoder(latent_dim=latent_dim, hidden=hidden, num_bands=num_bands)
        self.beta = beta  # ✅ aggiunto

    def forward(self, bands: torch.Tensor, epoch: int | None = None):
        """
        Forward pass through the VQ autoencoder.
        """
        # 1. Encode
        z_e = self.encoder(bands)

        # 2. Vector Quantization
        z_q, vq_loss_raw, idx = self.vq(z_e)

        # 3. Beta annealing
        if epoch is not None:
            # warm-up progressivo: 5 epoche
            beta_eff = min(self.beta, self.beta * (epoch / 5.0))
        else:
            beta_eff = self.beta

        # 4. Scale VQ loss
        vq_loss = beta_eff * vq_loss_raw

        # 5. Decode
        bands_hat = self.decoder(z_q)

        return bands_hat, z_e, z_q, idx, vq_loss

