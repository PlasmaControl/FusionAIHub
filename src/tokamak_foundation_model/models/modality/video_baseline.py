"""Video baseline modality autoencoder.

This module is refactored to follow the same structural template as other modality
baselines (see :mod:`fast_time_series_baseline.py`) while preserving the exact
architecture/parameters defined in the original `video_baseline.py`.

Key conventions:
- Encoder inherits :class:`~tokamak_foundation_model.models.modality.base.ModalityEncoder`
  and returns tokens shaped (B, n_tokens, d_model).
- Decoder inherits :class:`~tokamak_foundation_model.models.modality.base.ModalityDecoder`
  and reconstructs an output shaped (B, T, H, W) for grayscale video.
- Autoencoder composes encoder/decoder and returns (x_hat, tokens) for training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModalityEncoder, ModalityDecoder


class VideoBaselineEncoder(ModalityEncoder):
    """3D CNN encoder producing (B, n_tokens, d_model) tokens.

    Architecture is preserved from the original implementation:
    Conv3d(stride=2) stack -> flatten -> Linear -> reshape to (B, n_tokens, d_model).

    Parameters
    ----------
    n_channels:
        Number of input channels. Original model assumes grayscale=1.
    d_model:
        Token embedding dimension. Original model uses 512.
    n_tokens:
        Number of tokens, returned as the middle dimension of the latent (N x 512).
    t_chunk:
        Number of frames in the clip (T).
    img_size:
        Spatial size (H=W) used to infer the encoder output shape.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 512,
        n_tokens: int = 8,
        t_chunk: int = 25,
        img_size: int = 256,
    ):
        super().__init__(n_channels=n_channels, d_model=d_model, n_tokens=n_tokens)

        # Preserve original conv stack (stride=2 in all dims).
        self.enc = nn.Sequential(
            nn.Conv3d(n_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        # Infer encoder output shape for decoder reshaping (preserved behavior).
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, t_chunk, img_size, img_size)
            h = self.enc(dummy)
            self._enc_shape: Tuple[int, int, int, int, int] = tuple(h.shape)  # (1,C0,T0,H0,W0)
            flat_dim = h.flatten(1).shape[1]

        self.latent_dim = n_tokens * d_model
        self.fc = nn.Linear(flat_dim, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B,T,H,W) or (B,C,T,H,W) like other modalities.
        if x.ndim == 4:
            x = x.unsqueeze(1)
        elif x.ndim != 5:
            raise ValueError(f"Expected x with 4 or 5 dims, got {tuple(x.shape)}")

        if x.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {x.shape[1]}")
        h = self.enc(x)
        z_vec = self.fc(h.flatten(1))  # (B, n_tokens*d_model)
        tokens = z_vec.view(x.shape[0], self.n_tokens, self.d_model)  # (B, n_tokens, d_model)
        return tokens


class VideoBaselineDecoder(ModalityDecoder):
    """3D CNN decoder reconstructing clips from tokens.

    Architecture is preserved from the original implementation:
    Linear -> reshape to encoder feature volume -> ConvTranspose3d stack -> interpolate -> sigmoid.

    Parameters
    ----------
    n_channels:
        Number of output channels (grayscale=1).
    d_model:
        Token embedding dimension (512).
    n_tokens:
        Number of tokens in the latent.
    t_chunk:
        Target time length (T).
    img_size:
        Target spatial size (H=W).
    enc_shape:
        Shape tuple from encoder forward on a dummy input (1,C0,T0,H0,W0).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 512,
        n_tokens: int = 8,
        t_chunk: int = 25,
        img_size: int = 256,
        enc_shape: Tuple[int, int, int, int, int] = (1, 256, 1, 8, 8),
    ):
        super().__init__(n_channels=n_channels, d_model=d_model)
        self.n_tokens = n_tokens
        self.t_chunk = t_chunk
        self.img_size = img_size
        self.latent_dim = n_tokens * d_model

        _, C0, T0, H0, W0 = enc_shape
        self.C0, self.T0, self.H0, self.W0 = C0, T0, H0, W0

        self.fc = nn.Linear(self.latent_dim, C0 * T0 * H0 * W0)

        # Preserve original deconv stack.
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(C0, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, n_channels, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, z: torch.Tensor, output_shape=None) -> torch.Tensor:
        # z is expected (B, n_tokens, d_model)
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape (B,n_tokens,d_model), got {tuple(z.shape)}")

        B = z.shape[0]
        z_vec = z.reshape(B, self.latent_dim)  # (B, n_tokens*d_model) — preserves original mapping

        x = self.fc(z_vec).view(B, self.C0, self.T0, self.H0, self.W0)  # (B,C0,T0,H0,W0)
        x = self.dec(x)  # (B,C,T',H',W')

        # Determine target output size.
        if output_shape is None:
            T, H, W = self.t_chunk, self.img_size, self.img_size
        else:
            # output_shape can be (T,H,W) or (C,T,H,W)
            if len(output_shape) == 3:
                T, H, W = output_shape
            elif len(output_shape) == 4:
                _, T, H, W = output_shape
            else:
                raise ValueError("output_shape must be (T,H,W) or (C,T,H,W)")

        x = F.interpolate(x, size=(T, H, W), mode="trilinear", align_corners=False)
        x = torch.sigmoid(x)

        # Repo convention for grayscale: (B,T,H,W)
        if x.shape[1] == 1:
            return x.squeeze(1)
        return x


class VideoBaselineAutoEncoder(nn.Module):
    """Autoencoder wrapper that returns reconstructions and tokens.

    Forward returns
    --------------
    x_hat : torch.Tensor
        Reconstructed clip (B, T, H, W) for grayscale.
    tokens : torch.Tensor
        Latent tokens (B, n_tokens, d_model).
    """
    def __init__(
        self,
        n_tokens: int,
        t_chunk: int = 25,
        img_size: int = 256,
        token_dim: int = 512,
        n_channels: int = 1,
    ):
        super().__init__()
        self.encoder = VideoBaselineEncoder(
            n_channels=n_channels,
            d_model=token_dim,
            n_tokens=n_tokens,
            t_chunk=t_chunk,
            img_size=img_size,
        )
        self.decoder = VideoBaselineDecoder(
            n_channels=n_channels,
            d_model=token_dim,
            n_tokens=n_tokens,
            t_chunk=t_chunk,
            img_size=img_size,
            enc_shape=self.encoder._enc_shape,
        )

    def forward(self, x: torch.Tensor):
        tokens = self.encoder(x)
        x_hat = self.decoder(tokens)
        return x_hat

