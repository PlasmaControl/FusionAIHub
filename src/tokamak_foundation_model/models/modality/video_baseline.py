"""Video baseline modality autoencoder implementation."""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Assuming base classes are available in your project structure
# from .base import ModalityEncoder, ModalityDecoder

class VideoBaselineEncoder(nn.Module): # Inherit from ModalityEncoder in your repo
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 512,
        n_tokens: int = 256,
        t_chunk: int = 25,
        img_size: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens
        
        # Calculate a grid for Adaptive pooling to match n_tokens
        # We try to keep temporal/spatial ratios somewhat balanced
        t_grid = max(2, int(math.pow(n_tokens, 1/3) * (t_chunk/img_size)))
        remaining = n_tokens / t_grid
        hw_grid = int(math.sqrt(remaining))
        self.token_grid = (t_grid, hw_grid, hw_grid)

        # Architecture (< 2M params total)
        self.enc1 = nn.Conv3d(n_channels, 32, kernel_size=3, stride=(1,2,2), padding=1)
        self.enc2 = nn.Conv3d(32, 64, kernel_size=3, stride=(2,2,2), padding=1)
        self.enc3 = nn.Conv3d(64, 128, kernel_size=3, stride=(2,2,2), padding=1)
        self.enc4 = nn.Conv3d(128, 128, kernel_size=3, stride=(2,2,2), padding=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d(self.token_grid)
        self.to_latent = nn.Conv3d(128, d_model, kernel_size=1)
        
        # Required for the Decoder to initialize properly in the template
        self._enc_shape = (d_model, *self.token_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, T, H, W)
        # Add channel dim: (B, 1, T, H, W)
        x = x.unsqueeze(1)
            
        x = F.leaky_relu(self.enc1(x), 0.2)
        x = F.leaky_relu(self.enc2(x), 0.2)
        x = F.leaky_relu(self.enc3(x), 0.2)
        x = F.leaky_relu(self.enc4(x), 0.2)
        
        x = self.adaptive_pool(x)
        tokens = self.to_latent(x) # (B, d_model, T_g, H_g, W_g)
        
        # Flatten to (B, n_tokens, d_model)
        B, C, T, H, W = tokens.shape
        tokens = tokens.view(B, C, -1).transpose(1, 2)
        return tokens


class VideoBaselineDecoder(nn.Module): # Inherit from ModalityDecoder in your repo
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 512,
        n_tokens: int = 256,
        t_chunk: int = 25,
        img_size: int = 128,
        enc_shape: Tuple[int, int, int, int, int] = (512, 4, 8, 8),
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.t_chunk = t_chunk
        self.img_size = img_size
        self.enc_shape = enc_shape # (d_model, T_grid, H_grid, W_grid)

        self.from_latent = nn.Conv3d(d_model, 128, kernel_size=1)
        self.dec1 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.dec2 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.dec3 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.dec4 = nn.Conv3d(32, n_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, output_shape=None) -> torch.Tensor:
        # z: (B, n_tokens, d_model)
        B = z.shape[0]
        z = z.transpose(1, 2).view(B, *self.enc_shape)
        
        x = F.leaky_relu(self.from_latent(z), 0.2)
        
        # Hierarchical upsampling to match target t_chunk and img_size
        x = F.interpolate(x, size=(max(1, self.t_chunk//4), self.img_size//8, self.img_size//8), mode='trilinear')
        x = F.leaky_relu(self.dec1(x), 0.2)
        
        x = F.interpolate(x, size=(max(1, self.t_chunk//2), self.img_size//4, self.img_size//4), mode='trilinear')
        x = F.leaky_relu(self.dec2(x), 0.2)
        
        x = F.interpolate(x, size=(self.t_chunk, self.img_size//2, self.img_size//2), mode='trilinear')
        x = F.leaky_relu(self.dec3(x), 0.2)
        
        x = F.interpolate(x, size=(self.t_chunk, self.img_size, self.img_size), mode='trilinear')
        x = torch.sigmoid(self.dec4(x))
        
        # Return as (B, T, H, W) for grayscale as requested in template
        return x.squeeze(1) 


class VideoBaselineAutoEncoder(nn.Module):
    def __init__(
        self,
        n_tokens: int = 256,
        t_chunk: int = 25,
        img_size: int = 128,
        token_dim: int = 512,
        d_model: int | None = None,
        n_channels: int = 1,
        **kwargs,
    ):
        super().__init__()
        # Accept d_model as alias for token_dim (for build_model compatibility)
        effective_d_model = d_model if d_model is not None else token_dim
        self.encoder = VideoBaselineEncoder(
            n_channels=n_channels,
            d_model=effective_d_model,
            n_tokens=n_tokens,
            t_chunk=t_chunk,
            img_size=img_size,
        )
        self.decoder = VideoBaselineDecoder(
            n_channels=n_channels,
            d_model=effective_d_model,
            n_tokens=n_tokens,
            t_chunk=t_chunk,
            img_size=img_size,
            enc_shape=self.encoder._enc_shape,
        )

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(reconstructed, tokens)`` where tokens is ``(B, n_tokens, d_model)``.
        """
        tokens = self.encoder(x)  # x=[B,T,H,W]
        x_hat = self.decoder(tokens)
        return x_hat, tokens