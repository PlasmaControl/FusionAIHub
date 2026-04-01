"""Channel-AST autoencoder + PatchGAN discriminator with R3GAN stabilization.

Adds a lightweight PatchGAN discriminator to the existing Channel-AST
autoencoder (continuous, no FSQ).  The adversarial loss directly penalizes
blurry reconstructions — the discriminator learns to detect smoothness and
the generator is trained to produce sharp, realistic spectrograms.

Training is stabilized by **R3GAN** (Regularized Relativistic GAN):
  - RpGAN loss: relativistic paired comparisons (D scores fake *relative to*
    real, preventing mode collapse)
  - R1 + R2 gradient penalties: applied every iteration on real and fake data
    respectively, enforcing smooth discriminator landscapes
  - No normalization in D (gradient penalties provide regularization)

Architecture
------------
Generator: existing ``SpectrogramChannelASTFSQAutoEncoder`` with ``fsq_levels=[]``
  (B, C, F, T) → encoder → z → decoder → x̂ (B, C, F, T)

PatchGAN Discriminator (per-channel):
  (B*C, 1, F, T) → Conv2d stack → (B*C, 1, H', W') grid of logits

Loss
----
Generator:  L1(x̂, x) + λ_adv * softplus(D(real) - D(fake)).mean()
Discriminator: softplus(D(fake) - D(real)).mean() + R1 + R2

Return contract
---------------
Training : reconstructed — shape (B, C, F, T) matching input
           (same as Channel-AST no-FSQ; discriminator is called by trainer)
Eval     : reconstructed — shape (B, C, F, T)

References
----------
- Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks
  (Isola et al., CVPR 2017) — PatchGAN discriminator
- R3GAN: Back to Basics (Tschannen et al., ICLR 2025) — RpGAN + R1/R2
"""

import torch
import torch.nn as nn
from torch import Tensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder
from tokamak_foundation_model.models.modality.spectrogram_channel_ast_fsq import (
    SpectrogramChannelASTFSQAutoEncoder,
)


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

class _PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for 2D spectrograms (Pix2Pix style).

    Classifies overlapping local patches as real/fake via a fully-convolutional
    architecture.  Operates on single-channel input — the caller reshapes
    (B, C, F, T) → (B*C, 1, F, T) for per-channel discrimination.

    No normalization layers (R3GAN: gradient penalties provide regularization).
    Weight init: Gaussian N(0, 0.02).

    Parameters
    ----------
    channels : list[int]
        Hidden channel widths for stride-2 layers (default [64, 128, 256, 512]).
    """

    def __init__(self, channels: list[int] | None = None) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]

        layers: list[nn.Module] = []
        in_ch = 1
        # Stride-2 layers (downsample)
        for out_ch in channels[:-1]:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch

        # Stride-1 layers (refine without downsampling)
        layers.extend([
            nn.Conv2d(in_ch, channels[-1], 4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # Final 1-channel output (patch logits)
        layers.append(nn.Conv2d(channels[-1], 1, 4, stride=1, padding=1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """(B*C, 1, F, T) → (B*C, 1, H', W') grid of logits."""
        return self.net(x)


# ---------------------------------------------------------------------------
# GAN Wrapper Autoencoder
# ---------------------------------------------------------------------------

class SpectrogramChannelASTGANAutoEncoder(ModalityAutoEncoder):
    """Channel-AST autoencoder + PatchGAN discriminator.

    The generator is the existing ``SpectrogramChannelASTFSQAutoEncoder``
    in continuous mode (``fsq_levels=[]``).  The discriminator is a
    lightweight PatchGAN that the ``GANUnimodalTrainer`` uses for
    adversarial training — it is NOT called during ``forward()``.

    Parameters
    ----------
    n_channels, d_model, n_tokens : int
        Standard ModalityAutoEncoder interface.
    freq_bins, frame_width : int
        Spectrogram dimensions.
    n_enc_layers, n_dec_layers : int
        Encoder/decoder depth (default 4).
    n_heads : int
        Attention heads (default 4).
    dropout : float
        Dropout rate (default 0.1).
    max_channels, max_time_frames : int
        Positional embedding table capacities.
    time_conv_kernel : int
        Temporal ConvNeXt kernel size (default 7).
    d_channels : list[int] or None
        Discriminator channel widths (default [64, 128, 256, 512]).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 0,
        *,
        freq_bins: int = 512,
        frame_width: int = 16,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_channels: int = 64,
        max_time_frames: int = 2048,
        time_conv_kernel: int = 7,
        d_channels: list[int] | None = None,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)

        # Generator: Channel-AST autoencoder (continuous, no FSQ)
        self.autoencoder = SpectrogramChannelASTFSQAutoEncoder(
            n_channels=n_channels,
            d_model=d_model,
            n_tokens=n_tokens,
            freq_bins=freq_bins,
            frame_width=frame_width,
            n_enc_layers=n_enc_layers,
            n_dec_layers=n_dec_layers,
            n_heads=n_heads,
            dropout=dropout,
            fsq_levels=[],
            max_channels=max_channels,
            max_time_frames=max_time_frames,
            time_conv_kernel=time_conv_kernel,
        )

        # Expose encoder for tests and fusion pipeline
        self.encoder = self.autoencoder.encoder

        # PatchGAN Discriminator (called by trainer, not in forward)
        self.discriminator = _PatchDiscriminator(channels=d_channels)

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, F, T) → (B, C, F, T) reconstruction.

        Delegates to the inner autoencoder. The discriminator is NOT called
        here — it is used by GANUnimodalTrainer during training.
        """
        return self.autoencoder(x)
