"""Masked Autoencoder for tokamak spectrogram diagnostics.

Uses the convolutional SpecEncoder / SpecDecoder from aps_model.py to build a
pixel-space masked autoencoder (He et al., 2022). During training the model
returns (reconstructed, mask); during eval it returns just reconstructed.

Preprocessing note
------------------
The data pipeline applies log10(x + 1) (and optionally standardisation) before
reaching this model. SpecEncoder's BatchNorm2d layers handle scale differences
across signals. MHR/CO2 outputs are not zero-centred; for best training results,
switch their preprocessing to "log_standardize" in the dataset config.

The mask_token is a learnable scalar parameter (init 0). This lets the model
distinguish masked regions from genuinely quiet / DC-free regions where the
log-spectrogram is also 0.
"""

import torch
import torch.nn as nn

from tokamak_foundation_model.models.aps_model import SpecDecoder, SpecEncoder
from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder


class SpectrogramMAEAutoEncoder(ModalityAutoEncoder):
    """Masked Autoencoder for multichannel spectrogram signals.

    Divides the input spectrogram into non-overlapping (patch_h × patch_w) pixel
    patches, randomly masks mask_ratio of them, encodes the masked input with
    SpecEncoder, and decodes back to the original shape with SpecDecoder.

    Parameters
    ----------
    n_channels : int
        Number of input spectrogram channels (e.g. 4 for CO2, 48 for ECE).
    d_model : int
        Feature dimension at the bottleneck (encoder output channels).
    n_tokens : int
        Unused; kept for interface compatibility with ModalityAutoEncoder.
    mask_ratio : float
        Fraction of patches to mask during training (default 0.75).
    patch_h, patch_w : int
        Patch height and width in pixels (default 4 × 4).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 64,
        n_tokens: int = 0,
        *,
        mask_ratio: float = 0.75,
        patch_h: int = 4,
        patch_w: int = 4,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)
        self.mask_ratio = mask_ratio
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.encoder = SpecEncoder(n_channels, d_model)
        self.decoder = SpecDecoder(d_model, n_channels)

        # Learnable fill value for masked patches (scalar, init 0).
        self.mask_token = nn.Parameter(torch.zeros(1))

    def _mask_patches(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random patch masking.

        Parameters
        ----------
        x : Tensor of shape (B, C, F, T)

        Returns
        -------
        masked_x : Tensor (B, C, F, T) — input with masked patches replaced by mask_token.
        mask     : BoolTensor (B, 1, F, T) — True where patches were masked.
        """
        B, C, F, T = x.shape
        ph, pw = self.patch_h, self.patch_w

        # Number of complete patches
        n_h = F // ph
        n_w = T // pw
        n_patches = n_h * n_w
        n_masked = max(1, int(n_patches * self.mask_ratio))

        # Build per-sample patch mask: (B, n_patches), True = masked
        mask_patches = torch.zeros(B, n_patches, dtype=torch.bool, device=x.device)
        for i in range(B):
            idx = torch.randperm(n_patches, device=x.device)[:n_masked]
            mask_patches[i, idx] = True

        # Upsample patch mask to pixel space: (B, 1, n_h, n_w) → (B, 1, n_h*ph, n_w*pw)
        mask_pixel = (
            mask_patches
            .reshape(B, 1, n_h, n_w)
            .repeat_interleave(ph, dim=2)
            .repeat_interleave(pw, dim=3)
        )
        # Crop to exact (F, T) in case F or T is not divisible by patch size
        mask_pixel = mask_pixel[:, :, :F, :T]

        # Fill masked locations with learnable token value
        masked_x = x.masked_fill(mask_pixel, 0.0) + mask_pixel.float() * self.mask_token

        return masked_x, mask_pixel

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass.

        During training returns (reconstructed, mask) so the trainer can
        compute loss only on masked patches.  During eval returns only
        reconstructed so loss is computed on the full image.
        """
        B, C, F, T = x.shape

        masked_x, mask = self._mask_patches(x)

        z = self.encoder(masked_x)       # (B, d_model, F', T)
        reconstructed = self.decoder(z)  # (B, n_channels, F'', T)

        # Crop to original spatial dimensions (decoder may add a few extra rows)
        reconstructed = reconstructed[..., :F, :T]

        if self.training:
            return reconstructed, mask
        return reconstructed
