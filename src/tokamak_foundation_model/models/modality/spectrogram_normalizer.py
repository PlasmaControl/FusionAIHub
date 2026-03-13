"""Learned spectrogram normalizer and normalized autoencoder wrapper.

Amplifies SNR by applying learned gamma correction and subtracting a
smoothed per-frequency background.  Designed to be composed with any
spectrogram autoencoder so the model doesn't waste capacity learning
the stationary background.

Expects log-preprocessed input (always non-negative), so x**gamma is safe.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder


class SpectrogramNormalizer(nn.Module):
    """Learned normalizer: gamma correction + smoothed mean/std removal.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    n_freq_bins : int
        Number of frequency bins (F dimension).
    smooth_kernel_size : int
        Kernel size for depthwise smoothing conv along frequency axis.
    eps : float
        Epsilon for numerical stability in division.
    """

    def __init__(
        self,
        n_channels: int,
        n_freq_bins: int,
        smooth_kernel_size: int = 7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_freq_bins = n_freq_bins
        self.eps = eps

        # Gamma: gamma = 1 + softplus(log_gamma).
        # Init log_gamma = -5 → softplus(-5) ≈ 0.007 → gamma ≈ 1.0 (near-identity).
        # Starting near identity keeps data in its natural range so the CNN
        # can learn; gamma then increases gradually via gradient descent.
        self.log_gamma = nn.Parameter(torch.full((1,), -5.0))

        # Per-channel, per-frequency mean and log-std
        # Shape: (1, C, F, 1) — broadcast over batch and time
        self.raw_freq_mean = nn.Parameter(torch.zeros(1, n_channels, n_freq_bins, 1))
        self.raw_freq_log_std = nn.Parameter(torch.zeros(1, n_channels, n_freq_bins, 1))

        # Depthwise smoothing conv along frequency axis
        # Input: (B*T, C, F) → depthwise Conv1d → (B*T, C, F)
        self.smooth_conv = nn.Conv1d(
            n_channels, n_channels,
            kernel_size=smooth_kernel_size,
            padding=smooth_kernel_size // 2,
            groups=n_channels,
            bias=False,
        )
        # Initialize to Gaussian kernel
        with torch.no_grad():
            sigma = smooth_kernel_size / 4.0
            k = torch.arange(smooth_kernel_size, dtype=torch.float32)
            k = k - smooth_kernel_size // 2
            kernel = torch.exp(-0.5 * (k / sigma) ** 2)
            kernel = kernel / kernel.sum()
            # (out_channels, in_channels/groups, kW) = (C, 1, K)
            self.smooth_conv.weight.copy_(
                kernel.unsqueeze(0).unsqueeze(0).expand(n_channels, 1, -1)
            )

    @property
    def gamma(self) -> Tensor:
        return 1.0 + F.softplus(self.log_gamma)

    def _smooth_params(self) -> tuple[Tensor, Tensor]:
        """Smooth raw_freq_mean and raw_freq_std via depthwise conv.

        Returns smoothed_mean (1, C, F, 1) and smoothed_std (1, C, F, 1).
        """
        # raw_freq_mean: (1, C, F, 1) → squeeze time → (1, C, F)
        mean_in = self.raw_freq_mean.squeeze(-1)  # (1, C, F)
        smoothed_mean = self.smooth_conv(mean_in).unsqueeze(-1)  # (1, C, F, 1)

        log_std_in = self.raw_freq_log_std.squeeze(-1)  # (1, C, F)
        smoothed_log_std = self.smooth_conv(log_std_in).unsqueeze(-1)  # (1, C, F, 1)
        smoothed_std = smoothed_log_std.exp()  # init exp(0) = 1.0

        return smoothed_mean, smoothed_std

    def normalize(self, x: Tensor) -> Tensor:
        """Apply learned normalization: gamma correct → subtract mean → divide std."""
        gamma = self.gamma
        x_g = x ** gamma
        smoothed_mean, smoothed_std = self._smooth_params()
        return (x_g - smoothed_mean) / (smoothed_std + self.eps)

    def denormalize(self, x: Tensor) -> Tensor:
        """Invert normalization: multiply std → add mean → inverse gamma.

        Clamps to non-negative before the fractional power since the original
        data is always >= 0 (log-power spectrogram).  Without this, an
        untrained decoder producing negative values would yield NaN.
        """
        gamma = self.gamma
        smoothed_mean, smoothed_std = self._smooth_params()
        x = x * (smoothed_std + self.eps) + smoothed_mean
        x = x.clamp(min=0.0)
        return x ** (1.0 / gamma)


class NormalizedSpectrogramAutoEncoder(ModalityAutoEncoder):
    """Wrapper that applies learned normalization around any spectrogram AE.

    Parameters
    ----------
    inner : ModalityAutoEncoder
        The wrapped spectrogram autoencoder.
    n_channels : int
        Number of spectrogram channels.
    n_freq_bins : int
        Number of frequency bins.
    smooth_kernel_size : int
        Kernel size for the normalizer's smoothing conv.
    """

    def __init__(
        self,
        inner: ModalityAutoEncoder,
        n_channels: int,
        n_freq_bins: int,
        smooth_kernel_size: int = 7,
    ) -> None:
        super().__init__(n_channels, inner.d_model, inner.n_tokens)
        self.normalizer = SpectrogramNormalizer(
            n_channels, n_freq_bins, smooth_kernel_size
        )
        self.inner = inner
        # Expose encoder for test_encoder_output_is_finite
        self.encoder = inner.encoder

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        x_norm = self.normalizer.normalize(x)
        output = self.inner(x_norm)

        if isinstance(output, tuple):
            reconstructed, *rest = output
            return (self.normalizer.denormalize(reconstructed), *rest)
        return self.normalizer.denormalize(output)
