"""Masked Autoencoder (MAE) implementation for self-supervised learning.

This module provides the MaskedAutoencoder class and associated utilities
for training autoencoders with various masking strategies. The MAE approach
is particularly effective for learning robust representations from partially
observed data, such as audio spectrograms with missing frequency bands or
time frames.
"""

import random
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder import BlockBasedAutoencoder


class MaskType(Enum):
    """Enumeration of available masking strategies."""

    RANDOM = "random"
    FREQUENCY = "frequency"
    TIME = "time"
    PATCH = "patch"
    MIXED = "mixed"


# Export mask types for convenience
MASK_TYPES = [mask_type.value for mask_type in MaskType]


class MaskGenerator:
    """Generates various types of masks for audio spectrograms.

    This class provides different masking strategies for Masked Autoencoder
    training, including random masking, frequency band masking, temporal
    masking, and patch-based masking.

    Parameters
    ----------
    mask_ratio : float, default=0.75
        Ratio of input to mask (0.0 to 1.0). Higher values mean more masking.
    patch_size : tuple of int, default=(8, 8)
        Size of patches for patch-based masking (height, width).
    min_mask_size : int, default=1
        Minimum size for contiguous masked regions.
    max_mask_size : int, optional
        Maximum size for contiguous masked regions. If None, no upper limit.

    Attributes
    ----------
    mask_ratio : float
        Stored mask ratio.
    patch_size : tuple of int
        Stored patch size.
    min_mask_size : int
        Stored minimum mask size.
    max_mask_size : int or None
        Stored maximum mask size.

    Examples
    --------
    >>> mask_gen = MaskGenerator(mask_ratio=0.75)
    >>> shape = (2, 80, 100, 128)  # batch, channels, time, freq
    >>> mask = mask_gen.random_mask(shape)
    >>> print(f"Mask shape: {mask.shape}, Masked ratio: "
    ...      f"{(1 - mask.mean()).item():.2f}")
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        patch_size: tuple[int, int] = (8, 8),
        min_mask_size: int = 1,
        max_mask_size: Optional[int] = None,
    ) -> None:
        """Initialize MaskGenerator.

        Parameters
        ----------
        mask_ratio : float, default=0.75
            Ratio of input to mask.
        patch_size : tuple of int, default=(8, 8)
            Size of patches for patch-based masking.
        min_mask_size : int, default=1
            Minimum size for contiguous masked regions.
        max_mask_size : int, optional
            Maximum size for contiguous masked regions.
        """
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(
                f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}"
            )

        if len(patch_size) != 2 or any(s <= 0 for s in patch_size):
            raise ValueError(
                f"patch_size must be a tuple of two positive integers, got {patch_size}"
            )

        if min_mask_size <= 0:
            raise ValueError(f"min_mask_size must be positive, got {min_mask_size}")

        if max_mask_size is not None and max_mask_size < min_mask_size:
            raise ValueError(
                f"max_mask_size must be >= min_mask_size, got {max_mask_size}"
            )

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.min_mask_size = min_mask_size
        self.max_mask_size = max_mask_size

    def random_mask(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Generate random masking of individual time-frequency bins.

        Parameters
        ----------
        shape : tuple of int
            Input tensor shape (batch_size, channels, time_steps, freq_bins).

        Returns
        -------
        torch.Tensor
            Binary mask tensor where 1 = keep, 0 = mask.
        """
        batch_size, channels, time_steps, freq_bins = shape
        mask = torch.rand(batch_size, 1, time_steps, freq_bins) > self.mask_ratio
        return mask.float()

    def frequency_band_mask(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Generate frequency band masking.

        Masks contiguous frequency bands, which is common in audio augmentation
        and simulates frequency-selective interference.

        Parameters
        ----------
        shape : tuple of int
            Input tensor shape (batch_size, channels, time_steps, freq_bins).

        Returns
        -------
        torch.Tensor
            Binary mask tensor where 1 = keep, 0 = mask.
        """
        batch_size, channels, time_steps, freq_bins = shape
        mask = torch.ones(batch_size, 1, time_steps, freq_bins)

        for b in range(batch_size):
            # Calculate number of frequency bins to mask
            num_bins_to_mask = int(freq_bins * self.mask_ratio)

            if num_bins_to_mask > 0:
                # Choose number of bands to create
                max_bands = min(num_bins_to_mask, freq_bins // self.min_mask_size)
                num_bands = random.randint(1, max(1, max_bands))

                bins_per_band = num_bins_to_mask // num_bands
                remaining_bins = num_bins_to_mask % num_bands

                masked_bins = 0
                for _ in range(num_bands):
                    if masked_bins >= num_bins_to_mask:
                        break

                    # Size of this band
                    band_size = bins_per_band
                    if remaining_bins > 0:
                        band_size += 1
                        remaining_bins -= 1

                    # Ensure we don't exceed limits
                    if self.max_mask_size is not None:
                        band_size = min(band_size, self.max_mask_size)
                    band_size = max(band_size, self.min_mask_size)
                    band_size = min(band_size, freq_bins - masked_bins)

                    if band_size <= 0:
                        break

                    # Choose random start position
                    max_start = freq_bins - band_size
                    if max_start >= 0:
                        start_freq = random.randint(0, max_start)
                        mask[b, :, :, start_freq : start_freq + band_size] = 0
                        masked_bins += band_size

        return mask

    def time_frame_mask(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Generate temporal frame masking.

        Masks contiguous time frames, simulating temporal dropouts or
        transmission errors in audio signals.

        Parameters
        ----------
        shape : tuple of int
            Input tensor shape (batch_size, channels, time_steps, freq_bins).

        Returns
        -------
        torch.Tensor
            Binary mask tensor where 1 = keep, 0 = mask.
        """
        batch_size, channels, time_steps, freq_bins = shape
        mask = torch.ones(batch_size, 1, time_steps, freq_bins)

        for b in range(batch_size):
            # Calculate number of time frames to mask
            num_frames_to_mask = int(time_steps * self.mask_ratio)

            if num_frames_to_mask > 0:
                # Choose number of segments to create
                max_segments = min(num_frames_to_mask, time_steps // self.min_mask_size)
                num_segments = random.randint(1, max(1, max_segments))

                frames_per_segment = num_frames_to_mask // num_segments
                remaining_frames = num_frames_to_mask % num_segments

                masked_frames = 0
                for _ in range(num_segments):
                    if masked_frames >= num_frames_to_mask:
                        break

                    # Size of this segment
                    segment_size = frames_per_segment
                    if remaining_frames > 0:
                        segment_size += 1
                        remaining_frames -= 1

                    # Ensure we don't exceed limits
                    if self.max_mask_size is not None:
                        segment_size = min(segment_size, self.max_mask_size)
                    segment_size = max(segment_size, self.min_mask_size)
                    segment_size = min(segment_size, time_steps - masked_frames)

                    if segment_size <= 0:
                        break

                    # Choose random start position
                    max_start = time_steps - segment_size
                    if max_start >= 0:
                        start_time = random.randint(0, max_start)
                        mask[b, :, start_time : start_time + segment_size, :] = 0
                        masked_frames += segment_size

        return mask

    def patch_mask(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Generate patch-based masking.

        Masks rectangular patches, similar to Vision Transformer masking
        but adapted for spectrograms.

        Parameters
        ----------
        shape : tuple of int
            Input tensor shape (batch_size, channels, time_steps, freq_bins).

        Returns
        -------
        torch.Tensor
            Binary mask tensor where 1 = keep, 0 = mask.
        """
        batch_size, channels, time_steps, freq_bins = shape
        mask = torch.ones(batch_size, 1, time_steps, freq_bins)

        patch_time, patch_freq = self.patch_size

        # Calculate number of patches in each dimension
        num_patches_time = time_steps // patch_time
        num_patches_freq = freq_bins // patch_freq
        total_patches = num_patches_time * num_patches_freq

        if total_patches == 0:
            return mask

        for b in range(batch_size):
            # Number of patches to mask
            num_masked_patches = int(total_patches * self.mask_ratio)

            if num_masked_patches > 0:
                # Randomly select patches to mask
                masked_patches = random.sample(
                    range(total_patches),
                    min(num_masked_patches, total_patches),
                )

                for patch_idx in masked_patches:
                    # Convert patch index to 2D coordinates
                    patch_t = (patch_idx // num_patches_freq) * patch_time
                    patch_f = (patch_idx % num_patches_freq) * patch_freq

                    # Apply mask to patch
                    end_t = min(patch_t + patch_time, time_steps)
                    end_f = min(patch_f + patch_freq, freq_bins)

                    mask[b, :, patch_t:end_t, patch_f:end_f] = 0

        return mask

    def mixed_mask(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Generate mixed masking strategy.

        Combines multiple masking strategies for more diverse augmentation.

        Parameters
        ----------
        shape : tuple of int
            Input tensor shape (batch_size, channels, time_steps, freq_bins).

        Returns
        -------
        torch.Tensor
            Binary mask tensor where 1 = keep, 0 = mask.
        """
        batch_size = shape[0]

        # For mixed masking, use different strategies for different samples
        mask_strategies = [
            self.random_mask,
            self.frequency_band_mask,
            self.time_frame_mask,
            self.patch_mask,
        ]

        # Generate masks using different strategies
        masks = []
        for _b in range(batch_size):
            strategy = random.choice(mask_strategies)
            single_batch_shape = (1,) + shape[1:]
            single_mask = strategy(single_batch_shape)
            masks.append(single_mask)

        return torch.cat(masks, dim=0)

    def generate_mask(self, shape: tuple[int, ...], mask_type: str) -> torch.Tensor:
        """Generate mask using specified strategy.

        Parameters
        ----------
        shape : tuple of int
            Input tensor shape.
        mask_type : str
            Type of masking strategy to use.

        Returns
        -------
        torch.Tensor
            Generated mask tensor.
        """
        if mask_type == MaskType.RANDOM.value:
            return self.random_mask(shape)
        elif mask_type == MaskType.FREQUENCY.value:
            return self.frequency_band_mask(shape)
        elif mask_type == MaskType.TIME.value:
            return self.time_frame_mask(shape)
        elif mask_type == MaskType.PATCH.value:
            return self.patch_mask(shape)
        elif mask_type == MaskType.MIXED.value:
            return self.mixed_mask(shape)
        else:
            raise ValueError(
                f"Unknown mask_type: {mask_type}. Available types: {MASK_TYPES}"
            )


class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder using block-based architecture.

    This class wraps a BlockBasedAutoencoder and adds masking functionality
    for self-supervised learning. The key insight is that the model learns
    to reconstruct masked regions based on visible context, leading to
    robust representations.

    Parameters
    ----------
    autoencoder : BlockBasedAutoencoder
        The base autoencoder model to wrap.
    mask_generator : MaskGenerator
        Generator for creating masks.
    mask_token_value : float, default=0.0
        Value to use for masked regions in input.

    Attributes
    ----------
    autoencoder : BlockBasedAutoencoder
        The wrapped autoencoder.
    mask_generator : MaskGenerator
        The mask generator.
    mask_token_value : float
        Value used for masked tokens.

    Examples
    --------
    >>> autoencoder = BlockBasedAutoencoder(input_channels=80)
    >>> mask_gen = MaskGenerator(mask_ratio=0.75)
    >>> mae = MaskedAutoencoder(autoencoder, mask_gen)
    >>>
    >>> x = torch.randn(2, 80, 100, 128)
    >>> reconstructed, mask, masked_input = mae(x, mask_type='frequency')
    >>> loss = mae_loss(reconstructed, x, mask)
    """

    def __init__(
        self,
        autoencoder: BlockBasedAutoencoder,
        mask_generator: MaskGenerator,
        mask_token_value: float = 0.0,
    ) -> None:
        """Initialize MaskedAutoencoder.

        Parameters
        ----------
        autoencoder : BlockBasedAutoencoder
            The base autoencoder model.
        mask_generator : MaskGenerator
            Generator for creating masks.
        mask_token_value : float, default=0.0
            Value to use for masked regions.
        """
        super().__init__()

        self.autoencoder = autoencoder
        self.mask_generator = mask_generator
        self.mask_token_value = mask_token_value

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        mask : torch.Tensor
            Binary mask (1 = keep, 0 = mask).

        Returns
        -------
        torch.Tensor
            Masked input tensor.
        """
        # Apply mask: keep visible regions, masked regions to mask_token_value
        return x * mask + self.mask_token_value * (1 - mask)

    def forward(
        self,
        x: torch.Tensor,
        mask_type: str = "random",
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with masking.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, channels, height, width).
        mask_type : str, default='random'
            Type of masking to apply. Options: 'random', 'frequency',
            'time', 'patch', 'mixed'.
        mask : torch.Tensor, optional
            Pre-generated mask. If provided, mask_type is ignored.

        Returns
        -------
        tuple of torch.Tensor
            - reconstructed : torch.Tensor
                Full reconstruction of the input.
            - mask : torch.Tensor
                Applied mask (1 = keep, 0 = mask).
            - masked_input : torch.Tensor
                Input with mask applied.
        """
        # Generate mask if not provided
        if mask is None:
            mask = self.mask_generator.generate_mask(x.shape, mask_type)
            mask = mask.to(x.device)

        # Apply mask to input
        masked_input = self.apply_mask(x, mask)

        # Forward through autoencoder
        reconstructed, latent = self.autoencoder(masked_input)

        return reconstructed, mask, masked_input

    def encode(self, x: torch.Tensor, mask_type: str = "random") -> torch.Tensor:
        """Encode masked input to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        mask_type : str, default='random'
            Type of masking to apply.

        Returns
        -------
        torch.Tensor
            Latent representation of masked input.
        """
        mask = self.mask_generator.generate_mask(x.shape, mask_type)
        mask = mask.to(x.device)
        masked_input = self.apply_mask(x, mask)
        return self.autoencoder.encode(masked_input)

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "autoencoder_config": self.autoencoder.get_config(),
            "mask_ratio": self.mask_generator.mask_ratio,
            "patch_size": self.mask_generator.patch_size,
            "min_mask_size": self.mask_generator.min_mask_size,
            "max_mask_size": self.mask_generator.max_mask_size,
            "mask_token_value": self.mask_token_value,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MaskedAutoencoder":
        """Create MaskedAutoencoder from configuration."""
        # Recreate autoencoder
        autoencoder = BlockBasedAutoencoder.from_config(config["autoencoder_config"])

        # Recreate mask generator
        mask_generator = MaskGenerator(
            mask_ratio=config["mask_ratio"],
            patch_size=config["patch_size"],
            min_mask_size=config["min_mask_size"],
            max_mask_size=config["max_mask_size"],
        )

        return cls(
            autoencoder=autoencoder,
            mask_generator=mask_generator,
            mask_token_value=config["mask_token_value"],
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MaskedAutoencoder("
            f"mask_ratio={self.mask_generator.mask_ratio}, "
            f"autoencoder={self.autoencoder})"
        )


def mae_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mse",
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute MAE loss only on masked regions.

    This is the key insight of MAE: only compute reconstruction loss
    on the masked regions, not on the visible regions.

    Parameters
    ----------
    reconstructed : torch.Tensor
        Model output/reconstruction.
    target : torch.Tensor
        Original unmasked input.
    mask : torch.Tensor
        Binary mask tensor (1 = keep, 0 = mask).
    loss_type : str, default='mse'
        Type of loss function ('mse', 'l1', 'smooth_l1').
    reduction : str, default='mean'
        Reduction method ('mean', 'sum', 'none').

    Returns
    -------
    torch.Tensor
        Computed loss value.

    Examples
    --------
    >>> reconstructed = torch.randn(2, 80, 100, 128)
    >>> target = torch.randn(2, 80, 100, 128)
    >>> mask = torch.rand(2, 1, 100, 128) > 0.5
    >>> loss = mae_loss(reconstructed, target, mask, loss_type='mse')
    """
    # Invert mask: we want loss only on masked regions (where mask=0)
    masked_regions = 1 - mask

    # Compute loss per pixel
    if loss_type == "mse":
        loss_per_pixel = F.mse_loss(reconstructed, target, reduction="none")
    elif loss_type == "l1":
        loss_per_pixel = F.l1_loss(reconstructed, target, reduction="none")
    elif loss_type == "smooth_l1":
        loss_per_pixel = F.smooth_l1_loss(reconstructed, target, reduction="none")
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Available types: ['mse', 'l1', 'smooth_l1']"
        )

    # Only compute loss on masked regions
    masked_loss = loss_per_pixel * masked_regions

    # Apply reduction
    if reduction == "none":
        return masked_loss
    elif reduction == "sum":
        return masked_loss.sum()
    elif reduction == "mean":
        # Average over masked pixels only
        num_masked_pixels = masked_regions.sum()
        if num_masked_pixels > 0:
            return masked_loss.sum() / num_masked_pixels
        else:
        if num_masked_pixels < 1:
            raise ValueError(
                "No masked pixels found in mae_loss. This likely indicates a problem with mask generation or input data."
            )
        return masked_loss.sum() / num_masked_pixels
    else:
        raise ValueError(
            f"Unknown reduction: {reduction}. "
            f"Available reductions: ['none', 'sum', 'mean']"
        )
