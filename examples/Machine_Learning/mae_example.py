import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.faith.train.data.datasets.file_based import JoblibDataset
from src.faith.train.data.loaders.factory import worker_init_fn


class MaskedAutoencoderLoss(torch.nn.Module):
    """Masked Autoencoder Loss Module.

    This module computes the reconstruction loss for masked autoencoders.
    It applies a specified loss function (MSE, L1, Smooth L1) only on the
    masked regions of the input tensor.

    Parameters
    ----------
    loss_type : str, default='mse'
        Type of loss function to use ('mse', 'l1', 'smooth_l1').
    reduction : str, default='mean'
        Reduction method to apply ('mean', 'sum', 'none').
    """

    def __init__(self, loss_type: str = None, reduction: str = "mean") -> None:
        super().__init__()
        if loss_type is None:
            self.loss_function = nn.MSELoss(reduction=reduction, reduce=False)
        else:
            if loss_type == "mse":
                self.loss_function = nn.MSELoss(reduction=reduction, reduce=False)
            elif loss_type == "l1":
                self.loss_function = nn.L1Loss(reduction=reduction, reduce=False)
            elif loss_type == "smooth_l1":
                self.loss_function = nn.SmoothL1Loss(reduction=reduction, reduce=False)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}. "
                                 f"Available types: ['mse', 'l1', 'smooth_l1']")

    def forward(
            self,
            predictions: dict[str, torch.Tensor],
            targets: dict[str, torch.Tensor],
            masks: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the masked loss.

        Parameters
        ----------
        predictions : dict[torch.Tensor]
            Model output tensor.
        targets : dict[torch.Tensor]
            Original input tensor (unmasked).
        masks : dict[torch.Tensor]
            Binary mask tensor (1 = keep, 0 = mask).

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        total_loss = 0
        num_masked = 0

        for modality in predictions.keys():
            pred = predictions[modality]
            target = targets[modality]
            mask = masks[modality]

            diff = self.loss_function(pred, target)

            # Apply mask by multiplication (mask should be float)
            masked_diff = diff * mask

            # Sum only non-zero elements
            total_loss += masked_diff.sum()
            num_masked += mask.sum()

        return total_loss / num_masked if num_masked > 0 else 0


"""
def mae_loss(
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        loss_type: str = 'mse',
        reduction: str = 'mean'
) -> torch.Tensor:
    '''Compute MAE loss only on masked regions.

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
    '''
    # Invert mask: we want loss only on masked regions (where mask=0)
    masked_regions = (1 - mask)

    # Compute loss per pixel
    if loss_type == 'mse':
        loss_per_pixel = F.mse_loss(reconstructed, target, reduction='none')
    elif loss_type == 'l1':
        loss_per_pixel = F.l1_loss(reconstructed, target, reduction='none')
    elif loss_type == 'smooth_l1':
        loss_per_pixel = F.smooth_l1_loss(reconstructed, target,
                                          reduction='none')
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. "
                         f"Available types: ['mse', 'l1', 'smooth_l1']")

    # Only compute loss on masked regions
    masked_loss = loss_per_pixel * masked_regions

    # Apply reduction
    if reduction == 'none':
        return masked_loss
    elif reduction == 'sum':
        return masked_loss.sum()
    elif reduction == 'mean':
        # Average over masked pixels only
        num_masked_pixels = masked_regions.sum()
        if num_masked_pixels > 0:
            return masked_loss.sum() / num_masked_pixels
        else:
            return torch.tensor(0.0, device=reconstructed.device)
    else:
        raise ValueError(f"Unknown reduction: {reduction}. "
                         f"Available reductions: ['none', 'sum', 'mean']")
"""

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

    Attributes
    ----------
    mask_ratio : float
        Stored mask ratio.
    patch_size : tuple of int
        Stored patch size.

    Examples
    --------
    >>> mask_gen = MaskGenerator(mask_ratio=0.75)
    >>> shape = (2, 80, 100, 128)  # batch, channels, freq, time
    >>> mask = mask_gen.patch_mask(shape)
    >>> print(f"Mask shape: {mask.shape}, Masked ratio: {(1 - mask.mean()).item():.2f}")
    """

    def __init__(
            self,
            mask_ratio: float = 0.75,
            patch_size: tuple[int, int] = (100, 100),
    ) -> None:
        """Initialize MaskGenerator.

        Parameters
        ----------
        mask_ratio : float, default=0.75
            Ratio of input to mask.
        patch_size : tuple of int, default=(8, 8)
            Size of patches for patch-based masking.
        """
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(
                f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")

        if len(patch_size) != 2 or any(s <= 0 for s in patch_size):
            raise ValueError(f"patch_size must be a tuple of two positive "
                             f"integers, got {patch_size}")

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def patch_mask(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Generate patch-based masking.

        Masks rectangular patches, similar to Vision Transformer masking
        but adapted for spectrograms.

        Parameters
        ----------
        shape : tuple of int
            Input tensor shape (batch_size, channels, freq_bins, time_steps).

        Returns
        -------
        torch.Tensor
            Binary mask tensor where 1 = keep, 0 = mask.
        """
        batch_size, channels, freq_bins, time_steps = shape
        mask = torch.ones(batch_size, channels, freq_bins, time_steps)

        patch_time, patch_freq = self.patch_size

        # Calculate number of patches in each dimension
        num_patches_time = time_steps // patch_time
        num_patches_freq = freq_bins // patch_freq
        total_patches = num_patches_time * num_patches_freq

        if total_patches == 0:
            return mask

        for b in range(batch_size):
            for c in range(channels):
                # Number of patches to mask
                num_masked_patches = int(total_patches * self.mask_ratio)

                if num_masked_patches > 0:
                    # Randomly select patches to mask
                    masked_patches = random.sample(
                        range(total_patches), min(num_masked_patches, total_patches))

                    for patch_idx in masked_patches:
                        # Convert patch index to 2D coordinates
                        patch_t = (patch_idx // num_patches_freq) * patch_time
                        patch_f = (patch_idx % num_patches_freq) * patch_freq

                        # Apply mask to patch
                        end_t = min(patch_t + patch_time, time_steps)
                        end_f = min(patch_f + patch_freq, freq_bins)

                        mask[b, c, patch_f:end_f, patch_t:end_t] = 0

        return mask.int()


# Example usage and testing
if __name__ == "__main__":
    # Test MaskGenerator
    print("Testing MaskGenerator...")
    mask_gen = MaskGenerator(mask_ratio=0.75)
    # Test MaskedAutoencoder
    print("\nTesting MaskedAutoencoder...")

    dataset = JoblibDataset(
        file_paths = ["171348_0.joblib"],
        subseq_len=128,  # Extract 128-sample subsequences
        input_key=["mhr", "ece", "co2", ],  # Specify your input keys as list
        target_key=None,  # Autoencoder mode
        validate_on_init=True
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        worker_init_fn=worker_init_fn
    )

    # Test forward pass
    # Test loading a batch
    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Target shape: {targets.shape}")
        print(f"  Input dtype: {inputs.dtype}")
        print(f"  Target dtype: {targets.dtype}")

    batch_size, freq_bins, time_steps = (2, 512, 2048)
    x = {
        "hrs": torch.randn(batch_size, 8, freq_bins, time_steps),
        "ece": torch.randn(batch_size, 40, freq_bins, time_steps),
        "co2": torch.randn(batch_size, 4, freq_bins, time_steps),
    }

    y = {
        "hrs": torch.randn(batch_size, 8, freq_bins, time_steps),
        "ece": torch.randn(batch_size, 40, freq_bins, time_steps),
        "co2": torch.randn(batch_size, 4, freq_bins, time_steps),
    }

    # Generate masks
    masks = {
        "hrs": mask_gen.patch_mask(shape=(batch_size, 8, freq_bins, time_steps)),
        "ece": mask_gen.patch_mask(shape=(batch_size, 40, freq_bins, time_steps)),
        "co2": mask_gen.patch_mask(shape=(batch_size, 4, freq_bins, time_steps)),
    }
    # Test MAE loss
    mae_loss = MaskedAutoencoderLoss(loss_type="mse")
    loss = mae_loss(x, y, masks)
    # loss = mae_loss(reconstructed, x, mask, loss_type='mse')
    print(f"MAE loss: {loss.item():.6f}")
