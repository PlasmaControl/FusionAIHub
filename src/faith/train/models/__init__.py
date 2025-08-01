"""Model implementations for block-based autoencoders.

This package provides complete autoencoder models built from the modular
blocks, including standard autoencoders, masked autoencoders (MAE), and
various configuration utilities.

The main components are:
- BlockBasedAutoencoder: Complete autoencoder with configurable architecture
- MaskedAutoencoder: MAE implementation for self-supervised learning
- MaskGenerator: Various masking strategies for MAE training
- Configuration utilities for creating models from presets or config files

Examples
--------
Basic usage:
>>> from faith.train.models import (
...     BlockBasedAutoencoder, create_block_autoencoder)
>>> autoencoder = create_block_autoencoder('default', input_channels=80)
>>> x = torch.randn(1, 80, 100, 128)
>>> reconstructed, latent = autoencoder(x)

Masked autoencoder:
>>> from faith.train.models import MaskedAutoencoder, MaskGenerator
>>> mask_gen = MaskGenerator(mask_ratio=0.75)
>>> mae = MaskedAutoencoder(autoencoder, mask_gen)
>>> reconstructed, mask, masked_input = mae(x, mask_type='frequency')
"""

# TODO masked loss functions

# Core model implementations
from .autoencoder import BlockBasedAutoencoder

# Configuration and factory functions
from .configs import (
    PRESET_CONFIGS,
    ModelConfig,
    create_autoencoder_from_config,
    create_block_autoencoder,
    create_model_from_config_file,
    get_preset_config,
    list_preset_configs,
    load_model_config,
    save_model_config,
)

# Masked autoencoder components
from .mae import MaskedAutoencoder, MaskGenerator, mae_loss

# Utility functions
from .utils import (
    create_mae_model,
    get_memory_estimate,
    get_model_info,
    validate_input_shape,
)

# Public API - only these should be imported by users
__all__ = [
    # Core models
    "BlockBasedAutoencoder",
    # Masked autoencoder components
    "MaskedAutoencoder",
    "MaskGenerator",
    "mae_loss",
    # Configuration utilities
    "create_block_autoencoder",
    "create_autoencoder_from_config",
    "create_model_from_config_file",
    "get_preset_config",
    "list_preset_configs",
    "save_model_config",
    "load_model_config",
    "ModelConfig",
    "PRESET_CONFIGS",
    # Utility functions
    "create_mae_model",
    "get_model_info",
    "get_memory_estimate",
    "validate_input_shape",
]
