"""Complete autoencoder model implementation.

This module provides the BlockBasedAutoencoder class that combines the
encoder and decoder blocks into a complete autoencoder architecture.
The autoencoder uses the modular blocks from the blocks package to create
flexible and configurable models for audio and spectral data.
"""

import torch
import torch.nn as nn
from typing import Union, Optional, Any

# Import the building blocks
from ..blocks import BlockBasedEncoder
from ..blocks.decoder import BlockBasedDecoder


class BlockBasedAutoencoder(nn.Module):
    """Complete autoencoder built from modular encoder and decoder blocks.

    This autoencoder provides a flexible, block-based architecture where
    both encoder and decoder are constructed from configurable blocks.
    The decoder automatically mirrors the encoder's architecture for
    symmetric reconstruction.

    Parameters
    ----------
    input_channels : int
        Number of input channels in the data.
    block_configs : list of dict, optional
        Configuration for encoder blocks. If None, uses default configuration.
        Each dict should contain 'out_channels' and optionally 'pool_size',
        'dropout', 'kernel_size', 'bias', etc.
    bottleneck_channels : int, optional
        Number of channels in the bottleneck. If None, auto-calculated.
    kernel_size : int or tuple of int, default=3
        Default kernel size for convolutions.
    bias : bool, default=True
        Whether to use bias in convolution layers.
    upsampling_mode : str, default='nearest'
        Upsampling algorithm for decoder ('nearest', 'bilinear', etc.).
    use_batch_norm : bool, default=True
        Whether to use batch normalization in blocks.
    activation : str, default='relu'
        Default activation function for blocks.
    init_method : str, default='kaiming'
        Weight initialization method.

    Attributes
    ----------
    encoder : BlockBasedEncoder
        The encoder module.
    decoder : BlockBasedDecoder
        The decoder module.
    input_channels : int
        Number of input channels.
    block_configs : list of dict
        Encoder block configurations.

    Examples
    --------
    >>> # Basic usage with default configuration
    >>> autoencoder = BlockBasedAutoencoder(input_channels=80)
    >>> x = torch.randn(1, 80, 100, 128)
    >>> reconstructed, latent = autoencoder(x)

    >>> # Custom configuration
    >>> configs = [
    ...     {'out_channels': 64, 'pool_size': (1, 2)},
    ...     {'out_channels': 128, 'pool_size': (1, 4)},
    ... ]
    >>> autoencoder = BlockBasedAutoencoder(
    ...     input_channels=80,
    ...     block_configs=configs,
    ... )
    """

    def __init__(
            self,
            input_channels: int,
            block_configs: Optional[list[dict[str, Any]]] = None,
            bottleneck_channels: Optional[int] = None,
            kernel_size: Union[int, tuple[int, int]] = 3,
            bias: bool = True,
            upsampling_mode: str = 'nearest',
            use_batch_norm: bool = True,
            activation: str = 'relu',
            init_method: str = 'kaiming',
            **kwargs
    ) -> None:
        """Initialize BlockBasedAutoencoder.

        Parameters
        ----------
        input_channels : int
            Number of input channels in the data.
        block_configs : list of dict, optional
            Configuration for encoder blocks.
        bottleneck_channels : int, optional
            Number of channels in the bottleneck.
        kernel_size : int or tuple of int, default=3
            Default kernel size for convolutions.
        bias : bool, default=True
            Whether to use bias in convolution layers.
        upsampling_mode : str, default='nearest'
            Upsampling algorithm for decoder.
        use_batch_norm : bool, default=True
            Whether to use batch normalization.
        activation : str, default='relu'
            Default activation function.
        init_method : str, default='kaiming'
            Weight initialization method.
        **kwargs
            Additional arguments (for future extensibility).
        """
        super().__init__()

        if input_channels <= 0:
            raise ValueError(
                f"input_channels must be positive, got {input_channels}")

        # Store configuration
        self.input_channels = input_channels
        self.block_configs = block_configs
        self.upsampling_mode = upsampling_mode
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.init_method = init_method

        # Use default block configuration if none provided
        if block_configs is None:
            block_configs = self._get_default_block_configs()

        # Create encoder
        self.encoder = BlockBasedEncoder(
            input_channels=input_channels,
            block_configs=block_configs,
            bottleneck_channels=bottleneck_channels,
            kernel_size=kernel_size,
            bias=bias,
            bottleneck_activation=activation,
            bottleneck_init_method=init_method
        )

        # Create decoder that mirrors the encoder
        self.decoder = BlockBasedDecoder(
            output_channels=input_channels,
            encoder_blocks=self.encoder.blocks,
            bottleneck_channels=self.encoder.bottleneck_channels,
            kernel_size=kernel_size,
            bias=bias,
            upsampling_mode=upsampling_mode,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_method=init_method
        )

    def _get_default_block_configs(self) -> list[dict[str, Any]]:
        """Get default block configuration."""
        return [
            {'out_channels': 32, 'pool_size': (1, 2)},
            {'out_channels': 16, 'pool_size': (1, 2)},
        ]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input with shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Latent representation with shape
            (batch_size, bottleneck_channels, height', width').
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstructed output.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation with shape
            (batch_size, bottleneck_channels, height, width).

        Returns
        -------
        torch.Tensor
            Reconstructed output with shape approximately matching
            the original input dimensions.
        """
        return self.decoder(z)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete autoencoder.

        Parameters
        ----------
        inputs : torch.Tensor
            Input with shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Reconstructed input with the same shape as input.
        """
        latent = self.encode(inputs)
        reconstructed = self.decode(latent)
        return reconstructed

    def latent_with_reconstruction(self, inputs: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the complete autoencoder.

        Parameters
        ----------
        inputs : torch.Tensor
            Input with shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Latent representation with shape
            (batch_size, bottleneck_channels, height', width').
        torch.Tensor
            Reconstructed output with shape approximately matching the original
            input dimensions.
        """
        latent = self.encode(inputs)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for this autoencoder.

        Returns
        -------
        dict
            Configuration dictionary containing all parameters needed
            to reconstruct this autoencoder.
        """
        return {
            'input_channels': self.input_channels,
            'block_configs': self.block_configs,
            'bottleneck_channels': self.encoder.bottleneck_channels,
            'kernel_size': self.encoder.kernel_size,
            'bias': self.encoder.bias,
            'upsampling_mode': self.upsampling_mode,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
            'init_method': self.init_method,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'BlockBasedAutoencoder':
        """Create BlockBasedAutoencoder instance from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary from get_config().

        Returns
        -------
        BlockBasedAutoencoder
            New autoencoder instance.
        """
        return cls(**config)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[
        int, ...]:
        """Calculate output shape given input shape.

        Parameters
        ----------
        input_shape : tuple of int
            Input tensor shape (batch, channels, height, width).

        Returns
        -------
        tuple of int
            Output tensor shape (should match input for autoencoder).
        """
        # For autoencoder, output shape should match input shape
        latent_shape = self.encoder.get_output_shape(input_shape)
        output_shape = self.decoder.get_output_shape(latent_shape)
        return output_shape

    def get_latent_shape(self, input_shape: tuple[int, ...]) -> tuple[
        int, ...]:
        """Calculate latent representation shape given input shape.

        Parameters
        ----------
        input_shape : tuple of int
            Input tensor shape (batch, channels, height, width).

        Returns
        -------
        tuple of int
            Latent tensor shape.
        """
        return self.encoder.get_output_shape(input_shape)

    def get_feature_maps(self, inputs: torch.Tensor) -> dict[
        str, list[torch.Tensor]]:
        """Get intermediate feature maps from encoder and decoder.

        Useful for visualization and debugging.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.

        Returns
        -------
        dict
            Dictionary containing:
            - 'encoder': List of feature maps from encoder blocks
            - 'decoder': List of feature maps from decoder blocks
        """
        # Get encoder feature maps
        encoder_features = self.encoder.get_feature_maps(inputs)

        # Get latent representation
        latent = self.encoder(inputs)

        # Get decoder feature maps
        decoder_features = self.decoder.get_feature_maps(latent)

        return {
            'encoder': encoder_features,
            'decoder': decoder_features,
            'latent': latent
        }

    @property
    def parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def encoder_parameter_count(self) -> int:
        """Get number of trainable parameters in encoder."""
        return sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad)

    @property
    def decoder_parameter_count(self) -> int:
        """Get number of trainable parameters in decoder."""
        return sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad)

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters (useful for fine-tuning decoder only)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self) -> None:
        """Freeze decoder parameters (useful for feature extraction)."""
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def __repr__(self) -> str:
        """String representation of the BlockBasedAutoencoder."""
        return (f"BlockBasedAutoencoder("
                f"input_channels={self.input_channels}, "
                f"encoder_blocks={len(self.encoder.blocks)}, "
                f"decoder_blocks={len(self.decoder.blocks)}, "
                f"bottleneck_channels={self.encoder.bottleneck_channels}, "
                f"parameters={self.parameter_count:,})")


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    print("Testing BlockBasedAutoencoder...")

    # Create autoencoder with default config
    autoencoder = BlockBasedAutoencoder(input_channels=80)

    # Test forward pass
    x = torch.randn(2, 80, 100, 128)
    reconstructed = autoencoder(x)
    latent = autoencoder.encode(x)

    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Autoencoder: {autoencoder}")

    # Test individual methods
    latent_only = autoencoder.get_latent_representation(x)
    reconstructed_only = autoencoder.reconstruct(x)

    print(f"Latent only shape: {latent_only.shape}")
    print(f"Reconstructed only shape: {reconstructed_only.shape}")

    # Test configuration serialization
    config = autoencoder.get_config()
    print(f"Config keys: {list(config.keys())}")

    new_autoencoder = BlockBasedAutoencoder.from_config(config)
    print(f"Recreated autoencoder: {new_autoencoder}")

    # Test shape calculation
    output_shape = autoencoder.get_output_shape((1, 80, 100, 128))
    latent_shape = autoencoder.get_latent_shape((1, 80, 100, 128))
    print(f"Calculated output shape: {output_shape}")
    print(f"Calculated latent shape: {latent_shape}")

    # Test parameter counting
    print(f"Total parameters: {autoencoder.parameter_count:,}")
    print(f"Encoder parameters: {autoencoder.encoder_parameter_count:,}")
    print(f"Decoder parameters: {autoencoder.decoder_parameter_count:,}")

    # Test feature map extraction
    feature_maps = autoencoder.get_feature_maps(x)
    print(f"Encoder feature maps: {len(feature_maps['encoder'])}")
    print(f"Decoder feature maps: {len(feature_maps['decoder'])}")

    # Test custom configuration
    custom_configs = [
        {'out_channels': 64, 'pool_size': (1, 2), 'dropout': 0.2},
        {'out_channels': 128, 'pool_size': (1, 4), 'dropout': 0.3},
    ]

    custom_autoencoder = BlockBasedAutoencoder(
        input_channels=80,
        block_configs=custom_configs,
        activation='gelu'
    )

    x_custom = torch.randn(1, 80, 100, 128)
    reconstructed_custom, latent_custom = custom_autoencoder(x_custom)

    print(f"\nCustom autoencoder:")
    print(f"Input shape: {x_custom.shape}")
    print(f"Latent shape: {latent_custom.shape}")
    print(f"Reconstructed shape: {reconstructed_custom.shape}")
    print(f"Custom autoencoder: {custom_autoencoder}")
