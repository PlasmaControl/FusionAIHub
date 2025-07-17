"""Encoder block implementations derived from base classes.

This module implements the EncoderBlock and BlockBasedEncoder classes that
inherit from the base classes, following established patterns and interfaces.
"""

from typing import Any, Union

import torch
import torch.nn as nn

from .base import SequentialBlock
from .residual import ResidualBlock


class EncoderBlock(SequentialBlock):
    """
    Single encoder block: ResidualBlock + Dropout + MaxPool.

    This block represents the fundamental building unit of the encoder,
    combining feature extraction through ResidualBlock, regularization
    through Dropout, and spatial downsampling through MaxPooling.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels from the ResidualBlock.
    pool_size : tuple of int, default=(1, 2)
        Kernel size for MaxPool2d operation. Format: (height, width).
    kernel_size : int or tuple of int, default=3
        Kernel size for convolutions in ResidualBlock.
    stride : int or tuple of int, default=1
        Stride for convolutions in ResidualBlock. The EncoderBlock uses
        stride=1 and relies on MaxPool for downsampling.
    dropout : float, default=0.3
        Dropout probability. Must be between 0.0 and 1.0.
    bias : bool, default=True
        Whether to use bias in convolution layers.
    use_batch_norm : bool, default=True
        Whether to use batch normalization in ResidualBlock.
    activation : str, default='relu'
        Activation function for ResidualBlock.
    residual_init_method : str, default='kaiming'
        Weight initialization method for ResidualBlock.

    Attributes
    ----------
    residual_block : ResidualBlock
        The residual convolutional block for feature extraction.
    dropout : nn.Dropout
        Dropout layer for regularization.
    pool : nn.MaxPool2d
        Max pooling layer for spatial downsampling.
    pool_size : tuple of int
        Stored pooling size for decoder symmetry.
    dropout_prob : float
        Stored dropout probability.

    Examples
    --------
    >>> block = EncoderBlock(in_channels=64, out_channels=128)
    >>> x = torch.randn(1, 64, 32, 32)
    >>> out = block(x)
    >>> print(out.shape)
    torch.Size([1, 128, 32, 16])

    >>> # Custom configuration
    >>> block = EncoderBlock(
    ...     in_channels=64, out_channels=128,
    ...     pool_size=(2, 2), dropout=0.5, activation='gelu'
    ... )
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            pool_size: tuple[int, int] = (1, 2),
            kernel_size: Union[int, tuple[int, int]] = 3,
            stride: Union[int, tuple[int, int]] = 1,
            dropout: float = 0.3,
            bias: bool = True,
            use_batch_norm: bool = True,
            activation: str = 'relu',
            residual_init_method: str = 'kaiming'
    ) -> None:
        """Initialize EncoderBlock."""

        # Validate parameters
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(
                f"Dropout must be between 0.0 and 1.0, got {dropout}")

        if len(pool_size) != 2:
            raise ValueError(
                f"pool_size must be a tuple of length 2, got {pool_size}")

        # Store configuration
        self.pool_size = pool_size
        self.dropout_prob = dropout
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation
        self.residual_init_method = residual_init_method

        # Build the sequential operations
        operations = self._build_operations(
            in_channels, out_channels, kernel_size, stride,
            bias, use_batch_norm, activation, residual_init_method
        )

        # Initialize SequentialBlock with operations
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            operations=operations,
            kernel_size=kernel_size,
            bias=bias
        )

        # Store individual components for introspection
        self.residual_block = self.operations[0]
        self.dropout = self.operations[1]
        self.pool = self.operations[2]

    def _build_operations(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]],
            stride: Union[int, tuple[int, int]],
            bias: bool,
            use_batch_norm: bool,
            activation: str,
            init_method: str
    ) -> list[nn.Module]:
        """Build the list of operations for this encoder block."""

        operations = []

        # 1. ResidualBlock for feature extraction
        residual_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_method=init_method
        )
        operations.append(residual_block)

        # 2. Dropout for regularization
        dropout_layer = nn.Dropout(p=self.dropout_prob)
        operations.append(dropout_layer)

        # 3. MaxPool for downsampling
        pool_layer = nn.MaxPool2d(kernel_size=self.pool_size)
        operations.append(pool_layer)

        return operations

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for this block."""
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'dropout': self.dropout_prob,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation_name,
            'residual_init_method': self.residual_init_method,
            'stride': getattr(self.residual_block, 'stride', 1),
        })
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'EncoderBlock':
        """Create EncoderBlock instance from configuration dictionary."""
        return cls(**config)

    def get_output_shape(self, input_shape: tuple[int, ...]) \
            -> tuple[int, ...]:
        """Calculate output shape given input shape."""

        # Get shape after residual block
        residual_output_shape = (
            self.residual_block.get_output_shape(input_shape))

        # Apply pooling
        batch_size, channels, height, width = residual_output_shape

        # Calculate pooled dimensions
        pooled_height = height // self.pool_size[0]
        pooled_width = width // self.pool_size[1]

        return batch_size, channels, pooled_height, pooled_width

    def __repr__(self) -> str:
        """String representation of the EncoderBlock."""
        return (
            f"EncoderBlock("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"pool_size={self.pool_size}, "
            f"dropout={self.dropout_prob}, "
            f"activation='{self.activation_name}')"
        )


class BlockBasedEncoder(SequentialBlock):
    """Encoder architecture built from a sequence of EncoderBlocks.

    This encoder provides a flexible architecture where each encoding stage
    consists of an EncoderBlock with configurable parameters. The blocks are
    automatically chained together with matching input/output channels.

    Parameters
    ----------
    in_channels : int
        Number of input channels in the data.
    block_configs : list of dict
        Configuration for each encoder block. Each dict should contain:
        - 'out_channels' (int): Output channels for the block (required)
        - 'pool_size' (tuple, optional): MaxPool kernel size, default (1, 2)
        - 'dropout' (float, optional): Dropout probability, default 0.3
        - 'kernel_size' (int/tuple, optional): Conv kernel size, default 3
        - 'activation' (str, optional): Activation function, default 'relu'
        - 'use_batch_norm' (bool, optional): Use batch norm, default True
        - 'bias' (bool, optional): Use bias in convolutions, default True

    Attributes
    ----------
    blocks : list of EncoderBlock
        List of EncoderBlock modules (accessed via self.operations).
    block_configs : list of dict
        Stored block configurations.

    Examples
    --------
    >>> # Simple 3-block encoder
    >>> configs = [
    ...     {'out_channels': 64},
    ...     {'out_channels': 128, 'pool_size': (2, 2)},
    ...     {'out_channels': 256, 'dropout': 0.5}
    ... ]
    >>> encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)
    >>> x = torch.randn(1, 3, 32, 64)
    >>> output = encoder(x)
    """

    def __init__(
        self,
        in_channels: int,
        block_configs: list[dict[str, Any]],
        kernel_size: Union[int, tuple[int, int]] = 3,
        bias: bool = True,
        **kwargs
    ) -> None:
        """Initialize BlockBasedEncoder."""

        # Validate inputs
        if not block_configs:
            raise ValueError("block_configs cannot be empty")

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )

        # Validate that all configs have out_channels
        for i, config in enumerate(block_configs):
            if "out_channels" not in config:
                raise ValueError(
                    f"Block {i} missing required 'out_channels' key"
                )
            if config["out_channels"] <= 0:
                raise ValueError(f"out_channels must be positive, "
                                 f"got {config['out_channels']} in block {i}")

        self.block_configs = block_configs

        # Build encoder blocks
        operations = self._build_encoder_blocks(in_channels, kernel_size, bias)

        # Get final output channels from last block
        final_out_channels = block_configs[-1]["out_channels"]

        # Initialize SequentialBlock with operations
        super().__init__(
            in_channels=in_channels,
            out_channels=final_out_channels,
            operations=operations,
            kernel_size=kernel_size,
            bias=bias,
        )

    def _build_encoder_blocks(
        self,
        in_channels: int,
        default_kernel_size: Union[int, tuple[int, int]],
        default_bias: bool,
    ) -> list[nn.Module]:
        """
        Build the sequence of encoder blocks with automatic channel chaining.
        """
        blocks = []
        current_channels = in_channels

        for i, config in enumerate(self.block_configs):
            # Prepare block configuration with defaults
            block_config = {
                "in_channels": current_channels,
                "out_channels": config["out_channels"],
                "pool_size": config.get("pool_size", (1, 2)),
                "kernel_size": config.get("kernel_size", default_kernel_size),
                "stride": config.get("stride", 1),
                "dropout": config.get("dropout", 0.3),
                "bias": config.get("bias", default_bias),
                "use_batch_norm": config.get("use_batch_norm", True),
                "activation": config.get("activation", "relu"),
                "residual_init_method": config.get(
                    "residual_init_method", "kaiming"
                ),
            }

            # Create encoder block
            block = EncoderBlock(**block_config)
            blocks.append(block)

            # Update current channels for next block
            current_channels = config["out_channels"]

        return blocks

    def get_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get intermediate feature maps from each encoder block.

        Useful for visualization, debugging, and skip connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        list of torch.Tensor
            Feature maps after each encoder block.
        """
        feature_maps = []

        for block in self.operations:
            x = block(x)
            feature_maps.append(x.clone())

        return feature_maps

    def get_output_shape(self, input_shape: tuple[int, ...]) \
            -> tuple[int, ...]:
        """Calculate output shape given input shape."""
        shape = input_shape
        for block in self.operations:
            shape = block.get_output_shape(shape)
        return shape

    def get_channel_progression(self) -> list[int]:
        """Get the channel count progression through the encoder.

        Returns
        -------
        list of int
            Channel counts: [in_channels, block1_out, block2_out, ...]
        """
        channels = [self.in_channels]
        for config in self.block_configs:
            channels.append(config["out_channels"])
        return channels

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for this encoder."""
        config = super().get_config()
        config.update(
            {
                "block_configs": self.block_configs,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BlockBasedEncoder":
        """Create BlockBasedEncoder instance from configuration dictionary."""
        return cls(**config)

    @property
    def blocks(self) -> list[nn.Module]:
        """Access to encoder blocks for compatibility."""
        return list(self.operations)

    def __repr__(self) -> str:
        """String representation of the BlockBasedEncoder."""
        channel_progression = ' → '.join(
            map(str, self.get_channel_progression()))
        return (f"BlockBasedEncoder("
                f"blocks={len(self.operations)}, "
                f"channels={channel_progression})")
