"""Encoder block implementations derived from base classes.

This module implements the EncoderBlock and BlockBasedEncoder classes that
inherit from the base classes, following established patterns and interfaces.
"""

import torch
import torch.nn as nn
from typing import Union, Any, Optional
from .base import SequentialBlock, ConfigurableBlock, WeightInitializer
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
    padding : int, tuple of int, or str, default='auto'
        Padding for convolutions in ResidualBlock. 'auto' calculates
        padding to maintain spatial dimensions.
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
            padding: Union[int, tuple[int, int], str] = 'auto',
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
            in_channels, out_channels, kernel_size, stride, padding,
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
            padding: Union[int, tuple[int, int], str],
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
            padding=padding,
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
            'padding': getattr(self.residual_block, 'padding', 'auto'),
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

        return (batch_size, channels, pooled_height, pooled_width)

    def __repr__(self) -> str:
        """String representation of the EncoderBlock."""
        return (f"EncoderBlock("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"pool_size={self.pool_size}, "
                f"dropout={self.dropout_prob}, "
                f"activation='{self.activation_name}')")


class BlockBasedEncoder(ConfigurableBlock):
    """Encoder architecture built from a sequence of EncoderBlocks.

    This encoder provides a flexible architecture where each encoding stage
    consists of an EncoderBlock (ResidualBlock + Dropout + MaxPool) followed
    by an optional bottleneck compression layer.

    Parameters
    ----------
    input_channels : int
        Number of input channels in the data.
    block_configs : list of dict
        Configuration for each encoder block. Each dict should contain:
        - 'out_channels' (int): Output channels for the block
        - 'pool_size' (tuple, optional): MaxPool kernel size, default (1, 2)
        - 'dropout' (float, optional): Dropout probability, default 0.3
        - 'kernel_size' (int/tuple, optional): Conv kernel size, default 3
        - 'bias' (bool, optional): Use bias in convolutions, default True
        - Other ResidualBlock parameters (activation, use_batch_norm, etc.)
    bottleneck_channels : int, optional
        Number of channels in the bottleneck layer. If None, defaults to
        max(16, last_block_channels // 2).
    hidden_dim : int, optional
        Target frequency dimension after adaptive pooling. If None, no
        adaptive pooling is applied.
    kernel_size : int or tuple of int, default=3
        Default kernel size for blocks that don't specify one.
    bias : bool, default=True
        Default bias setting for blocks that don't specify one.
    bottleneck_activation : str, default='relu'
        Activation function for bottleneck layer.
    bottleneck_init_method : str, default='kaiming'
        Weight initialization method for bottleneck.

    Attributes
    ----------
    blocks : nn.ModuleList
        List of EncoderBlock modules.
    bottleneck : nn.Sequential
        Bottleneck compression layers.
    bottleneck_channels : int
        Number of channels in the bottleneck output.
    block_configs : list of dict
        Stored block configurations.
    hidden_dim : int or None
        Stored target frequency dimension.
    """

    def __init__(
            self,
            input_channels: int,
            block_configs: list[dict[str, Any]],
            bottleneck_channels: Optional[int] = None,
            hidden_dim: Optional[int] = None,
            kernel_size: Union[int, tuple[int, int]] = 3,
            bias: bool = True,
            bottleneck_activation: str = 'relu',
            bottleneck_init_method: str = 'kaiming',
            **kwargs
    ) -> None:
        """Initialize BlockBasedEncoder."""

        # Initialize ConfigurableBlock
        super().__init__(
            in_channels=input_channels,
            out_channels=input_channels,  # Will be updated after building
            kernel_size=kernel_size,
            bias=bias,
            block_configs=block_configs,
            bottleneck_channels=bottleneck_channels,
            hidden_dim=hidden_dim,
            bottleneck_activation=bottleneck_activation,
            bottleneck_init_method=bottleneck_init_method,
            **kwargs
        )

        # Validate inputs
        if not block_configs:
            raise ValueError("block_configs cannot be empty")

        if input_channels <= 0:
            raise ValueError(
                f"input_channels must be positive, got {input_channels}")

        self.input_channels = input_channels
        self.block_configs = block_configs
        self.hidden_dim = hidden_dim
        self.bottleneck_activation = bottleneck_activation
        self.bottleneck_init_method = bottleneck_init_method

        # Build encoder blocks
        self.blocks = self._build_encoder_blocks()

        # Build bottleneck
        self.bottleneck, self.bottleneck_channels = self._build_bottleneck(
            bottleneck_channels, kernel_size, bias
        )

        # Update out_channels after building
        self.out_channels = self.bottleneck_channels

    def _build_encoder_blocks(self) -> nn.ModuleList:
        """Build the sequence of encoder blocks."""
        blocks = []
        current_channels = self.input_channels

        for i, config in enumerate(self.block_configs):
            if 'out_channels' not in config:
                raise ValueError(
                    f"Block {i} missing required 'out_channels' key")

            # Extract config with defaults
            block_config = self._prepare_block_config(config, current_channels)

            # Validate channels
            out_channels = block_config['out_channels']
            if out_channels <= 0:
                raise ValueError(f"out_channels must be positive, "
                                 f"got {out_channels} in block {i}")

            # Create encoder block
            block = EncoderBlock(**block_config)
            blocks.append(block)
            current_channels = out_channels

        return nn.ModuleList(blocks)

    def _prepare_block_config(
            self,
            config: dict[str, Any],
            current_channels: int
    ) -> dict[str, Any]:
        """Prepare block configuration with defaults."""
        block_config = {
            'in_channels': current_channels,
            'out_channels': config['out_channels'],
            'pool_size': config.get('pool_size', (1, 2)),
            'kernel_size': config.get('kernel_size', self.kernel_size),
            'stride': config.get('stride', 1),
            'padding': config.get('padding', 'auto'),
            'dropout': config.get('dropout', 0.3),
            'bias': config.get('bias', self.bias),
            'use_batch_norm': config.get('use_batch_norm', True),
            'activation': config.get('activation', 'relu'),
            'residual_init_method': config.get(
                'residual_init_method', 'kaiming'),
        }
        return block_config

    def _build_bottleneck(
            self,
            bottleneck_channels: Optional[int],
            kernel_size: Union[int, tuple[int, int]],
            bias: bool
    ) -> tuple[nn.Sequential, int]:
        """Build the bottleneck compression layers."""
        bottleneck_layers = []

        # Get input channels from last block
        if self.blocks:
            current_channels = self.blocks[-1].out_channels
        else:
            current_channels = self.input_channels

        # Optional adaptive pooling
        if self.hidden_dim is not None:
            if self.hidden_dim <= 0:
                raise ValueError(f"hidden_dim must be positive, "
                                 f"got {self.hidden_dim}")
            bottleneck_layers.append(
                nn.AdaptiveAvgPool2d((None, self.hidden_dim)))

        # Channel compression
        if bottleneck_channels is None:
            bottleneck_channels = max(16, current_channels // 2)

        if bottleneck_channels <= 0:
            raise ValueError(f"bottleneck_channels must be positive, "
                             f"got {bottleneck_channels}")

        # Calculate padding for bottleneck convolution
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = tuple(k // 2 for k in kernel_size)

        # Add compression layers
        bottleneck_layers.extend([
            nn.Conv2d(
                current_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias
            ),
            nn.BatchNorm2d(bottleneck_channels),
            self._create_activation(self.bottleneck_activation),
        ])

        bottleneck = nn.Sequential(*bottleneck_layers)

        # Initialize bottleneck weights
        self._initialize_bottleneck_weights(bottleneck)

        return bottleneck, bottleneck_channels

    def _create_activation(self, activation: str) -> nn.Module:
        """Create activation function based on name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': nn.Mish(),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]

    def _initialize_bottleneck_weights(
            self,
            bottleneck: nn.Sequential
    ) -> None:
        """Initialize bottleneck weights."""
        if self.bottleneck_init_method == 'kaiming':
            bottleneck.apply(WeightInitializer.kaiming_normal_)
        elif self.bottleneck_init_method == 'xavier':
            bottleneck.apply(WeightInitializer.xavier_uniform_)

        # Always properly initialize batch norm
        bottleneck.apply(WeightInitializer.init_batch_norm_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, input_channels, height, width)

        Returns
        -------
        torch.Tensor
            Encoded latent representation with shape
            (batch_size, bottleneck_channels, height', width') where height'
            and width' depend on the pooling operations and hidden_dim.
        """
        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x)

        # Pass through bottleneck
        x = self.bottleneck(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get intermediate feature maps from each encoder block.

        Useful for visualization and debugging.

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

        for block in self.blocks:
            x = block(x)
            feature_maps.append(x.clone())

        return feature_maps

    def get_output_shape(
            self,
            input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Calculate output shape given input shape."""
        current_shape = input_shape

        # Apply each encoder block
        for block in self.blocks:
            current_shape = block.get_output_shape(current_shape)

        # Apply adaptive pooling if present
        if self.hidden_dim is not None:
            batch_size, channels, height, _ = current_shape
            current_shape = (batch_size, channels, height, self.hidden_dim)

        # Apply bottleneck channel reduction
        batch_size, _, height, width = current_shape
        return (batch_size, self.bottleneck_channels, height, width)

    def __repr__(self) -> str:
        """String representation of the BlockBasedEncoder."""
        return (f"BlockBasedEncoder("
                f"input_channels={self.input_channels}, "
                f"num_blocks={len(self.blocks)}, "
                f"bottleneck_channels={self.bottleneck_channels}, "
                f"hidden_dim={self.hidden_dim})")
