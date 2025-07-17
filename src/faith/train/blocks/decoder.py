"""Decoder block implementations derived from base classes.

This module implements the DecoderBlock and BlockBasedDecoder classes that
inherit from the base classes, following established patterns and interfaces.
The decoder creates a symmetric reconstruction path to the encoder.
"""

from typing import Any, Optional, Union

import torch
import torch.nn as nn

from .base import ConfigurableBlock, SequentialBlock, WeightInitializer
from .encoder import EncoderBlock
from .residual import ResidualBlock


class DecoderBlock(SequentialBlock):
    """
    Single decoder block: ConvTranspose2d + Dropout + ResidualBlock.

    This block represents the fundamental building unit of the decoder,
    combining spatial upsampling through ConvTranspose2d, regularization
    through Dropout, and feature refinement through ResidualBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels from the ResidualBlock.
    upsample_factor : tuple of int, default=(1, 2)
        Scale factor for upsampling. Format: (height_factor, width_factor).
    kernel_size : int or tuple of int, default=3
        Kernel size for convolutions in ResidualBlock.
    stride : int or tuple of int, default=1
        Stride for convolutions in ResidualBlock. The DecoderBlock uses
        stride=1 and relies on ConvTranspose2d for upsampling.
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
    conv_transpose : nn.ConvTranspose2d
        Transposed convolution layer for spatial upsampling.
    dropout_layer : nn.Dropout
        Dropout layer for regularization.
    residual_block : ResidualBlock
        The residual convolutional block for feature refinement.
    upsample_factor : tuple of int
        Stored upsampling factor for encoder symmetry.
    dropout : float
        Stored dropout probability.

    Examples
    --------
    >>> block = DecoderBlock(in_channels=128, out_channels=64)
    >>> x = torch.randn(1, 128, 16, 8)
    >>> out = block(x)
    >>> print(out.shape)
    torch.Size([1, 64, 16, 16])

    >>> # Custom configuration
    >>> block = DecoderBlock(
    ...     in_channels=128, out_channels=64,
    ...     upsample_factor=(2, 2), dropout=0.5, activation='gelu'
    ... )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: tuple[int, int] = (1, 2),
        kernel_size: Union[int, tuple[int, int]] = 3,
        stride: Union[int, tuple[int, int]] = 1,
        dropout: float = 0.3,
        bias: bool = True,
        use_batch_norm: bool = True,
        activation: str = "relu",
        residual_init_method: str = "kaiming",
    ) -> None:
        """Initialize DecoderBlock."""

        # Validate parameters
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(
                f"Dropout must be between 0.0 and 1.0, got {dropout}"
            )

        if len(upsample_factor) != 2:
            raise ValueError(f"upsample_factor must be a tuple of length 2, "
                             f"got {upsample_factor}")

        # Store configuration
        self.upsample_factor = upsample_factor
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation
        self.residual_init_method = residual_init_method

        # Build the sequential operations
        operations = self._build_operations(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias,
            use_batch_norm,
            activation,
            residual_init_method,
        )

        # Initialize SequentialBlock with operations
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            operations=operations,
            kernel_size=kernel_size,
            bias=bias,
        )

        # Store individual components for introspection
        self.conv_transpose = self.operations[0]
        self.dropout_layer = self.operations[1]
        self.residual_block = self.operations[2]

    def _build_operations(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]],
        bias: bool,
        use_batch_norm: bool,
        activation: str,
        init_method: str,
    ) -> list[nn.Module]:
        """Build the list of operations for this decoder block."""

        operations = []

        # 1. ConvTranspose2d for upsampling
        # Calculate kernel size and padding based on upsample factor
        # For stride=1 in any dimension, we need to handle it carefully

        # Determine kernel size and padding for each dimension
        kernel_h = 3 if self.upsample_factor[0] == 1 else 4
        kernel_w = 3 if self.upsample_factor[1] == 1 else 4

        conv_transpose_layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,  # Keep same channels for upsampling
            kernel_size=(kernel_h, kernel_w),
            stride=self.upsample_factor,
            padding=(1, 1),  # Use padding=1 for both cases
            bias=bias,
        )
        operations.append(conv_transpose_layer)

        # 2. Dropout for regularization
        dropout_layer = nn.Dropout(p=self.dropout)
        operations.append(dropout_layer)

        # 3. ResidualBlock for feature refinement
        residual_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_method=init_method,
        )
        operations.append(residual_block)

        return operations

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for this block."""
        config = super().get_config()
        config.update(
            {
                "upsample_factor": self.upsample_factor,
                "dropout": self.dropout,
                "use_batch_norm": self.use_batch_norm,
                "activation": self.activation_name,
                "residual_init_method": self.residual_init_method,
                "stride": getattr(self.residual_block, "stride", 1),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DecoderBlock":
        """Create DecoderBlock instance from configuration dictionary."""
        return cls(**config)

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Calculate output shape given input shape."""

        # Get shape after ConvTranspose2d upsampling
        batch_size, channels, height, width = input_shape

        # Calculate upsampled dimensions using ConvTranspose2d formula
        # output_size = (input_size - 1) * stride - 2 * padding + kernel_size

        # Use appropriate kernel for each dimension based on upsample factor
        kernel_h = 3 if self.upsample_factor[0] == 1 else 4
        kernel_w = 3 if self.upsample_factor[1] == 1 else 4
        padding = 1  # Always use padding=1

        upsampled_height = (
            (height - 1) * self.upsample_factor[0] - 2 * padding + kernel_h
        )
        upsampled_width = (
            (width - 1) * self.upsample_factor[1] - 2 * padding + kernel_w
        )

        # Shape after ConvTranspose2d (channels remain the same)
        conv_transpose_output_shape = (
            batch_size,
            channels,
            upsampled_height,
            upsampled_width,
        )

        # Apply ResidualBlock (changes channels to out_channels)
        residual_output_shape = self.residual_block.get_output_shape(
            conv_transpose_output_shape
        )

        return residual_output_shape

    def __repr__(self) -> str:
        """String representation of the DecoderBlock."""
        return (
            f"DecoderBlock("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"upsample_factor={self.upsample_factor}, "
            f"dropout={self.dropout}, "
            f"activation='{self.activation_name}')"
        )


class BlockBasedDecoder(ConfigurableBlock):
    """Decoder architecture built from a sequence of DecoderBlocks.

    This decoder mirrors the encoder architecture by using the encoder's
    block configuration to create a symmetric upsampling path. Each decoding
    stage consists of a DecoderBlock (Upsample + ResidualBlock + Dropout).

    Parameters
    ----------
    output_channels : int
        Number of channels in the final output (should match encoder input).
    encoder_blocks : list of EncoderBlock
        List of encoder blocks to create symmetric decoder from.
    bottleneck_channels : int
        Number of channels from the encoder's bottleneck.
    kernel_size : int or tuple of int, default=3
        Default kernel size for convolutions.
    bias : bool, default=True
        Default bias setting for convolutions.
    upsampling_mode : str, default='nearest'
        Upsampling algorithm for all decoder blocks.
    use_batch_norm : bool, default=True
        Whether to use batch normalization in blocks.
    activation : str, default='relu'
        Default activation function for blocks.
    init_method : str, default='kaiming'
        Weight initialization method.
    reconstruction_kernel_size : int or tuple of int, optional
        Kernel size for final reconstruction layer. If None, uses kernel_size.

    Attributes
    ----------
    decoder_start : nn.Sequential
        Initial layers to process bottleneck output.
    blocks : nn.ModuleList
        List of DecoderBlock modules.
    reconstruction : nn.Conv2d
        Final reconstruction convolution layer.
    output_channels : int
        Number of output channels.
    upsampling_mode : str
        Upsampling mode used throughout decoder.
    """

    def __init__(
            self,
            output_channels: int,
            encoder_blocks: list[EncoderBlock],
            bottleneck_channels: int,
            kernel_size: Union[int, tuple[int, int]] = 3,
            bias: bool = True,
            upsampling_mode: str = 'nearest',
            use_batch_norm: bool = True,
            activation: str = 'relu',
            init_method: str = 'kaiming',
            reconstruction_kernel_size:
            Optional[Union[int, tuple[int, int]]] = None,
            **kwargs
    ) -> None:
        """Initialize BlockBasedDecoder."""

        # Initialize ConfigurableBlock
        super().__init__(
            in_channels=bottleneck_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            bias=bias,
            encoder_blocks=encoder_blocks,
            bottleneck_channels=bottleneck_channels,
            upsampling_mode=upsampling_mode,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_method=init_method,
            reconstruction_kernel_size=reconstruction_kernel_size,
            **kwargs
        )

        # Validate inputs
        if output_channels <= 0:
            raise ValueError(f"output_channels must be positive, "
                             f"got {output_channels}")

        if bottleneck_channels <= 0:
            raise ValueError(f"bottleneck_channels must be positive, "
                             f"got {bottleneck_channels}")

        self.output_channels = output_channels
        self.encoder_blocks = encoder_blocks
        self.bottleneck_channels = bottleneck_channels
        self.upsampling_mode = upsampling_mode
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation
        self.init_method = init_method

        if reconstruction_kernel_size is None:
            reconstruction_kernel_size = kernel_size
        self.reconstruction_kernel_size = reconstruction_kernel_size

        # Build decoder components
        self.decoder_start = self._build_decoder_start(kernel_size, bias)
        self.blocks = self._build_decoder_blocks()
        self.reconstruction = self._build_reconstruction_layer()

        # Initialize weights
        self._initialize_weights()

    def _build_decoder_start(
            self,
            kernel_size: Union[int, tuple[int, int]],
            bias: bool
    ) -> nn.Sequential:
        """Build the initial decoder layers to process bottleneck output."""
        # Calculate padding for convolution
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = tuple(k // 2 for k in kernel_size)

        # Determine first block channels
        first_block_channels = (
            self.encoder_blocks[-1].out_channels
            if self.encoder_blocks
            else self.bottleneck_channels
        )

        layers = [
            nn.Conv2d(
                self.bottleneck_channels,
                first_block_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias and not self.use_batch_norm
            )
        ]

        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(first_block_channels))

        layers.append(self._create_activation(self.activation_name))

        return nn.Sequential(*layers)

    def _build_decoder_blocks(self) -> nn.ModuleList:
        """Build decoder blocks by mirroring encoder blocks."""
        blocks = []

        # Get the output channels from decoder_start
        current_channels = (
            self.encoder_blocks[-1].out_channels
            if self.encoder_blocks
            else self.bottleneck_channels
        )

        # Create symmetric decoder by reversing encoder blocks
        for i, encoder_block in enumerate(reversed(self.encoder_blocks)):
            # Determine output channels for this decoder block
            if i == len(self.encoder_blocks) - 1:
                # Last block outputs to final channels
                out_channels = self.output_channels
            else:
                # Use the input channels of the corresponding encoder block
                corresponding_encoder_idx = len(self.encoder_blocks) - 2 - i
                out_channels = (
                    self.encoder_blocks[corresponding_encoder_idx].in_channels)

            # Create decoder block configuration
            block_config = self._create_decoder_block_config(
                encoder_block, current_channels, out_channels
            )

            # Create decoder block
            decoder_block = DecoderBlock(**block_config)
            blocks.append(decoder_block)
            current_channels = out_channels

        return nn.ModuleList(blocks)

    def _create_decoder_block_config(
        self,
        encoder_block: EncoderBlock,
        in_channels: int,
        out_channels: int
    ) -> dict[str, Any]:
        """Create configuration for a decoder block based on encoder block."""
        return {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'upsample_factor': encoder_block.pool_size,  # Mirror the pooling
            'kernel_size': self.kernel_size,
            'stride': 1,  # Always use stride=1 in decoder
            'dropout': encoder_block.dropout,  # Match encoder dropout
            'bias': self.bias,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation_name,
            'residual_init_method': self.init_method,
        }

    def _build_reconstruction_layer(self) -> nn.Conv2d:
        """Build the final reconstruction convolution layer."""
        # Calculate padding for reconstruction layer
        if isinstance(self.reconstruction_kernel_size, int):
            padding = self.reconstruction_kernel_size // 2
        else:
            padding = tuple(k // 2 for k in self.reconstruction_kernel_size)

        return nn.Conv2d(
            self.output_channels,
            self.output_channels,
            kernel_size=self.reconstruction_kernel_size,
            padding=padding,
            bias=self.bias
        )

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

    def _initialize_weights(self) -> None:
        """Initialize weights according to the specified method."""
        if self.init_method == 'kaiming':
            self.apply(WeightInitializer.kaiming_normal_)
        elif self.init_method == 'xavier':
            self.apply(WeightInitializer.xavier_uniform_)

        # Always properly initialize batch norm
        if self.use_batch_norm:
            self.apply(WeightInitializer.init_batch_norm_)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation with shape
            (batch_size, bottleneck_channels, height, width).

        Returns
        -------
        torch.Tensor
            Reconstructed output with shape
            (batch_size, output_channels, height', width') where height'
            and width' are restored to approximate original input dimensions.
        """
        # Process through decoder start
        x = self.decoder_start(z)

        # Process through decoder blocks
        for block in self.blocks:
            x = block(x)

        # Final reconstruction
        x = self.reconstruction(x)

        return x

    def get_feature_maps(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Get intermediate feature maps from each decoder block.

        Useful for visualization and debugging.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation.

        Returns
        -------
        list of torch.Tensor
            Feature maps after each decoder block.
        """
        feature_maps = []

        # Process through decoder start
        x = self.decoder_start(z)
        feature_maps.append(x.clone())

        # Process through decoder blocks
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

        # Apply decoder start (changes channels but not spatial dims)
        batch_size, _, height, width = current_shape
        first_channels = (
            self.encoder_blocks[-1].out_channels
            if self.encoder_blocks
            else self.bottleneck_channels
        )
        current_shape = (batch_size, first_channels, height, width)

        # Apply each decoder block
        for block in self.blocks:
            current_shape = block.get_output_shape(current_shape)

        # Final reconstruction doesn't change shape
        return current_shape

    @classmethod
    def from_encoder(
            cls,
            encoder_blocks: list[EncoderBlock],
            bottleneck_channels: int,
            output_channels: int,
            **kwargs
    ) -> 'BlockBasedDecoder':
        """Create decoder that mirrors the given encoder blocks.

        Parameters
        ----------
        encoder_blocks : list of EncoderBlock
            Encoder blocks to mirror.
        bottleneck_channels : int
            Number of channels from encoder bottleneck.
        output_channels : int
            Number of output channels.
        **kwargs
            Additional arguments for decoder configuration.

        Returns
        -------
        BlockBasedDecoder
            Configured decoder instance.
        """
        return cls(
            output_channels=output_channels,
            encoder_blocks=encoder_blocks,
            bottleneck_channels=bottleneck_channels,
            **kwargs
        )

    def __repr__(self) -> str:
        """String representation of the BlockBasedDecoder."""
        return (f"BlockBasedDecoder("
                f"output_channels={self.output_channels}, "
                f"num_blocks={len(self.blocks)}, "
                f"bottleneck_channels={self.bottleneck_channels}, "
                f"upsampling_mode='{self.upsampling_mode}')")
